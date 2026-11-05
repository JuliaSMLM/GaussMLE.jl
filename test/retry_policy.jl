# The GPU retry loop's policy, driven on the CPU by a fake vendor whose "GPU" backend is the
# CPU: a deterministic error rethrows under :gpu and falls back to the CPU at once under
# :auto, a resource error is retried until the deadline, an interrupt always rethrows.
using Test, GaussMLE, SMLMData, Logging

struct FakeResourceError <: Exception end
struct FakeOOMError <: Exception end
Base.@kwdef struct FakeVendor <: GaussMLE.GPUVendor
    failure::Any = nothing  # thrown by _activate_gpu!, standing in for the failing GPU call
    oom_above::Int = typemax(Int)  # out of memory when a batch exceeds this many ROIs
    cleanup_failure::Any = nothing  # thrown by the cleanup after a failure
    totals::Vector{Int} = Int[]  # the fake devices' total memory
    # thrown once the context is live (by _ka_gpu_backend), and whether the out-of-memory
    # check runs there instead of in _activate_gpu!
    after_activate::Any = nothing
    oom_after_activate::Bool = false
    poll_failure::Any = nothing  # thrown by the poll
    # the poll reports "cannot fit even idle" for batches above this many ROIs
    never_fits_above::Int = typemax(Int)
end
const ACTIVATE_CALLS = Ref(0)
const POLL_CALLS = Ref(0)
const RELEASE_CALLS = Ref(0)
const RECLAIM_CALLS = Ref(0)
const POLLED_BYTES = Ref(0)
const BOX = 7

function GaussMLE._poll_gpu(v::FakeVendor, backend, bytes; timeout, on_wait = nothing)
    sleep(0.02)
    POLL_CALLS[] += 1
    v.poll_failure === nothing || throw(v.poll_failure)
    POLLED_BYTES[] = bytes
    if v.never_fits_above != typemax(Int) &&
            bytes > GaussMLE.estimate_batch_memory(v.never_fits_above, BOX, 4)
        return (-1, false, false, 1234)
    end
    return (0, true, false, nothing)
end
function GaussMLE._activate_gpu!(v::FakeVendor, i)
    ACTIVATE_CALLS[] += 1
    v.failure === nothing || throw(v.failure)
    v.oom_after_activate || check_fake_oom(v)
    return nothing
end
function check_fake_oom(v::FakeVendor)
    v.oom_above == typemax(Int) && return nothing  # no limit (the estimate would overflow)
    limit = GaussMLE.estimate_batch_memory(v.oom_above, BOX, 4)
    POLLED_BYTES[] > limit && throw(FakeOOMError())
    return nothing
end
function GaussMLE._release_gpu_context(v::FakeVendor, i)
    RELEASE_CALLS[] += 1
    v.cleanup_failure === nothing || throw(v.cleanup_failure)
    return nothing
end
GaussMLE._is_resource_error(::FakeVendor, e) = e isa FakeResourceError || e isa FakeOOMError
GaussMLE._is_oom_error(::FakeVendor, e) = e isa FakeOOMError
function GaussMLE._ka_gpu_backend(v::FakeVendor)
    v.after_activate === nothing || throw(v.after_activate)
    v.oom_after_activate && check_fake_oom(v)
    return GaussMLE.KernelAbstractions.CPU()
end
function GaussMLE._reclaim_gpu(v::FakeVendor)
    RECLAIM_CALLS[] += 1
    v.cleanup_failure === nothing || throw(v.cleanup_failure)
    return nothing
end
GaussMLE._active_device_id(::FakeVendor) = 0
GaussMLE._gpu_total_memory(v::FakeVendor) = v.totals

# Run the loop against `vendor`; return the thrown exception or the fit's backend, the
# batch count, the activate and release counts, the elapsed seconds and the warnings.
function run_retry_loop(vendor, backend; timeout = 0.4)
    psf_model = GaussianXYNB(0.13f0)
    batch = GaussMLE.generate_roi_batch(
        IdealCamera(32, 32, 0.1), psf_model; n_rois = 4, roi_size = BOX, seed = 1
    )
    fitter = GaussMLEConfig(;
        psf_model, backend, auto_timeout = timeout, gpu_timeout = timeout
    )
    n_fits = size(batch.data, 3)
    results = Matrix{Float32}(undef, 4, n_fits)
    uncertainties = similar(results)
    covariances = Matrix{Float32}(undef, 3, n_fits)
    log_likelihoods = Vector{Float32}(undef, n_fits)
    chi2 = Vector{Float32}(undef, n_fits)
    ACTIVATE_CALLS[] = 0
    POLL_CALLS[] = 0
    RELEASE_CALLS[] = 0
    RECLAIM_CALLS[] = 0
    logger = Test.TestLogger()
    outcome = nothing
    n_batches = 0
    elapsed = @elapsed with_logger(logger) do
        outcome = try
            out = GaussMLE._run_mle_kernel!(
                results, uncertainties, covariances, log_likelihoods, chi2,
                convert(Array{Float32, 3}, batch.data),
                GaussMLE.to_pixel_units(psf_model, 0.1f0), Val(false),
                zeros(Float32, 7, 7), batch.x_corners, batch.y_corners, fitter;
                vendor
            )
            n_batches = out[4]
            out[1]
        catch e
            e
        end
    end
    warnings = count(r -> r.level == Logging.Warn, logger.logs)
    return (;
        outcome, activations = ACTIVATE_CALLS[], releases = RELEASE_CALLS[],
        reclaims = RECLAIM_CALLS[], polls = POLL_CALLS[], elapsed, warnings, n_batches,
    )
end

@testset "GPU retry policy" begin
    @testset "deterministic error under :gpu rethrows at once" begin
        r = run_retry_loop(FakeVendor(; failure = ErrorException("compile error")), :gpu)
        @test r.outcome isa ErrorException
        @test r.activations == 1
        @test r.releases == 1
        @test r.elapsed < 0.4
    end

    @testset "deterministic error under :auto falls back to the CPU with one warning" begin
        r = run_retry_loop(FakeVendor(; failure = ErrorException("compile error")), :auto)
        @test r.outcome === :cpu
        @test r.activations == 1
        @test r.releases == 1
        @test r.warnings == 1
        @test r.elapsed < 0.4 + 5  # the CPU fit itself, not the 0.4 s timeout
    end

    @testset "resource error is retried until the deadline" begin
        r = run_retry_loop(FakeVendor(; failure = FakeResourceError()), :gpu)
        @test r.outcome isa Exception
        @test r.activations > 1
        @test r.releases == r.activations
        @test r.elapsed >= 0.4

        r = run_retry_loop(FakeVendor(; failure = FakeResourceError()), :auto)
        @test r.outcome === :cpu
        @test r.activations > 1
        @test r.elapsed >= 0.4
    end

    @testset "InterruptException rethrows in both modes" begin
        for backend in (:gpu, :auto)
            r = run_retry_loop(FakeVendor(; failure = InterruptException()), backend)
            @test r.outcome isa InterruptException
            @test r.activations == 1
        end
    end

    @testset "an interrupt in the poll or the live context rethrows under :auto" begin
        r = run_retry_loop(FakeVendor(; poll_failure = InterruptException()), :auto)
        @test r.outcome isa InterruptException  # no CPU fit ran
        @test r.activations == 0

        vendor = FakeVendor(; after_activate = InterruptException())
        r = run_retry_loop(vendor, :auto)
        @test r.outcome isa InterruptException
        @test r.activations == 1
        @test r.reclaims == 1
    end

    @testset "a request no idle device can hold halves the batch without waiting" begin
        for backend in (:gpu, :auto)
            r = run_retry_loop(FakeVendor(; never_fits_above = 1), backend; timeout = 30.0)
            @test r.outcome === :gpu
            @test r.n_batches == 4  # 4 -> 2 -> 1 ROIs per batch
            @test r.polls == 3
            @test r.activations == 1
            @test r.elapsed < 10  # no wait for the 30 s timeout
        end
    end

    @testset "one ROI no idle device can hold errors (:gpu) or uses the CPU (:auto)" begin
        vendor = FakeVendor(; never_fits_above = 0)
        r = run_retry_loop(vendor, :gpu; timeout = 30.0)
        @test r.outcome isa ErrorException
        msg = sprint(showerror, r.outcome)
        @test occursin("with margin", msg)
        @test occursin("idle GPU", msg)
        @test occursin("batch of 1 ROI", msg)
        @test r.activations == 0
        @test r.elapsed < 10

        r = run_retry_loop(vendor, :auto; timeout = 30.0)
        @test r.outcome === :cpu
        @test r.activations == 0
        @test r.warnings == 1
        @test r.elapsed < 10
    end

    @testset "the idle-device decision" begin
        never_fits = GaussMLE._never_fits
        @test never_fits(900, 1000, 1000, false)  # idle, free below the request
        @test !never_fits(900, 1000, 1000, true)  # another process may release it
        @test !never_fits(900, 1000, 1000, nothing)  # unknown: keep waiting
        @test !never_fits(1000, 1200, 1000, false)  # free covers the request
        @test never_fits(900, 999, 1000, true)  # total below the request, any process
        @test never_fits(900, 999, 1000, nothing)
    end

    @testset "out of memory halves the batch and retries at once" begin
        for backend in (:gpu, :auto)
            r = run_retry_loop(FakeVendor(; oom_above = 2), backend; timeout = 30.0)
            @test r.outcome === :gpu
            @test r.n_batches == 2  # 4 ROIs failed, then 2 ROIs per batch
            @test r.activations == 2
            @test r.releases == 1
            @test r.elapsed < 10  # no wait for the 30 s timeout
        end
    end

    @testset "a one-ROI out-of-memory error waits for the deadline" begin
        r = run_retry_loop(FakeVendor(; oom_above = 0), :gpu)
        @test r.outcome isa FakeOOMError
        @test r.elapsed >= 0.4
        @test r.activations > 3  # 4 -> 2 -> 1 ROIs, then re-polls of the one-ROI batch

        r = run_retry_loop(FakeVendor(; oom_above = 0), :auto)
        @test r.outcome === :cpu
        @test r.elapsed >= 0.4
    end

    @testset "one ROI beyond every device's memory fails or uses the CPU at once" begin
        @test GaussMLE._exceeds_device_memory(1000, [1000, 1400])
        @test !GaussMLE._exceeds_device_memory(1000, [1000, 1600])
        @test !GaussMLE._exceeds_device_memory(1000, Int[])

        vendor = FakeVendor(; totals = [100, 200])
        r = run_retry_loop(vendor, :gpu; timeout = 30.0)
        @test r.outcome isa ErrorException
        @test occursin("total memory", sprint(showerror, r.outcome))
        @test r.activations == 0
        @test r.elapsed < 10

        r = run_retry_loop(vendor, :auto; timeout = 30.0)
        @test r.outcome === :cpu
        @test r.activations == 0
        @test r.warnings == 1
        @test r.elapsed < 10
    end

    @testset "a failing cleanup never replaces the original error" begin
        vendor = FakeVendor(;
            failure = ErrorException("compile error"),
            cleanup_failure = ArgumentError("poisoned context")
        )
        r = run_retry_loop(vendor, :gpu)
        @test r.outcome isa ErrorException
        @test r.outcome.msg == "compile error"
        @test r.releases == 1

        r = run_retry_loop(vendor, :auto)
        @test r.outcome === :cpu
        @test r.releases == 1
        @test r.warnings == 2  # the cleanup failure and the fallback
    end

    # Once the context is live, a failure keeps it (releasing CUDA's primary context breaks
    # every later call in the process with CONTEXT_IS_DESTROYED) and only reclaims memory.
    @testset "a failure after the context is live keeps the context" begin
        vendor = FakeVendor(; after_activate = ErrorException("launch failed"))
        r = run_retry_loop(vendor, :auto)
        @test r.outcome === :cpu
        @test r.releases == 0
        @test r.reclaims == 1

        r = run_retry_loop(vendor, :gpu)
        @test r.outcome isa ErrorException
        @test r.releases == 0

        vendor = FakeVendor(;
            after_activate = ErrorException("launch failed"),
            cleanup_failure = ArgumentError("poisoned context")
        )
        r = run_retry_loop(vendor, :gpu)
        @test r.outcome isa ErrorException
        @test r.outcome.msg == "launch failed"
    end

    @testset "out of memory in a live context halves the batch and keeps the context" begin
        vendor = FakeVendor(; oom_above = 2, oom_after_activate = true)
        r = run_retry_loop(vendor, :gpu; timeout = 30.0)
        @test r.outcome === :gpu
        @test r.n_batches == 2
        @test r.releases == 0
        @test r.reclaims == 2  # after the failure and after the fit
        @test r.elapsed < 10
    end
end
