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
    # seconds the poll waits (reporting to on_wait first) before it finds the device
    poll_wait::Float64 = 0.0
end
const ACTIVATE_CALLS = Ref(0)
const POLL_CALLS = Ref(0)
const RELEASE_CALLS = Ref(0)
const RECLAIM_CALLS = Ref(0)
const POLLED_BYTES = Ref(0)
const BOX = 7

function GaussMLE._poll_gpu(v::FakeVendor, backend, bytes; timeout, on_wait = nothing)
    sleep(0.02)
    if v.poll_wait > 0
        on_wait === nothing || on_wait(0.0, 0, bytes)
        sleep(v.poll_wait)
    end
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
    # Compile the fake vendor's kernel launch and its out-of-memory recovery once, so the
    # timed testsets below measure the retry loop and not first compilation (on Julia 1.11
    # with bounds checking, as Pkg.test runs it, the out-of-memory path alone took 110 s).
    @test run_retry_loop(FakeVendor(), :gpu; timeout = 30.0).outcome === :gpu
    @test run_retry_loop(FakeVendor(; oom_above = 2), :gpu; timeout = 30.0).outcome === :gpu

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
        vendor = FakeVendor(; failure = FakeResourceError())
        r = run_retry_loop(vendor, :gpu; timeout = 0.8)
        @test r.outcome isa Exception
        @test r.activations > 1
        @test r.releases == r.activations
        @test r.elapsed >= 0.8

        r = run_retry_loop(vendor, :auto; timeout = 0.8)
        @test r.outcome === :cpu
        @test r.activations > 1
        @test r.elapsed >= 0.8
    end

    # A resource error that is not out of memory, on a device the poll reports available at
    # once, pauses 0.5 s before each new poll instead of spinning; one warning per batch
    # size.
    @testset "a non-OOM resource error on an available device pauses between tries" begin
        r = run_retry_loop(FakeVendor(; failure = FakeResourceError()), :gpu; timeout = 1.3)
        @test r.outcome isa FakeResourceError || r.outcome isa ErrorException
        @test 2 <= r.activations <= 4  # about timeout / 0.5
        @test r.warnings == 1
        @test r.elapsed >= 1.3
    end

    # A poll that had to wait means the device was busy: the one-ROI out-of-memory count
    # restarts and no pause is added, so the fit keeps retrying until the deadline.
    @testset "a poll that waited resets the one-ROI out-of-memory count" begin
        vendor = FakeVendor(; oom_above = 0, poll_wait = 0.1)
        r = run_retry_loop(vendor, :gpu; timeout = 1.0)
        @test r.outcome isa Exception
        @test r.activations > 5  # 4 -> 2 -> 1 ROIs, then past three one-ROI failures
        @test r.elapsed >= 1.0
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

    # A one-ROI failure that no wait clears (a hard memory limit below the table plus one
    # ROI) ends after three tries in a row on an available device, 0.5 s apart, even with
    # no timeout; the warning is logged once per batch size (4, 2 and 1 ROIs).
    @testset "a permanent one-ROI out-of-memory error ends without a timeout" begin
        r = run_retry_loop(FakeVendor(; oom_above = 0), :gpu; timeout = Inf)
        @test r.outcome isa FakeOOMError
        @test r.activations == 5  # 4 -> 2 -> 1 ROIs, then two more one-ROI tries
        @test r.warnings == 3
        @test 0.9 <= r.elapsed < 10  # two 0.5 s pauses

        r = run_retry_loop(FakeVendor(; oom_above = 0), :auto; timeout = Inf)
        @test r.outcome === :cpu
        @test r.activations == 5
        @test r.warnings == 4  # the three above and the fall back to the CPU
        @test 0.9 <= r.elapsed < 10 + 1
    end

    @testset "a one-ROI out-of-memory error stops at the deadline" begin
        r = run_retry_loop(FakeVendor(; oom_above = 0), :gpu; timeout = 0.4)
        @test r.outcome isa Exception
        @test r.elapsed >= 0.4
        @test r.elapsed < 10

        r = run_retry_loop(FakeVendor(; oom_above = 0), :auto; timeout = 0.4)
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
