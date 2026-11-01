# The GPU retry loop's policy, driven on the CPU by a fake vendor whose "GPU" backend is the
# CPU: a deterministic error rethrows under :gpu and falls back to the CPU at once under
# :auto, a resource error is retried until the deadline, an interrupt always rethrows.
using Test, GaussMLE, SMLMData, Logging

struct FakeResourceError <: Exception end
struct FakeVendor <: GaussMLE.GPUVendor
    failure::Any  # thrown by _activate_gpu!, standing in for the failing GPU call
end
const ACTIVATE_CALLS = Ref(0)
const RELEASE_CALLS = Ref(0)

GaussMLE._poll_gpu(::FakeVendor, backend, bytes; timeout, on_wait = nothing) =
    (sleep(0.02); (0, true, false))
function GaussMLE._activate_gpu!(v::FakeVendor, i)
    ACTIVATE_CALLS[] += 1
    throw(v.failure)
end
GaussMLE._release_gpu_context(::FakeVendor, i) = (RELEASE_CALLS[] += 1; nothing)
GaussMLE._is_resource_error(::FakeVendor, e) = e isa FakeResourceError

# Run the loop with `failure`; return the thrown exception or the fit's backend, the
# activate and release counts, the elapsed seconds and the warnings logged.
function run_retry_loop(failure, backend; timeout = 0.4)
    psf_model = GaussianXYNB(0.13f0)
    batch = GaussMLE.generate_roi_batch(
        IdealCamera(32, 32, 0.1), psf_model; n_rois = 4, roi_size = 7, seed = 1
    )
    fitter = GaussMLEConfig(;
        psf_model, backend, auto_timeout = timeout, gpu_timeout = timeout
    )
    n_fits = size(batch.data, 3)
    results = Matrix{Float32}(undef, 4, n_fits)
    uncertainties = similar(results)
    covariances = Matrix{Float32}(undef, 3, n_fits)
    log_likelihoods = Vector{Float32}(undef, n_fits)
    ACTIVATE_CALLS[] = 0
    RELEASE_CALLS[] = 0
    logger = Test.TestLogger()
    outcome = nothing
    elapsed = @elapsed with_logger(logger) do
        outcome = try
            GaussMLE._run_mle_kernel!(
                results, uncertainties, covariances, log_likelihoods,
                convert(Array{Float32, 3}, batch.data),
                GaussMLE.to_pixel_units(psf_model, 0.1f0), Val(false),
                zeros(Float32, 7, 7), batch.x_corners, batch.y_corners, fitter;
                vendor = FakeVendor(failure)
            )[1]
        catch e
            e
        end
    end
    warnings = count(r -> r.level == Logging.Warn, logger.logs)
    return (;
        outcome, activations = ACTIVATE_CALLS[], releases = RELEASE_CALLS[],
        elapsed, warnings,
    )
end

@testset "GPU retry policy" begin
    @testset "deterministic error under :gpu rethrows at once" begin
        r = run_retry_loop(ErrorException("compile error"), :gpu)
        @test r.outcome isa ErrorException
        @test r.activations == 1
        @test r.releases == 1
        @test r.elapsed < 0.4
    end

    @testset "deterministic error under :auto falls back to the CPU with one warning" begin
        r = run_retry_loop(ErrorException("compile error"), :auto)
        @test r.outcome === :cpu
        @test r.activations == 1
        @test r.releases == 1
        @test r.warnings == 1
        @test r.elapsed < 0.4 + 5  # the CPU fit itself, not the 0.4 s timeout
    end

    @testset "resource error is retried until the deadline" begin
        r = run_retry_loop(FakeResourceError(), :gpu)
        @test r.outcome isa Exception
        @test r.activations > 1
        @test r.releases == r.activations
        @test r.elapsed >= 0.4

        r = run_retry_loop(FakeResourceError(), :auto)
        @test r.outcome === :cpu
        @test r.activations > 1
        @test r.elapsed >= 0.4
    end

    @testset "InterruptException rethrows in both modes" begin
        for backend in (:gpu, :auto)
            r = run_retry_loop(InterruptException(), backend)
            @test r.outcome isa InterruptException
            @test r.activations == 1
        end
    end
end
