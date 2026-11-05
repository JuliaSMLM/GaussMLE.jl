# Fits that did not localize an emitter report infinite uncertainties: a fit pinned on a
# position bound (applied on the host after the kernel) and a Fisher matrix singular to
# working precision (decided in the kernel by `_invert_fisher`). The ROI below is
# rendered noise-free by SMLMSim: a Gaussian PSF on a 9 x 9 sCMOS camera (0.1 um pixels,
# gain 1, offset 0, read noise 1 e-) with the emitter outside the box. The first (x 1.3,
# y 11.4 px, PSF sigma 0.07 um) is fitted with y on its upper bound (11 px) and the PSF
# width on its lower bound, where its Fisher matrix is singular; before the fix it
# reported zero uncertainty for x, y, photons and width.
using Test, GaussMLE, SMLMData, StaticArrays, Random
using MicroscopePSFs: GaussianPSF
using SMLMSim: gen_images, poisson_noise

const SIM_CAMERA = SCMOSCamera(9, 9, 0.1f0, 1.0f0)

# Noise-free 9 x 9 ROI on SIM_CAMERA of one emitter at pixel position (x, y), pixel centers
# at 1..9.
function sim_roi(x, y, photons, bg, psf_sigma)
    um(p) = (p - 0.5f0) * 0.1f0
    e = Emitter2DFit{Float32}(
        um(x), um(y), Float32(photons), 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0
    )
    images, _ = gen_images(BasicSMLD([e], SIM_CAMERA, 1, 1), GaussianPSF(psf_sigma); bg)
    return reshape(Float32.(images[:, :, 1]), 9, 9, 1)
end

@testset "Fits on a bound and singular Fisher matrices" begin
    zp_batch = ROIBatch(
        sim_roi(1.3f0, 11.4f0, 19280, 4.1, 0.07), Int32[1], Int32[1], Int32[1], SIM_CAMERA
    )
    emitter_sigmas(e) = (e.σ_x, e.σ_y, e.σ_photons, e.σ_bg, e.σ_σ)

    @testset "a zero-precision ROI pins y and reports Inf" begin
        smld, _ = fit(zp_batch; psf_model = GaussianXYNBS(), iterations = 20, backend = :cpu)
        e = only(smld.emitters)
        @test all(==(Inf32), emitter_sigmas(e))
        @test e.σ_xy == 0
        @test isfinite(e.x) && isfinite(e.y) && isfinite(e.photons) && isfinite(e.pvalue)
    end

    @testset "the kernel itself reports Inf, never a zero, for the singular matrix" begin
        # Bypasses the host step: the fit's y sits on its bound, but the kernel alone must
        # not turn the failed inversion into zero precision.
        psf = GaussMLE._model_for_camera(GaussianXYNBS(), SIM_CAMERA)
        variance_map = GaussMLE.extract_variance_map(SIM_CAMERA, Float32)
        data = Float32.(
            GaussMLE.to_electrons(zp_batch.data, SIM_CAMERA, Int32[1], Int32[1])
        )
        results = zeros(Float32, 5, 1)
        uncertainties = zeros(Float32, 5, 1)
        covariances = zeros(Float32, 3, 1)
        llr = zeros(Float32, 1)
        chi2 = zeros(Float32, 1)
        constraints = GaussMLE._resolve_constraints(nothing, psf, 9)
        ka = GaussMLE.KernelAbstractions
        kernel = GaussMLE.unified_gaussian_mle_kernel!(ka.CPU(), 64)
        kernel(
            results, uncertainties, covariances, llr, chi2, data, psf, Val(true),
            variance_map, Int32[1], Int32[1], constraints, 20; ndrange = 1
        )
        ka.synchronize(ka.CPU())
        @test results[2, 1] == constraints.upper[2]
        @test all(==(Inf32), uncertainties)
        @test all(iszero, covariances)
    end

    @testset "_invert_fisher fails singular matrices and inverts a good one" begin
        singular = SMatrix{3, 3, Float32}(1, 0, 1, 0, 1, 1, 1, 1, 2)  # row 3 = row 1 + row 2
        ok, _ = GaussMLE._invert_fisher(singular)
        @test !ok
        # Singular to Float32 precision (condition about 1.7e7): the plain Cholesky passes
        # with pivot 2^-22, below 3 * eps(Float32), and its inverse would be rounding.
        a = 1.0f0 - eps(Float32)
        near = SMatrix{3, 3, Float32}(1, a, 0, a, 1, 0, 0, 0, 1)
        @test first(GaussMLE.static_cholesky_decomposition(near))
        ok, _ = GaussMLE._invert_fisher(near)
        @test !ok
        good = SMatrix{3, 3, Float32}(4, 1, 0, 1, 3, 1, 0, 1, 2)
        ok, inverse = GaussMLE._invert_fisher(good)
        @test ok
        @test Matrix{Float64}(inverse) ≈ inv(Matrix{Float64}(good)) rtol = 1.0e-5
    end

    @testset "an IdealCamera emitter 2 px outside the box edge reports Inf" begin
        box = 9
        psf = GaussMLE.to_pixel_units(GaussianXYNB(0.13f0), 0.1f0)
        truth(x, y) = GaussMLE.Params{4}(x, y, 5000.0f0, 10.0f0)
        mu(p) = Float32[
            GaussMLE.evaluate_psf(psf, i, j, p) for i in 1:box, j in 1:box
        ]
        Random.seed!(11)
        # The first emitter is 2 px beyond the edge (x = 0.5 - 2), the second is centered.
        data = cat(
            Float32.(poisson_noise(mu(truth(-1.5f0, 5.0f0)))),
            Float32.(poisson_noise(mu(truth(5.0f0, 5.0f0)))); dims = 3
        )
        camera = IdealCamera(32, 32, 0.1f0)
        batch = ROIBatch(data, Int32[1, 12], Int32[1, 1], Int32[1, 2], camera)
        smld, _ = fit(batch; backend = :cpu, iterations = 100)
        pinned, centered = smld.emitters
        @test (pinned.σ_x, pinned.σ_y, pinned.σ_photons, pinned.σ_bg) ==
            ntuple(_ -> Inf32, 4)
        @test pinned.σ_xy == 0
        @test isfinite(pinned.x) && isfinite(pinned.photons)
        @test all(isfinite, (centered.σ_x, centered.σ_y, centered.σ_photons, centered.σ_bg))

        # A bound from the caller's constraints counts the same way.
        constraints = GaussMLE.ParameterConstraints{4}(
            GaussMLE.Params{4}(-1.0f0, -1.0f0, 1.0f0, 0.01f0),
            GaussMLE.Params{4}(3.0f0, 11.0f0, Inf32, Inf32),
            GaussMLE.Params{4}(1.0f0, 1.0f0, 100.0f0, 2.0f0),
        )
        smld, _ = fit(batch; backend = :cpu, iterations = 100, constraints)
        @test smld.emitters[2].σ_x == Inf32  # centered at x = 5, held at the upper bound 3
        @test isfinite(smld.emitters[2].x)
    end
end
