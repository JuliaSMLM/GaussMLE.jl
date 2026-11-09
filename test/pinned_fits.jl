# Fits that did not localize an emitter report infinite uncertainties: a fit pinned on a
# position bound (applied on the host after the kernel) and a Fisher matrix singular to
# working precision (decided in the kernel by `_invert_fisher`). The ROIs below are
# rendered noise-free by SMLMSim: a Gaussian PSF on a 9 x 9 sCMOS camera (0.1 um pixels,
# gain 1, offset 0, read noise 1 e-) with the emitter at or beyond the box edge, fitted
# with GaussianXYNBS. The first (x 0.0, y 10.6 px, 2766 photons, PSF sigma 0.07 um,
# background 414 per pixel) is fitted with y on its upper bound (11 px), and its Fisher
# matrix is singular to working precision as well. A fit capped before its convergence test
# also reports Inf, so each of these is refitted with a cap of 1000 passes: the same result
# shows the fit stopped on its test and the Inf is the path's own.
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
        sim_roi(0.0f0, 10.6f0, 2766, 414.0, 0.07), Int32[1], Int32[1], Int32[1], SIM_CAMERA
    )
    emitter_sigmas(e) = (e.σ_x, e.σ_y, e.σ_photons, e.σ_bg, e.σ_σ)
    # The fit stopped on its convergence test: a cap of 1000 passes returns the same fit.
    function stopped_before_cap(batch, e)
        smld, _ = fit(batch; psf_model = GaussianXYNBS(), backend = :cpu, iterations = 1000)
        long = only(smld.emitters)
        return (long.x, long.y, long.photons, long.bg, long.σ) ==
            (e.x, e.y, e.photons, e.bg, e.σ)
    end

    @testset "a fit pinned on its y bound reports Inf" begin
        smld, _ = fit(zp_batch; psf_model = GaussianXYNBS(), backend = :cpu)
        e = only(smld.emitters)
        @test e.y ≈ (11 - 0.5) * 0.1 atol = 1.0e-6  # the upper bound, 11 px, in microns
        @test all(==(Inf32), emitter_sigmas(e))
        @test e.σ_xy == 0
        @test isfinite(e.x) && isfinite(e.y) && isfinite(e.photons) && isfinite(e.pvalue)
        @test stopped_before_cap(zp_batch, e)
    end

    @testset "a ROI with a Fisher matrix singular to accumulation error" begin
        # The emitter is just beyond the first row (x 8.8, y -0.9 px, PSF sigma 0.05 um,
        # 57163 photons, background 9.8). No position ends on a bound, but the Fisher matrix
        # is singular to working precision. At the fit's end point its smallest Jacobi-scaled
        # Cholesky pivot is about 21 to 23 eps on the CPU paths of KernelAbstractions 0.9 and
        # 0.10: above N * eps, where that tolerance reports sigma_x = 72 nm (finite, and
        # plausible) and sigma_y = 1.5 um, and at or below n_pixels * eps, the accumulation
        # error of the Float32 sum over the box's pixels. The pivot is rounding noise and
        # moves with the platform; the constructed matrix in the `_invert_fisher` test below
        # holds that band on every path.
        roi = sim_roi(8.8f0, -0.9f0, 57163, 9.8, 0.05)
        batch = ROIBatch(roi, Int32[1], Int32[1], Int32[1], SIM_CAMERA)
        smld, _ = fit(batch; psf_model = GaussianXYNBS(), backend = :cpu)
        e = only(smld.emitters)
        lo, hi = (-1 - 0.5) * 0.1, (11 - 0.5) * 0.1  # the x and y bounds in microns
        @test lo + 1.0e-3 < e.x < hi - 1.0e-3 && lo + 1.0e-3 < e.y < hi - 1.0e-3
        @test all(==(Inf32), emitter_sigmas(e))
        @test isfinite(e.x) && isfinite(e.y) && isfinite(e.photons)
        @test stopped_before_cap(batch, e)
    end

    @testset "parameters the caller holds fixed are not pinned" begin
        box = 9
        psf = GaussMLE.to_pixel_units(GaussianXYNB(0.13f0), 0.1f0)
        theta = GaussMLE.Params{4}(5.0f0, 5.0f0, 2000.0f0, 10.0f0)
        mu = Float32[GaussMLE.evaluate_psf(psf, i, j, theta) for i in 1:box, j in 1:box]
        Random.seed!(3)
        data = reshape(Float32.(poisson_noise(mu)), box, box, 1)
        batch = ROIBatch(data, Int32[1], Int32[1], Int32[1], IdealCamera(32, 32, 0.1f0))
        # x and y held at the truth: lower == upper, inside the box
        constraints = GaussMLE.ParameterConstraints{4}(
            GaussMLE.Params{4}(5.0f0, 5.0f0, 1.0f0, 0.01f0),
            GaussMLE.Params{4}(5.0f0, 5.0f0, Inf32, Inf32),
            GaussMLE.Params{4}(1.0f0, 1.0f0, 100.0f0, 2.0f0),
        )
        smld, _ = fit(batch; backend = :cpu, iterations = 50, constraints)
        e = only(smld.emitters)
        @test isfinite(e.x) && isfinite(e.y)
        @test isfinite(e.σ_photons) && isfinite(e.σ_bg)
    end

    @testset "the kernel itself reports Inf, never a zero, for a singular matrix" begin
        # Bypasses the host step. The emitter is beyond the corner past the last row and
        # column (x 11.0, y 10.5 px, PSF sigma 0.07 um, 44955 photons, background 2.1); the
        # fit stops inside its bounds (just outside the box) with a Fisher matrix singular in
        # Float32 (at its end point the Jacobi-scaled Cholesky meets a negative pivot on the
        # CPU paths of KernelAbstractions 0.9 and 0.10), and the kernel alone must not turn
        # the failed inversion into a number.
        psf = GaussMLE._model_for_camera(GaussianXYNBS(), SIM_CAMERA)
        variance_map = GaussMLE.extract_variance_map(SIM_CAMERA, Float32)
        data = Float32.(
            GaussMLE.to_electrons(
                sim_roi(11.0f0, 10.5f0, 44955, 2.1, 0.07), SIM_CAMERA, Int32[1], Int32[1]
            )
        )
        constraints = GaussMLE._resolve_constraints(nothing, psf, 9)
        ka = GaussMLE.KernelAbstractions
        kernel = GaussMLE.unified_gaussian_mle_kernel!(ka.CPU(), 64)
        function run_kernel(iterations)
            results = zeros(Float32, 5, 1)
            uncertainties = zeros(Float32, 5, 1)
            covariances = zeros(Float32, 3, 1)
            llr = zeros(Float32, 1)
            chi2 = zeros(Float32, 1)
            kernel(
                results, uncertainties, covariances, llr, chi2, data, psf, Val(true),
                variance_map, Int32[1], Int32[1], constraints, iterations; ndrange = 1
            )
            ka.synchronize(ka.CPU())
            return results, uncertainties, covariances
        end
        results, uncertainties, covariances = run_kernel(GaussMLE._default_iterations(psf))
        @test all(k -> constraints.lower[k] < results[k, 1] < constraints.upper[k], 1:2)
        @test all(==(Inf32), uncertainties)
        @test all(iszero, covariances)
        @test first(run_kernel(1000)) == results  # stopped on its test, not at the cap
    end

    @testset "_invert_fisher fails singular matrices and inverts a good one" begin
        singular = SMatrix{3, 3, Float32}(1, 0, 1, 0, 1, 1, 1, 1, 2)  # row 3 = row 1 + row 2
        ok, _ = GaussMLE._invert_fisher(singular, 81)
        @test !ok
        # Singular to Float32 precision (condition about 1.7e7): the plain Cholesky passes
        # with pivot 2^-22, below 81 * eps(Float32), the tolerance of a 9 x 9 box, and its
        # inverse would be rounding.
        a = 1.0f0 - eps(Float32)
        near = SMatrix{3, 3, Float32}(1, a, 0, a, 1, 0, 0, 0, 1)
        @test first(GaussMLE.static_cholesky_decomposition(near))
        ok, _ = GaussMLE._invert_fisher(near, 81)
        @test !ok
        # Pivot 1 - b^2 of about 20 eps: above N * eps (N = 3 parameters), so a tolerance of
        # N * eps passes it, and at or below 81 * eps, the accumulation error of a 9 x 9 box.
        b = sqrt(1.0f0 - 20 * eps(Float32))
        band = SMatrix{3, 3, Float32}(1, b, 0, b, 1, 0, 0, 0, 1)
        @test 3 * eps(Float32) < 1 - b^2 <= 81 * eps(Float32)
        @test first(GaussMLE.static_cholesky_decomposition(band, 3 * eps(Float32)))
        ok, _ = GaussMLE._invert_fisher(band, 81)
        @test !ok
        good = SMatrix{3, 3, Float32}(4, 1, 0, 1, 3, 1, 0, 1, 2)
        ok, inverse = GaussMLE._invert_fisher(good, 81)
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
