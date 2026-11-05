# Fits that did not localize an emitter report infinite uncertainties: a fit pinned on a
# position bound (applied on the host after the kernel) and a Fisher matrix singular to
# working precision (decided in the kernel by `_invert_fisher`). The real ROI is from
# Genmab A431 Cell_01 (sensor corner (1, 39), frame 3601): raw ADU with its pixels'
# calibration; before the fix its y ended at -1 px with sigma_y = 0 and sigma_x = 631 um.
using Test, GaussMLE, SMLMData, StaticArrays, Random
using SMLMSim: poisson_noise

# Box 9 px, pixel 0.0978 um.
const ZP_DATA = Float32[716.0 510.0 392.0 312.0 280.0 278.0 281.0 257.0 281.0; 670.0 543.0 428.0 308.0 262.0 274.0 302.0 247.0 287.0; 551.0 437.0 377.0 287.0 316.0 264.0 287.0 296.0 271.0; 717.0 453.0 308.0 306.0 295.0 241.0 283.0 241.0 232.0; 825.0 474.0 304.0 277.0 287.0 260.0 266.0 238.0 262.0; 769.0 457.0 332.0 291.0 240.0 286.0 283.0 261.0 258.0; 441.0 362.0 295.0 264.0 277.0 248.0 268.0 238.0 271.0; 331.0 336.0 286.0 256.0 275.0 267.0 282.0 238.0 280.0; 353.0 342.0 310.0 281.0 292.0 249.0 256.0 236.0 225.0]
const ZP_OFFSET = Float32[100.705 99.783 100.215 100.247 100.244 100.167 100.639 100.415 99.424; 99.646 99.935 99.302 100.681 99.598 100.141 99.67 100.652 99.532; 100.014 99.597 98.873 101.281 101.028 100.059 99.534 101.24 99.182; 100.083 100.44 99.678 100.112 99.56 99.938 99.952 100.112 98.826; 100.141 99.916 99.823 100.276 100.286 100.511 98.718 100.038 99.143; 100.046 99.51 99.794 99.697 100.396 99.707 100.733 100.62 99.353; 100.253 99.138 99.822 100.425 100.282 99.621 100.168 100.074 98.564; 100.026 100.134 99.924 99.212 100.004 100.93 100.077 100.649 99.246; 99.419 100.572 99.855 100.204 100.452 101.775 100.325 99.95 98.643]
const ZP_GAIN = Float32[0.45346132 0.45548758 0.46271524 0.4689943 0.45672745 0.47330812 0.49367386 0.4583496 0.4646086; 0.4603116 0.46538916 0.47362942 0.4534542 0.47700614 0.47533503 0.47096944 0.4568801 0.4821673; 0.45818076 0.46246377 0.45439678 0.47796506 0.47050267 0.46386614 0.4696913 0.46843383 0.48937374; 0.46651003 0.45750636 0.455104 0.4670465 0.457598 0.47108814 0.48265275 0.47140422 0.46387038; 0.4589299 0.46702105 0.4827733 0.47197807 0.4604989 0.46166423 0.47122008 0.48168036 0.4657985; 0.46099395 0.44662037 0.45344326 0.4583793 0.45922348 0.45800865 0.4732372 0.45578322 0.46313435; 0.44398928 0.4603788 0.46556407 0.47337946 0.45321265 0.4483679 0.4878706 0.46187901 0.45562407; 0.42916545 0.46589038 0.46435687 0.4550515 0.46613026 0.41515443 0.4676286 0.46812704 0.46339643; 0.4591993 0.47661328 0.455087 0.46082973 0.45211196 0.44607425 0.46386892 0.47306365 0.47360882]
const ZP_READNOISE = Float32[0.89710456 1.0048431 1.3805438 0.9358701 0.77176315 1.6470407 0.80626905 0.8723559 0.6873101; 1.004856 0.721758 0.8761834 0.8999869 0.6753292 0.7375385 0.7225126 4.7555814 0.6332208; 0.7450794 0.8165732 2.4534874 3.27854 0.8406609 0.7186073 1.1055564 7.5294213 0.7680731; 0.682165 0.6945508 0.676027 0.84517866 0.6843545 0.65717846 0.67943877 0.73694783 1.2997952; 0.7031454 0.7512433 0.7378718 0.8143225 0.8262606 0.7155464 3.1781309 0.8873843 0.8611578; 0.5542152 0.661557 0.74760294 0.6558559 0.989175 0.72383577 1.4376547 0.8877123 0.88458043; 0.70976394 0.95235693 0.81618434 0.7569507 1.2325295 0.8444232 0.7298545 0.8608811 2.96752; 0.6820738 0.8416142 0.8649623 4.011866 0.95803165 1.5887862 0.76607573 0.7537283 0.67082006; 0.8428282 0.8152371 0.69021446 0.93680304 0.74514896 0.87255245 0.70037496 1.0916216 3.9419715]

@testset "Fits on a bound and singular Fisher matrices" begin
    ZP_CAMERA = SCMOSCamera(
        9, 9, 0.0978f0, ZP_READNOISE; offset = ZP_OFFSET, gain = ZP_GAIN
    )
    zp_batch = ROIBatch(
        reshape(ZP_DATA, 9, 9, 1), Int32[1], Int32[1], Int32[1], ZP_CAMERA
    )
    emitter_sigmas(e) = (e.σ_x, e.σ_y, e.σ_photons, e.σ_bg, e.σ_σ)

    @testset "the real zero-precision ROI pins y and reports Inf" begin
        smld, _ = fit(zp_batch; psf_model = GaussianXYNBS(), iterations = 20, backend = :cpu)
        e = only(smld.emitters)
        @test all(==(Inf32), emitter_sigmas(e))
        @test e.σ_xy == 0
        @test isfinite(e.x) && isfinite(e.y) && isfinite(e.photons) && isfinite(e.pvalue)
    end

    @testset "the kernel itself reports Inf, never a zero, for the singular matrix" begin
        # Bypasses the host step: the fit's y sits on its bound, but the kernel alone must
        # not turn the failed inversion into zero precision.
        psf = GaussMLE._model_for_camera(GaussianXYNBS(), ZP_CAMERA)
        variance_map = GaussMLE.extract_variance_map(ZP_CAMERA, Float32)
        data = Float32.(
            GaussMLE.to_electrons(zp_batch.data, ZP_CAMERA, Int32[1], Int32[1])
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
        @test results[2, 1] == constraints.lower[2]
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
