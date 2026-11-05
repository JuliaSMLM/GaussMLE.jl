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

# Real ROI from Genmab A431 Cell_01 block 1 (roi 418593, sensor corner (168, 248), frame
# 2160), raw ADU with its pixels' calibration, box 9 px. Its Fisher matrix is singular to
# working precision (Float64 Jacobi condition 2.5e12), but the Float32 sum over the box's
# pixels lifts its smallest Cholesky pivot to 1.65e-6, above N * eps (5.96e-7); it
# reported sigma_x = 76.95 um.
const SR_DATA = Float32[239.0 304.0 286.0 239.0 285.0 238.0 262.0 292.0 298.0; 266.0 276.0 256.0 237.0 267.0 258.0 264.0 281.0 280.0; 240.0 265.0 216.0 282.0 279.0 260.0 299.0 252.0 297.0; 259.0 227.0 215.0 248.0 276.0 269.0 253.0 261.0 282.0; 233.0 229.0 223.0 222.0 263.0 269.0 253.0 231.0 295.0; 217.0 288.0 242.0 248.0 245.0 266.0 259.0 275.0 284.0; 219.0 243.0 224.0 213.0 260.0 260.0 301.0 233.0 298.0; 201.0 229.0 259.0 242.0 291.0 283.0 290.0 303.0 264.0; 295.0 317.0 364.0 406.0 453.0 420.0 340.0 300.0 314.0]
const SR_OFFSET = Float32[99.886 100.447 98.774 100.151 100.875 99.773 100.605 100.454 100.304; 99.672 100.606 100.215 99.912 100.089 99.616 100.116 99.876 100.513; 99.848 100.563 100.065 100.624 100.089 99.373 100.348 100.479 100.05; 99.941 100.971 100.211 101.376 99.713 98.908 100.776 100.254 99.757; 99.583 100.255 100.523 99.266 100.259 99.349 100.387 100.117 99.805; 99.626 100.433 100.46 99.606 99.895 99.72 100.282 99.898 100.054; 99.993 101.602 100.183 99.465 101.041 99.057 100.789 100.475 100.095; 100.378 100.194 100.134 100.357 100.252 99.928 100.336 100.898 99.995; 100.856 100.888 101.194 100.351 101.388 100.303 101.451 100.367 100.277]
const SR_GAIN = Float32[0.46621233 0.474723 0.45119497 0.41218513 0.4647713 0.4697526 0.47062102 0.4624116 0.4554165; 0.4631929 0.4685913 0.46302402 0.44453186 0.45360687 0.44588664 0.46012935 0.47542092 0.45555153; 0.45681858 0.4633494 0.47661328 0.4566014 0.45407924 0.4539269 0.45340887 0.4647246 0.4513628; 0.46748132 0.47376028 0.47446018 0.4455296 0.45627993 0.44828364 0.47710994 0.4646304 0.4617571; 0.46036586 0.4712273 0.45520818 0.4506867 0.4662369 0.48422676 0.4673716 0.43826854 0.47857642; 0.48228973 0.47409475 0.47105247 0.44163233 0.4529046 0.47335687 0.44462264 0.4637571 0.46336463; 0.46954173 0.46251482 0.42103297 0.45149386 0.45590505 0.45474276 0.45488662 0.4578416 0.47211367; 0.46654704 0.44949728 0.4656201 0.45803466 0.4618092 0.46196458 0.45332965 0.4549292 0.46309826; 0.47801366 0.46622038 0.46928084 0.45217627 0.44712916 0.47418115 0.4518273 0.4624259 0.45957863]
const SR_READNOISE = Float32[0.7487859 0.7493531 1.3496594 0.5536215 0.7081774 0.7480724 0.7173378 0.79755694 1.9399747; 1.0594548 0.8031531 0.5754023 0.7294864 0.6742275 0.70515263 1.266661 1.8828349 2.195882; 0.68692595 0.76202625 0.61007005 0.8752344 2.1567369 0.7206446 0.8596235 0.7650732 0.8105345; 1.5418642 1.110023 0.6061432 1.0247103 0.81312776 0.8967398 0.809223 0.7492088 1.2623311; 0.7904409 1.0773686 0.75270844 0.8152994 0.71962446 0.74916184 0.92854977 0.63562757 0.6863896; 0.7522086 0.8261437 1.4595059 1.03329 1.2326494 0.7694401 0.67088765 1.0376766 0.7844351; 0.717538 0.89605707 0.53508013 0.7157693 0.90624046 1.1515118 0.802664 0.7151457 0.8137528; 1.1692942 0.6100937 0.57055986 0.7757449 0.9845343 0.75506645 0.7971647 0.90318483 1.1689326; 0.7008095 1.8144609 0.679154 0.6920147 0.8442378 0.7745064 2.1163423 0.69341457 0.75585926]

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

    @testset "a real ROI with a Fisher matrix singular to accumulation error" begin
        camera = SCMOSCamera(
            9, 9, 0.0978f0, SR_READNOISE; offset = SR_OFFSET, gain = SR_GAIN
        )
        batch = ROIBatch(
            reshape(SR_DATA, 9, 9, 1), Int32[1], Int32[1], Int32[1], camera
        )
        smld, _ = fit(batch; psf_model = GaussianXYNBS(), iterations = 20, backend = :cpu)
        e = only(smld.emitters)
        @test all(==(Inf32), emitter_sigmas(e))
        @test isfinite(e.x) && isfinite(e.y) && isfinite(e.photons)
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
