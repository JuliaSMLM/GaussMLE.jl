# The CRLB of bright emitters. The Fisher entries span many orders of magnitude (F_xx grows
# like N, F_NN shrinks like 1/N), and an absolute regularizer on the diagonal once
# collapsed sigma_photons below sqrt(N) from N = 1e5. A noiseless box fits to the truth, so
# the reported uncertainties and covariances are compared with a Float64 CRLB built from
# the kernel's own derivatives at the truth, for every model.
using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays, Random, LinearAlgebra

include(joinpath(@__DIR__, "long", "utils", "spline_psf_helpers.jl"))

# Float64 inverse Fisher matrix at the true parameters `θ` (kernel units: pixels, z in
# microns) of `model` in pixel units, and the expected counts of a `box` x `box` ROI.
function reference_crlb(model, box, θ)
    n = length(θ)
    θ32 = SVector{n, Float32}(θ)
    mu = zeros(Float32, box, box)
    F = zeros(n, n)
    for j in 1:box, i in 1:box
        m, dudt = GaussMLE.compute_pixel_derivatives(i, j, θ32, model)
        mu[i, j] = m
        F .+= Float64.(dudt) * Float64.(dudt)' ./ Float64(m)
    end
    return inv(F), mu
end

@testset "CRLB of bright emitters against a Float64 reference" begin
    px, box = 0.1, 13
    camera = IdealCamera(box, box, px)
    spline = SplinePSFModel(astig_shape; pixel_size = px, z_range = (-0.5, 0.5))
    astig = AstigmaticXYZNB{Float32}(0.13, 0.13, 0.05, -0.05, 0.01, -0.01, 0.2, 0.5)
    # (name, user-facing model, true parameters at N photons, photon index)
    models = (
        ("GaussianXYNB", GaussianXYNB(0.13f0), N -> [6.2, 6.9, N, N / 100], 3),
        ("GaussianXYNBS", GaussianXYNBS(), N -> [6.2, 6.9, N, N / 100, 1.3], 3),
        (
            "GaussianXYNBSXSY", GaussianXYNBSXSY(),
            N -> [6.2, 6.9, N, N / 100, 1.3, 1.5], 3,
        ),
        ("AstigmaticXYZNB", astig, N -> [6.2, 6.9, 0.15, N, N / 100], 4),
        ("SplinePSFModel", spline, N -> [6.2, 6.9, 0.15, N, N / 100], 4),
    )
    for (name, model, truth, ip) in models
        @testset "$name" begin
            pixel_model = GaussMLE._model_for_camera(model, camera)
            has_z = GaussMLE._has_axial_position(typeof(model))
            for N in (1.0e3, 1.0e5, 1.0e6)
                C, mu = reference_crlb(pixel_model, box, truth(N))
                data = reshape(mu, box, box, 1)
                ids = Int32[1]
                batch = ROIBatch(data, ids, ids, ids, camera)
                smld, _ = fit(batch; psf_model = model, backend = :cpu, iterations = 100)
                e = only(smld.emitters)
                sd(k) = sqrt(C[k, k])
                @test e.σ_photons ≈ sd(ip) rtol = 0.01
                @test e.σ_photons >= sqrt(N)  # no unbiased estimator of N does better
                @test e.σ_bg ≈ sd(ip + 1) rtol = 0.01
                @test e.σ_x ≈ px * sd(1) rtol = 0.01
                @test e.σ_y ≈ px * sd(2) rtol = 0.01
                # Covariances to 1 % of the pair's standard deviations (they can be near 0).
                @test abs(e.σ_xy - px^2 * C[1, 2]) <= 0.01 * px^2 * sd(1) * sd(2)
                if has_z
                    @test e.σ_z ≈ sd(3) rtol = 0.01
                    @test abs(e.σ_xz - px * C[1, 3]) <= 0.01 * px * sd(1) * sd(3)
                    @test abs(e.σ_yz - px * C[2, 3]) <= 0.01 * px * sd(2) * sd(3)
                end
            end
        end
    end
end

# The Gaussian fits' stop. On a noise-free box the likelihood's maximum is the truth and the
# observed and expected information agree there, so a fit stopped by g' F⁻¹ g / 2 < 1e-4
# is within sqrt(2e-4) = 0.0141 CRLB of the truth in every parameter. A noisy box stops
# before the cap within the same distance of where a fit that never stops (a tolerance
# Float32 cannot meet, 500 passes) ends.
using SMLMSim: poisson_noise, gen_images
@testset "Gaussian fits stop at the maximum" begin
    px, box = 0.1, 13
    camera = IdealCamera(box, box, px)
    astig = AstigmaticXYZNB{Float32}(0.13, 0.13, 0.05, -0.05, 0.01, -0.01, 0.2, 0.5)
    models = (
        (GaussianXYNB(0.13f0), N -> [6.2, 6.9, N, N / 100]),
        (GaussianXYNBS(), N -> [6.2, 6.9, N, N / 100, 1.3]),
        (GaussianXYNBSXSY(), N -> [6.2, 6.9, N, N / 100, 1.3, 1.5]),
        (astig, N -> [6.2, 6.9, 0.15, N, N / 100]),
    )
    vmap = zeros(Float32, box, box)
    for (model, truth) in models
        @testset "$(nameof(typeof(model)))" begin
            pm = GaussMLE._model_for_camera(model, camera)
            n = length(truth(1.0))
            cons = GaussMLE.default_constraints(pm, box)
            cap = GaussMLE._default_iterations(model)
            tol = Float32(GaussMLE._LM_GAIN_TOL)
            lm(roi, gain_tol, cap) = GaussMLE._lm_fit(
                roi, box, pm, Val(false), vmap, Int32(1), Int32(1), cons, cap,
                GaussMLE.simple_initialize(roi, box, pm), gain_tol
            )
            for N in (1.0e3, 1.0e5, 1.0e6)
                C, mu = reference_crlb(pm, box, truth(N))
                θ, _, passes, stopped = lm(mu, tol, cap)
                @test stopped && passes < cap
                @test all(k -> abs(θ[k] - truth(N)[k]) <= 0.015 * sqrt(C[k, k]), 1:n)
            end
            C, mu = reference_crlb(pm, box, truth(2000.0))
            Random.seed!(17)
            roi = Float32.(poisson_noise(mu))
            θ, _, passes, stopped = lm(roi, tol, cap)
            @test stopped && passes < cap
            θ_long = first(lm(roi, 1.0f-9, 500))
            @test all(k -> abs(θ[k] - θ_long[k]) <= 0.015 * sqrt(C[k, k]), 1:n)
        end
    end
end

# The Gaussian fits' step rules (gain_tol > 0) where they differ from SplinePSFModel's fixed
# schedule (gain_tol = 0): the trial step is scaled whole to the step limits, a gain within
# the objective's rounding is judged by the sign of the trapezoid of the gradients, and the
# damping follows the ratio of actual to predicted gain. The SMLMSim ROIs below are ones
# earlier rules fitted badly: bright XYNBS fits that ran to any cap without stopping (the
# objective could not resolve their last gains), and low-count astigmatic fits that took 36
# to 57 passes with lambda divided by 10 on every accepted pass and each step component
# clamped alone, or ran to the cap of 50 through chains of trials accepted as ties at the
# rounding level. Each stops within 6 to 15 passes now.
@testset "Gaussian fit step rules" begin
    @testset "a trial step is scaled whole to the step limits" begin
        cons = GaussMLE.default_constraints(GaussianXYNBS(), 13)  # x, y: 1 px; σ: 0.5 px
        θ = SVector{5, Float32}(6.0, 7.0, 1000.0, 10.0, 1.3)
        g = SVector{5, Float32}(1.0, -2.0, 0.5, 0.1, 3.0)
        F = SMatrix{5, 5, Float32}(Diagonal(Float32[4, 3, 1.0e-3, 0.5, 20])) +
            SMatrix{5, 5, Float32}([i + j == 3 ? 0.5f0 : 0.0f0 for i in 1:5, j in 1:5])
        pred(s) = dot(g, s) - dot(s, F * s) / 2
        # x at 2.5 times its limit: every component scaled by 0.4
        Δ = SVector{5, Float32}(2.5, -0.5, 300.0, -3.0, 0.2)
        θt, _ = GaussMLE._adaptive_trial(Δ, θ, g, F, cons)
        s = θt - θ
        @test s ≈ 0.4f0 * Δ rtol = 1.0e-6
        @test all(abs.(s) .<= cons.max_step)
        # a non-finite component is bounded by its limit and does not scale the others
        θt, p = GaussMLE._adaptive_trial(
            SVector{5, Float32}(Inf, 0.5, 10.0, 1.0, 0.1), θ, g, F, cons
        )
        @test θt - θ ≈ SVector{5, Float32}(1.0, 0.5, 10.0, 1.0, 0.1) rtol = 1.0e-6
        @test isfinite(p)
        # a step past a bound ends on it, and the prediction is for the step taken
        θb = Base.setindex(θ, 14.6f0, 1)  # upper x bound of a 13 px box: 15 px
        θt, p = GaussMLE._adaptive_trial(
            SVector{5, Float32}(0.8, 0, 0, 0, 0), θb, g, F, cons
        )
        @test θt[1] == cons.upper[1] && θt[2:5] == θb[2:5]
        @test p ≈ pred(θt - θb) rtol = 1.0e-5
    end

    box = 13
    camera = IdealCamera(box, box, 0.1)
    vmap = zeros(Float32, box, box)
    tol = Float32(GaussMLE._LM_GAIN_TOL)
    function expected_counts(pm, θ)
        θ32 = SVector{length(θ), Float32}(θ)
        return Float32[
            first(GaussMLE.compute_pixel_derivatives(i, j, θ32, pm)) for i in 1:box, j in 1:box
        ]
    end
    function noisy_roi(pm, θ, seed)
        mu = expected_counts(pm, θ)
        Random.seed!(seed)
        return Float32.(poisson_noise(mu))
    end
    lm(roi, pm, cons, cap, θ0, gain_tol) = GaussMLE._lm_fit(
        roi, box, pm, Val(false), vmap, Int32(1), Int32(1), cons, cap, θ0, gain_tol
    )
    astig = AstigmaticXYZNB{Float32}(0.13, 0.13, 0.05, -0.05, 0.01, -0.01, 0.2, 0.5)

    @testset "the first step: clamped per parameter or scaled whole" begin
        # Noise-free ROI at z = 0.3 um fitted from z = -0.1 um: the first step in z, 0.26 um,
        # exceeds its 0.1 um limit. Clamping z alone keeps the photon and background steps
        # made for the full z step; scaling shortens them all.
        pm = GaussMLE._model_for_camera(astig, camera)
        cons = GaussMLE.default_constraints(pm, box)
        mu = expected_counts(pm, [6.2, 6.9, 0.3, 2000.0, 20.0])
        θ0 = SVector{5, Float32}(6.2, 6.9, -0.1, 2000.0, 20.0)
        g = zeros(5)
        F = zeros(5, 5)
        for j in 1:box, i in 1:box
            m, dudt = GaussMLE.compute_pixel_derivatives(i, j, θ0, pm)
            g .+= Float64.(dudt) .* ((mu[i, j] - m) / m)
            F .+= Float64.(dudt) * Float64.(dudt)' ./ m
        end
        Δ = (F + 1.0e-3 * Diagonal(diag(F))) \ g  # the first pass's damping, 1e-3
        lim = Float64.(cons.max_step)
        @test abs(Δ[3]) > 2 * lim[3]
        clamped = Float64.(θ0) .+ clamp.(Δ, -lim, lim)
        scaled = Float64.(θ0) .+ Δ .* (lim[3] / abs(Δ[3]))
        @test abs(clamped[4] - scaled[4]) > 100  # photons
        # iterations = 2: the start, then the first trial (accepted in both)
        @test lm(mu, pm, cons, 2, θ0, 0.0f0)[1] ≈ clamped rtol = 1.0e-5
        @test lm(mu, pm, cons, 2, θ0, tol)[1] ≈ scaled rtol = 1.0e-5
    end

    @testset "bright fits stop within a few passes" begin
        pm = GaussMLE._model_for_camera(GaussianXYNBS(), camera)
        cons = GaussMLE.default_constraints(pm, box)
        for seed in (41, 150, 177)
            roi = noisy_roi(pm, [6.2, 6.9, 1.0e5, 2.0, 1.3], seed)
            _, _, passes, stopped = lm(
                roi, pm, cons, 50, GaussMLE.simple_initialize(roi, box, pm), tol
            )
            @test stopped && passes <= 8
        end
    end

    @testset "low-count astigmatic fits stop within 20 passes" begin
        pm = GaussMLE._model_for_camera(
            AstigmaticXYZNB{Float32}(0.13, 0.13, 0.05, 0.05, 0.3, 0.3, 0.05, 0.4), camera
        )
        cons = GaussMLE.default_constraints(pm, box)
        for seed in (199, 245, 350, 1598, 2407)
            roi = noisy_roi(pm, [6.2, 6.9, 0.15, 500.0, 20.0], seed)
            _, _, passes, stopped = lm(
                roi, pm, cons, 50, GaussMLE.simple_initialize(roi, box, pm), tol
            )
            @test stopped && passes <= 20
        end
    end

    @testset "a bright fit to a mismatched PSF stops" begin
        # Noise-free SMLMSim ROI of a Gaussian PSF 0.18 um wide (300,000 photons, background
        # 50), fitted with the 0.13 um fixed-width model. The mismatch makes the objective
        # about -3.4e4, where the Float32 spacing (0.004) exceeds the last gains, so they are
        # judged by the trapezoid's sign. Accepting only a rise of the objective, or adding
        # the trapezoid gain to the objective before comparing, rounds them away and the fit
        # runs to the cap.
        psf = GaussianXYNB(0.13f0)
        pm = GaussMLE._model_for_camera(psf, camera)
        cons = GaussMLE.default_constraints(pm, box)
        um(p) = (p - 0.5f0) * 0.1f0
        e = Emitter2DFit{Float32}(
            um(6.7f0), um(7.2f0), 3.0f5, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0
        )
        images, _ = gen_images(BasicSMLD([e], camera, 1, 1), GaussianPSF(0.18); bg = 50.0)
        roi = Float32.(images[:, :, 1])
        θ0 = GaussMLE.simple_initialize(roi, box, pm)
        θ, obj, passes, stopped = lm(roi, pm, cons, 50, θ0, tol)
        @test obj < -1.0e4
        @test stopped && passes <= 15
        C, _ = reference_crlb(pm, box, θ)
        θ_long = first(lm(roi, pm, cons, 500, θ0, 1.0f-9))
        @test all(k -> abs(θ[k] - θ_long[k]) <= 0.015 * sqrt(C[k, k]), 1:4)
    end

    @testset "a fit whose next step rounds away ends early" begin
        # SMLMSim ROI of a narrow PSF (sigma 0.31 px, near its 0.3 px bound; 1e5 photons,
        # background 5; 11 px box). Near the maximum the Float32 objective's noise exceeds
        # the remaining gain, every trial is rejected until the step rounds to the accepted
        # point, and from there each pass would repeat the last: the fit ends at once,
        # without meeting the stop test, with the result a longer cap gives. 1 in 4,000 of
        # these ROIs (simulated positions within 1 px of the center) ends this way.
        b11 = 11
        cam11 = IdealCamera(b11, b11, 0.1)
        pm = GaussMLE._model_for_camera(GaussianXYNBS(), cam11)
        cons = GaussMLE.default_constraints(pm, b11)
        θt = SVector{5, Float32}(5.8400774f0, 6.8125696f0, 1.0f5, 5, 0.31f0)
        mu = Float32[
            first(GaussMLE.compute_pixel_derivatives(i, j, θt, pm)) for i in 1:b11, j in 1:b11
        ]
        Random.seed!(1729)
        roi = Float32.(poisson_noise(mu))
        v11 = zeros(Float32, b11, b11)
        fit11(cap) = GaussMLE._lm_fit(
            roi, b11, pm, Val(false), v11, Int32(1), Int32(1), cons, cap,
            GaussMLE.simple_initialize(roi, b11, pm), tol
        )
        θ, _, passes, stopped = fit11(100)
        @test !stopped && passes < 100
        @test first(fit11(1000)) == θ
    end

    @testset "a fit capped before its stop is counted as not converged" begin
        psf = GaussianXYNB(0.13f0)
        pm = GaussMLE._model_for_camera(psf, camera)
        cons = GaussMLE.default_constraints(pm, box)
        roi = noisy_roi(pm, [6.2, 6.9, 2000.0, 20.0], 17)
        _, _, passes, stopped = lm(
            roi, pm, cons, 50, GaussMLE.simple_initialize(roi, box, pm), tol
        )
        @test stopped && passes > 1
        batch = ROIBatch(reshape(roi, box, box, 1), Int32[1], Int32[1], Int32[1], camera)
        fitted(iterations) = fit(batch; psf_model = psf, backend = :cpu, iterations)
        (capped, capped_info), (done, done_info) = fitted(passes - 1), fitted(passes)
        @test capped_info.n_converged == 0
        @test done_info.n_converged == 1
        # the uncertainties stay the CRLB at the end point
        e = only(capped.emitters)
        @test all(isfinite, (e.x, e.photons, e.σ_x, e.σ_y, e.σ_photons, e.σ_bg))
        @test all(isfinite, (only(done.emitters).σ_x, only(done.emitters).σ_photons))
    end
end
