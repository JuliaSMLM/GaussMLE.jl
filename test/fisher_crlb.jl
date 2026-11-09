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
# is within sqrt(2e-4) = 0.0141 CRLB of the truth in every parameter. With the stop off
# (gain_tol = 0) the fit runs every pass. A noisy box stops before the cap too.
using SMLMSim: poisson_noise
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
            lm(roi, gain_tol) = GaussMLE._lm_fit(
                roi, box, pm, Val(false), vmap, Int32(1), Int32(1), cons, cap,
                GaussMLE.simple_initialize(roi, box, pm), gain_tol
            )
            for N in (1.0e3, 1.0e5, 1.0e6)
                C, mu = reference_crlb(pm, box, truth(N))
                θ, _, passes = lm(mu, tol)
                @test passes < cap
                @test all(k -> abs(θ[k] - truth(N)[k]) <= 0.015 * sqrt(C[k, k]), 1:n)
                @test lm(mu, 0.0f0)[3] == cap
            end
            _, mu = reference_crlb(pm, box, truth(2000.0))
            Random.seed!(17)
            @test lm(Float32.(poisson_noise(mu)), tol)[3] < cap
        end
    end
end
