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
        m, dudt, _ = GaussMLE.compute_pixel_derivatives(i, j, θ32, model)
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
