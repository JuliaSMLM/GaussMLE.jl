# SplinePSFModel accuracy: 200 Poisson ROIs from an astigmatic MicroscopePSFs SplinePSF,
# fitted through fit(::ROIBatch). The data come from the model's own forward model (PSF
# sampled at pixel centres), so this checks the fitter and its CRLB, not pixel-integration
# effects.
using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays, Distributions, Random,
    Statistics

@testset "SplinePSFModel: bias and CRLB (astigmatic SplinePSF)" begin
    zc = ZernikeCoefficients(15)
    zc.phase[6] = 0.5  # vertical astigmatism, 0.5 rad RMS
    spline = SplinePSF(
        ScalarPSF(1.4, 0.6, 1.518; zernike_coeffs = zc);
        lateral_range = 1.0, axial_range = 0.6, lateral_step = 0.05, axial_step = 0.05
    )
    px, box, n = 0.1, 13, 200
    n_photons, bg = 2000.0f0, 10.0f0
    model = SplinePSFModel(spline; pixel_size = px)

    rng = Xoshiro(42)
    # x, y within ±1 pixel of the ROI centre; z within ±0.3 µm
    truth = [(7 + 2rand(rng) - 1, 7 + 2rand(rng) - 1, 0.6rand(rng) - 0.3) for _ in 1:n]
    data = zeros(Float32, box, box, n)
    for (k, (x, y, z)) in enumerate(truth), j in 1:box, i in 1:box
        μ = GaussMLE.evaluate_psf(model, i, j, SVector{5, Float32}(x, y, z, n_photons, bg))
        data[i, j, k] = rand(rng, Poisson(μ))
    end
    batch = ROIBatch(
        data, ones(Int32, n), ones(Int32, n), Int32.(1:n), IdealCamera(64, 64, px)
    )

    smld, info = fit(batch; psf_model = model, backend = :cpu)
    @test info.backend == :cpu
    e = smld.emitters
    @test length(e) == n

    # Errors in nm; ROI corner (1, 1), so pixel x sits at (x - 0.5) * px µm
    dx = [(e[k].x - (truth[k][1] - 0.5) * px) * 1000 for k in 1:n]
    dy = [(e[k].y - (truth[k][2] - 0.5) * px) * 1000 for k in 1:n]
    dz = [(e[k].z - truth[k][3]) * 1000 for k in 1:n]
    crlb_x = sqrt(mean(abs2, [em.σ_x for em in e])) * 1000
    crlb_y = sqrt(mean(abs2, [em.σ_y for em in e])) * 1000

    @test abs(mean(dx)) <= 3
    @test abs(mean(dy)) <= 3
    @test abs(mean(dz)) <= 10
    @test std(dx) <= 1.3 * crlb_x
    @test std(dy) <= 1.3 * crlb_y
end
