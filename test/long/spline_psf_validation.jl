# SplinePSFModel accuracy: 200 Poisson ROIs from an astigmatic MicroscopePSFs SplinePSF,
# fitted through fit(::ROIBatch). MicroscopePSFs.integrate_pixels renders the point photon
# density independently, so bias, total-photon scale and CRLB checks cover pixel
# integration.
using Test, GaussMLE, MicroscopePSFs, SMLMData, Random, Statistics
using SMLMSim: poisson_noise

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
    edges = collect(0:box) .* px

    rng = Xoshiro(42)
    Random.seed!(42)  # SMLMSim's noise draws from the global generator
    # x, y within ±1 pixel of the ROI centre; z within ±0.3 µm
    truth = [(7 + 2rand(rng) - 1, 7 + 2rand(rng) - 1, 0.6rand(rng) - 0.3) for _ in 1:n]
    data = zeros(Float32, box, box, n)
    for (k, (x, y, z)) in enumerate(truth)
        emitters = [Emitter3D((x - 0.5) * px, (y - 0.5) * px, z, Float64(n_photons))]
        signal = MicroscopePSFs.integrate_pixels(
            spline, edges, edges, emitters; sampling = 8
        )
        data[:, :, k] .= poisson_noise(bg .+ signal)
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
    crlb_z = sqrt(mean(abs2, [em.σ_z for em in e])) * 1000

    @test abs(mean(dx)) <= 3
    @test abs(mean(dy)) <= 3
    @test abs(mean(dz)) <= 10
    # With 200 fits, a 30% two-sided tolerance allows sampling variation; observed
    # SD/CRLB ratios on kitt were 0.88–1.12. Inflated or infinite CRLBs must fail too.
    for field in (:σ_x, :σ_y, :σ_z)
        @test all(em -> isfinite(getproperty(em, field)) && getproperty(em, field) > 0, e)
    end
    for crlb in (crlb_x, crlb_y, crlb_z)
        @test isfinite(crlb) && crlb > 0
    end
    @test abs(1 - std(dx) / crlb_x) < 0.3
    @test abs(1 - std(dy) / crlb_y) < 0.3
    @test abs(1 - std(dz) / crlb_z) < 0.3
end

# Every 0.1 µm band of z over ±0.5 µm meets the same thresholds. SMLMAnalysis S1 found fits
# at |z| 0.4-0.5 µm falling into a wrong minimum on the other side of focus (z near ∓0.25,
# a third of the photons) when every fit started at z = 0.
@testset "SplinePSFModel: every z band over ±0.5 µm" begin
    zc = ZernikeCoefficients(15)
    zc.phase[6] = 0.5  # vertical astigmatism, 0.5 rad RMS
    spline = SplinePSF(
        ScalarPSF(1.4, 0.6, 1.518; zernike_coeffs = zc);
        lateral_range = 1.0, axial_range = 0.8, lateral_step = 0.05, axial_step = 0.05
    )
    px, box, n = 0.1, 15, 200
    n_photons, bg = 3000.0f0, 10.0f0
    model = SplinePSFModel(spline; pixel_size = px)
    edges = collect(0:box) .* px
    rng = Xoshiro(7)
    Random.seed!(7)

    for zlo in -0.5:0.1:0.4
        truth = [(8 + 2rand(rng) - 1, 8 + 2rand(rng) - 1, zlo + 0.1rand(rng)) for _ in 1:n]
        data = zeros(Float32, box, box, n)
        for (k, (x, y, z)) in enumerate(truth)
            emitters = [Emitter3D((x - 0.5) * px, (y - 0.5) * px, z, Float64(n_photons))]
            signal = MicroscopePSFs.integrate_pixels(
                spline, edges, edges, emitters; sampling = 8
            )
            data[:, :, k] .= poisson_noise(bg .+ signal)
        end
        batch = ROIBatch(
            data, ones(Int32, n), ones(Int32, n), Int32.(1:n), IdealCamera(64, 64, px)
        )
        e = first(fit(batch; psf_model = model, backend = :cpu)).emitters

        dx = [(e[k].x - (truth[k][1] - 0.5) * px) * 1000 for k in 1:n]
        dy = [(e[k].y - (truth[k][2] - 0.5) * px) * 1000 for k in 1:n]
        dz = [(e[k].z - truth[k][3]) * 1000 for k in 1:n]
        crlb_x = sqrt(mean(abs2, [em.σ_x for em in e])) * 1000
        crlb_y = sqrt(mean(abs2, [em.σ_y for em in e])) * 1000
        crlb_z = sqrt(mean(abs2, [em.σ_z for em in e])) * 1000

        @testset "z in [$(round(zlo; digits = 1)), $(round(zlo + 0.1; digits = 1))]" begin
            @test abs(mean(dx)) <= 3
            @test abs(mean(dy)) <= 3
            @test abs(mean(dz)) <= 10
            @test abs(mean(em.photons for em in e) / n_photons - 1) < 0.01
            # The same 30% tolerance accounts for 200-fit sampling variation per band.
            for field in (:σ_x, :σ_y, :σ_z)
                @test all(
                    em -> isfinite(getproperty(em, field)) && getproperty(em, field) > 0, e
                )
            end
            for crlb in (crlb_x, crlb_y, crlb_z)
                @test isfinite(crlb) && crlb > 0
            end
            @test abs(1 - std(dx) / crlb_x) < 0.3
            @test abs(1 - std(dy) / crlb_y) < 0.3
            @test abs(1 - std(dz) / crlb_z) < 0.3
        end
        println(
            "z band ", round(zlo; digits = 1), ": bias x ", round(mean(dx); digits = 2),
            " y ", round(mean(dy); digits = 2), " z ", round(mean(dz); digits = 2),
            " nm; SD/CRLB x ", round(std(dx) / crlb_x; digits = 3), " y ",
            round(std(dy) / crlb_y; digits = 3), " z ", round(std(dz) / crlb_z; digits = 3),
            "; |dz| > 200 nm: ", count(>(200) ∘ abs, dz), "/", n
        )
    end
end
