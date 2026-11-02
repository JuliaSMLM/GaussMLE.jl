# SplinePSFModel fits of defocused ROIs whose box cuts off much of the PSF. Guards the 0.4.3
# Levenberg-Marquardt fit reaching the maximum at the public default iterations. PSFLearning
# measured the 0.4.2-era spline fit (diagonal Newton at fixed iterations) stopping 3 to 9
# log-likelihood units short on such ROIs. Scene: a scalar PSF with 0.8 rad astigmatism and
# 0.08 rad spherical aberration, a 9 px box of 0.1 µm pixels holding 0.58 / 0.85 / 0.60 of N
# at z -0.5 / 0 / +0.5 µm. Over 400 fits at z -0.5..-0.4 and +0.4..+0.5 µm, the LM fit at 20
# iterations had 0 fits more than 0.05 below the reference, with median dz at most 0.003 µm.
# The diagonal Newton at 20 iterations fell short for 395 of 400 fits, by a median of 0.9 to
# 1.2 units, with median dz of ±0.033 µm. The reference is the same LM fit from the true
# parameters at 200 iterations.
using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays, Random, Statistics
using SMLMSim: poisson_noise

@testset "SplinePSFModel: defocused ROIs reach the maximum at default iterations" begin
    zc = ZernikeCoefficients(15)
    zc.phase[6] = 0.8  # vertical astigmatism, 0.8 rad RMS
    zc.phase[11] = 0.08  # spherical aberration, 0.08 rad RMS
    spline = SplinePSF(
        ScalarPSF(1.4, 0.68, 1.518; zernike_coeffs = zc);
        lateral_range = 1.5, axial_range = 0.7, lateral_step = 0.05, axial_step = 0.05
    )
    px, box, n = 0.1, 9, 400
    photons, bg = 3000.0, 10.0
    model = SplinePSFModel(spline; pixel_size = px, z_range = (-0.6, 0.6))
    edges = collect(0:box) .* px

    rng = Xoshiro(5)
    Random.seed!(5)  # SMLMSim's noise draws from the global generator
    # x, y within ±0.5 pixel of the ROI center; z in -0.5..-0.4 for the first half,
    # 0.4..0.5 after
    c = (box + 1) ÷ 2
    truth = [
        (c + rand(rng) - 0.5, c + rand(rng) - 0.5, (k <= n ÷ 2 ? -0.5 : 0.4) + 0.1rand(rng))
            for k in 1:n
    ]
    signal = zeros(Float32, box, box, n)
    for (k, (x, y, z)) in enumerate(truth)
        emitters = [Emitter3D((x - 0.5) * px, (y - 0.5) * px, z, photons)]
        signal[:, :, k] .= MicroscopePSFs.integrate_pixels(
            spline, edges, edges, emitters; sampling = 8
        ) .+ bg
    end
    data = Float32.(poisson_noise(signal))

    batch = ROIBatch(
        data, ones(Int32, n), ones(Int32, n), Int32.(1:n), IdealCamera(64, 64, px)
    )
    smld, info = fit(batch; psf_model = model, backend = :cpu)
    @test info.backend == :cpu
    e = smld.emitters
    @test length(e) == n

    # Poisson negative log-likelihood of one ROI under θ, summed in Float64
    nll(roi, θ) = sum(
        Float64(GaussMLE.evaluate_psf(model, i, j, θ)) -
            roi[i, j] * log(Float64(GaussMLE.evaluate_psf(model, i, j, θ)))
            for j in 1:box, i in 1:box
    )
    cons = GaussMLE.default_constraints(model, box)
    vmap = zeros(Float32, box, box)
    lm(roi, θ0, it) = first(
        GaussMLE._lm_fit(
            roi, box, model, Val(false), vmap, Int32(1), Int32(1), cons, it, θ0
        )
    )

    gap = zeros(n)
    dz = zeros(n)
    for k in 1:n
        roi = @view data[:, :, k]
        x, y, z = truth[k]
        ref = nll(roi, lm(roi, SVector{5, Float32}(x, y, z, photons, bg), 200))
        # ROI corner (1, 1): pixel x sits at e.x / px + 0.5, as in spline_psf_validation.jl
        θ = SVector{5, Float32}(
            e[k].x / px + 0.5, e[k].y / px + 0.5, e[k].z, e[k].photons, e[k].bg
        )
        gap[k] = nll(roi, θ) - ref
        dz[k] = e[k].z - z
    end

    for (name, s) in (("z -0.5..-0.4 µm", 1:(n ÷ 2)), ("z +0.4..+0.5 µm", (n ÷ 2 + 1):n))
        @testset "$name" begin
            println(
                "  $name: max NLL gap = ", maximum(gap[s]),
                ", median dz = ", median(dz[s])
            )
            # One-sided: a fit at or above the reference's likelihood passes.
            @test maximum(gap[s]) <= 0.05
            @test abs(median(dz[s])) <= 0.01
        end
    end
end
