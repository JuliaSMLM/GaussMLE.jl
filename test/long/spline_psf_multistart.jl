# SplinePSFModel fits from several z starts. One Levenberg-Marquardt start often sits in the
# wrong basin of a defocused ROI; the default (the scan start plus fixed z starts, one per
# 0.4 µm over the central 80 % of z_range) recovers it. Scene: a vector PSF with mismatched
# media (n_medium 1.335, coverslip and immersion 1.516) and Noll 5/6/7/11 aberrations of
# 0.25/-0.15/0.12/0.08 rad, a 15 px box of 0.1 µm pixels, 3000 photons, background 10, 400
# ROIs at z -0.5..-0.4 and +0.4..+0.5 µm, z_range (-0.6, 0.6). In the probe (probe6.jl),
# starts = 1 ended more than 0.05 below the reference for 101 of 200 fits at -0.5..-0.4
# and 15 of 200 at +0.4..+0.5; the default had 0 of 400. The reference is the better of the
# LM fit from the true parameters at 200 iterations and the fits themselves.
using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays, Random
using SMLMSim: poisson_noise

@testset "SplinePSFModel: several z starts reach the likelihood maximum" begin
    zc = ZernikeCoefficients(15)
    zc.phase[5] = 0.25
    zc.phase[6] = -0.15
    zc.phase[7] = 0.12
    zc.phase[11] = 0.08
    vpsf = VectorPSF(
        1.4, 0.68; base_zernike = zc, n_medium = 1.335, n_coverslip = 1.516,
        n_immersion = 1.516
    )
    spline = SplinePSF(
        vpsf; lateral_range = 1.0, axial_range = 0.7, lateral_step = 0.1, axial_step = 0.1
    )
    px, box, n = 0.1, 15, 400
    photons, bg = 3000.0, 10.0
    z_range = (-0.6, 0.6)
    model_one = SplinePSFModel(spline; pixel_size = px, z_range, starts = 1)
    model_default = SplinePSFModel(spline; pixel_size = px, z_range)
    @test model_default.starts == 5

    edges = collect(0:box) .* px
    rng = Xoshiro(3)
    Random.seed!(3)  # SMLMSim's noise draws from the global generator
    truth = [
        (
            8 + rand(rng) - 0.5, 8 + rand(rng) - 0.5,
            (k <= n ÷ 2 ? -0.5 : 0.4) + 0.1rand(rng),
        ) for k in 1:n
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

    # Poisson log-likelihood of one ROI under θ (pixels), summed in Float64
    ll(roi, θ) = sum(
        roi[i, j] * log(Float64(GaussMLE.evaluate_psf(model_one, i, j, θ))) -
            Float64(GaussMLE.evaluate_psf(model_one, i, j, θ)) for j in 1:box, i in 1:box
    )
    cons = GaussMLE.default_constraints(model_one, box)
    vmap = zeros(Float32, box, box)
    lm(roi, θ0, it) = first(
        GaussMLE._lm_fit(
            roi, box, model_one, Val(false), vmap, Int32(1), Int32(1), cons, it, θ0
        )
    )

    fitted = map((model_one, model_default)) do model
        smld, _ = fit(batch; psf_model = model, backend = :cpu)
        e = smld.emitters
        # ROI corner (1, 1): pixel x sits at e.x / px + 0.5, as in spline_psf_defocus.jl
        [
            ll(
                view(data, :, :, k),
                SVector{5, Float32}(
                    e[k].x / px + 0.5, e[k].y / px + 0.5, e[k].z, e[k].photons, e[k].bg
                )
            ) for k in 1:n
        ]
    end
    reference = map(1:n) do k
        x, y, z = truth[k]
        best = ll(
            view(data, :, :, k),
            lm(view(data, :, :, k), SVector{5, Float32}(x, y, z, photons, bg), 200)
        )
        max(best, fitted[1][k], fitted[2][k])
    end

    for (name, s, min_short) in (
            ("z -0.5..-0.4 µm", 1:(n ÷ 2), 50),
            ("z +0.4..+0.5 µm", (n ÷ 2 + 1):n, 0),
        )
        @testset "$name" begin
            short_one = count(>(0.05), reference[s] .- fitted[1][s])
            short_default = count(>(0.05), reference[s] .- fitted[2][s])
            @info "$name: short $short_one (starts = 1), $short_default (default)"
            @test short_default == 0
            @test short_one >= min_short
        end
    end
end
