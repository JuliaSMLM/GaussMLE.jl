# The end start of the spline models (`_end_start`, `_end_start_replaces`). A calibration
# whose focus lies 0.075 µm below the upper end of a 1 µm z_range, with a weakly astigmatic
# PSF (Zernike astigmatism 0.03), so the two sides of focus look alike. An emitter 0.06 µm
# above focus, between the focus and the end, has a fit maximum on each side and no fixed
# start on its own; with the fixed starts alone the fit took the other side for about half
# of these ROIs (51 of 120 within 0.03 µm of the truth for one image; for two images 159
# of 300 ended more than 0.03 µm from a fit from 81 starts). The end start covers that side.
using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays, Random
using SMLMSim: poisson_noise

@testset "spline models: the end start" begin
    zc = ZernikeCoefficients(15)
    zc.phase[6] = 0.03
    weak = SplinePSF(
        ScalarPSF(1.4, 0.68, 1.518; zernike_coeffs = zc),
        -1.0:0.1:1.0, -1.0:0.1:1.0, -0.925:0.05:0.075
    )
    camera = IdealCamera(64, 64, 0.1)
    box, corner, n, z = 9, 28, 120, 0.06
    ex = Float64.(camera.pixel_edges_x[corner:(corner + box)])
    ey = Float64.(camera.pixel_edges_y[corner:(corner + box)])
    c = fill(Int32(corner), n)
    frames = Int32.(1:n)

    @testset "one image" begin
        m = SplinePSFModel(weak; pixel_size = 0.1)
        @test m.z_range[2] ≈ 0.075f0
        mu = 20.0 .+ MicroscopePSFs.integrate_pixels(
            weak, ex, ey, [Emitter3D(3.263, 3.241, z, 20000.0)]; sampling = 4
        )
        Random.seed!(5)
        data = Float32.(poisson_noise(repeat(mu; outer = (1, 1, n))))
        smld, info = fit(ROIBatch(data, c, c, frames, camera); psf_model = m, backend = :cpu)
        @test info.n_converged == n
        @test count(e -> abs(e.z - z) < 0.03, smld.emitters) >= 0.9n

        # starts = 1 is the scan start alone, with no end start, also on these ROIs, where
        # the end start changes the default result for many
        one = SplinePSFModel(weak; pixel_size = 0.1, starts = 1)
        cons = GaussMLE.default_constraints(one, box)
        vmap = zeros(Float32, box, box)
        single = 0
        changed = 0
        for k in 1:n
            roi = data[:, :, k]
            θ0 = GaussMLE.simple_initialize(roi, box, one)
            a = (Val(false), vmap, Int32(1), Int32(1), cons, 100, θ0)
            θ1 = first(GaussMLE._optimize(roi, box, one, a...))
            single += θ1 == first(GaussMLE._lm_fit(roi, box, one, a...))
            changed += θ1[3] != first(GaussMLE._optimize(roi, box, m, a...))[3]
        end
        @test single == n
        @test changed >= n ÷ 4
    end

    if !isdefined(MicroscopePSFs, :MultiChannelPSF)
        @info "long/spline_end_start.jl: MicroscopePSFs $(pkgversion(MicroscopePSFs)) has no " *
            "MultiChannelPSF (0.5.7 and later), so JointSplinePSFModel is not defined"
    else
        @testset "two images" begin
            # two identical channels, identity registration, a 45/55 split, background 10 each
            I23 = [1.0 0 0; 0 1 0]
            mc = MultiChannelPSF(;
                channels = [weak, weak], registration = [I23, I23], transmission = [0.45, 0.55]
            )
            jm = JointSplinePSFModel(mc; pixel_size = 0.1)
            mu = ntuple(
                v -> 10.0 .+ MicroscopePSFs.integrate_pixels(
                    weak, ex, ey, [Emitter3D(3.263, 3.241, z, 20000.0 * mc.transmission[v])];
                    sampling = 4
                ), 2
            )
            Random.seed!(6)
            d = ntuple(v -> Float32.(poisson_noise(repeat(mu[v]; outer = (1, 1, n)))), 2)
            batches = (ROIBatch(d[1], c, c, frames, camera), ROIBatch(d[2], c, c, frames, camera))
            smld, info = fit(batches; psf_model = jm, backend = :cpu)
            @test info.n_converged == n
            @test count(e -> abs(e.z - z) < 0.03, smld.emitters) >= 0.9n
        end
    end
end
