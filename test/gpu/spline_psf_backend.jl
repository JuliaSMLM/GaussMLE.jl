# Independent pixel-integral data, CPU/GPU agreement and GPU bias/CRLB in every z band.
using Test, GaussMLE, MicroscopePSFs, SMLMData, Distributions, Random, Statistics

include(joinpath(@__DIR__, "utils", "gpu_vendor.jl"))

@testset "SplinePSFModel on CPU and GPU" begin
    @test gpu_backend() !== nothing
    zc = ZernikeCoefficients(15)
    zc.phase[6] = 0.5
    spline = SplinePSF(
        ScalarPSF(1.4, 0.6, 1.518; zernike_coeffs = zc);
        lateral_range = 1.0, axial_range = 0.8, lateral_step = 0.05, axial_step = 0.05
    )
    # Rectangular pixels exercise both lateral derivative scales on each backend.
    px_x, px_y = 0.1313, 0.1258
    model = SplinePSFModel(spline; pixel_size = (px_x, px_y), z_range = (-0.5, 0.5))
    box, n = 15, 200
    edges_x, edges_y = collect(0:box) .* px_x, collect(0:box) .* px_y
    photons, bg = 3000.0, 10.0
    rng = Xoshiro(7)
    parameter_fields = (:x, :y, :z, :photons, :bg)
    uncertainty_fields = (:σ_x, :σ_y, :σ_z, :σ_photons, :σ_bg)

    for band in 0:9
        zlo = -0.5 + 0.1band
        truth = [
            (8 + 2rand(rng) - 1, 8 + 2rand(rng) - 1, zlo + 0.1rand(rng))
                for _ in 1:n
        ]
        signal = zeros(Float32, box, box, n)
        for (k, (x, y, z)) in enumerate(truth)
            emitters = [Emitter3D((x - 0.5) * px_x, (y - 0.5) * px_y, z, photons)]
            signal[:, :, k] .= MicroscopePSFs.integrate_pixels(
                spline, edges_x, edges_y, emitters; sampling = 8
            ) .+ bg
        end

        @testset "z band $band, $camera_kind" for camera_kind in (:ideal, :scmos)
            if camera_kind == :ideal
                camera = IdealCamera(64, 64, (px_x, px_y))
                data = Float32[rand(rng, Poisson(mu)) for mu in signal]
            else
                # Raw ADU, with nonuniform full-sensor noise and nontrivial ROI corners.
                readnoise = [
                    1.0 + 0.01 * i + 0.005 * j for i in 1:64, j in 1:64
                ]
                gain, offset = 0.5, 100.0
                camera = SCMOSCamera(
                    64, 64, (px_x, px_y), readnoise; gain, offset, qe = 1.0
                )
                data = similar(signal)
                for k in 1:n, j in 1:box, i in 1:box
                    electrons = rand(rng, Poisson(signal[i, j, k])) +
                        randn(rng, Float32) * readnoise[i + 10, j + 5]
                    data[i, j, k] = electrons / gain + offset
                end
            end
            corner_x, corner_y = camera_kind == :ideal ? (1, 1) : (6, 11)
            batch = ROIBatch(
                data, fill(Int32(corner_x), n), fill(Int32(corner_y), n),
                Int32.(1:n), camera
            )
            cpu, cpu_info = fit(batch; psf_model = model, backend = :cpu)
            # Several batches must share one adapted table.
            gpu, gpu_info = fit(
                batch; psf_model = model, backend = :gpu, batch_size = 73,
                gpu_timeout = 60.0
            )
            @test cpu_info.backend == :cpu
            @test gpu_info.backend == :gpu
            @test gpu_info.n_batches == 3
            @test length(cpu.emitters) == length(gpu.emitters) == n
            for field in parameter_fields
                a = [getproperty(e, field) for e in cpu.emitters]
                b = [getproperty(e, field) for e in gpu.emitters]
                @test all(isfinite, a) && all(isfinite, b)
                @test a ≈ b rtol = 1.0e-4
            end
            for field in uncertainty_fields
                a = [getproperty(e, field) for e in cpu.emitters]
                b = [getproperty(e, field) for e in gpu.emitters]
                @test all(x -> isfinite(x) && x > 0, a)
                @test all(x -> isfinite(x) && x > 0, b)
                @test a ≈ b rtol = 1.0e-3
            end
            # Spatial covariance can be near zero; use its variance scale as in the
            # Gaussian agreement test, including xz and yz for the axial trait paths.
            for (field, sigma_a, sigma_b) in (
                    (:σ_xy, :σ_x, :σ_y), (:σ_xz, :σ_x, :σ_z), (:σ_yz, :σ_y, :σ_z),
                )
                a = [getproperty(e, field) for e in cpu.emitters]
                b = [getproperty(e, field) for e in gpu.emitters]
                scale = maximum(
                    e -> getproperty(e, sigma_a) * getproperty(e, sigma_b), cpu.emitters
                )
                @test a ≈ b rtol = 1.0e-3 atol = 1.0e-3 * scale
            end

            # Long thresholds: 3/3/10 nm bias, 30% SD/CRLB and 1% photon bias.
            for (axis, px, corner, sigma, tolerance) in (
                    (1, px_x, corner_x, :σ_x, 3),
                    (2, px_y, corner_y, :σ_y, 3),
                    (3, 1.0, 0, :σ_z, 10),
                )
                field = parameter_fields[axis]
                errors = [
                    1000 * (
                            getproperty(gpu.emitters[k], field) -
                            (axis == 3 ? truth[k][axis] :
                                (corner + truth[k][axis] - 1.5) * px)
                        ) for k in 1:n
                ]
                crlb = 1000 * sqrt(mean(
                    abs2, [getproperty(e, sigma) for e in gpu.emitters]
                ))
                @test abs(mean(errors)) <= tolerance
                @test isfinite(crlb) && crlb > 0
                @test abs(1 - std(errors) / crlb) < 0.3
            end
            @test abs(mean(e.photons for e in gpu.emitters) / photons - 1) < 0.01

            if band == 0
                auto, info = fit(
                    batch; psf_model = model, backend = :auto, auto_timeout = 60.0
                )
                @test info.backend == :gpu
                @test length(auto.emitters) == n
            end
        end
    end
end
