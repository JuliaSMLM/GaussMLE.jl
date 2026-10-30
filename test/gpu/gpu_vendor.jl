# Vendor routing and the public fit contract on either CUDA or AMD GPUs.
using Test, GaussMLE

include(joinpath(@__DIR__, "utils", "gpu_vendor.jl"))

@testset "GPU vendor routing" begin
    vendor = GaussMLE._gpu_vendor()
    expected = CUDA.functional() ? GaussMLE.CUDAGPU() : GaussMLE.ROCmGPU()
    @test vendor === expected

    @testset "Memory polling" begin
        @test GaussMLE._poll_gpu(
            vendor, :auto, typemax(Int) >> 2; timeout = 0.0,
        ) == (-1, false, false)
        @test GaussMLE._poll_gpu(vendor, :auto, 1; timeout = 0.0) == (0, true, false)
    end

    @testset "Public fit agrees with CPU" begin
        models = (
            GaussianXYNB(0.13f0),
            AstigmaticXYZNB{Float32}(
                0.13f0, 0.13f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.25f0, 0.4f0,
            ),
        )
        cameras = (
            IdealCamera(64, 64, 0.1f0),
            SCMOSCamera(
                64, 64, 0.1f0, fill(5.0f0, 64, 64);
                offset = 100.0f0, gain = 0.5f0, qe = 0.82f0,
            ),
        )

        @testset "$model, $(typeof(camera))" for model in models, camera in cameras
            batch = generate_roi_batch(
                camera, model; n_rois = 32, roi_size = 11, seed = 42,
            )
            cpu, _ = fit(batch, GaussMLEConfig(psf_model = model, backend = :cpu))
            parameters = model isa AstigmaticXYZNB ? (:x, :y, :z, :photons, :bg) :
                (:x, :y, :photons, :bg)
            uncertainties = model isa AstigmaticXYZNB ?
                (:σ_x, :σ_y, :σ_z, :σ_photons, :σ_bg) : (:σ_x, :σ_y, :σ_photons, :σ_bg)

            @testset "backend = $backend" for backend in (:gpu, :auto)
                gpu, info = fit(
                    batch, GaussMLEConfig(
                        psf_model = model, backend = backend, batch_size = 10,
                        gpu_timeout = 30.0, auto_timeout = 30.0,
                    ),
                )
                @test info.backend == :gpu
                @test info.device_id >= 0
                @test info.n_fits == 32
                @test info.batch_size == 10
                @test info.n_batches == 4
                @test length(gpu.emitters) == length(cpu.emitters)
                for field in parameters
                    @test getproperty.(gpu.emitters, field) ≈
                        getproperty.(cpu.emitters, field) rtol = 1.0e-4
                end
                for field in uncertainties
                    @test getproperty.(gpu.emitters, field) ≈
                        getproperty.(cpu.emitters, field) rtol = 1.0e-3
                end
            end
        end
    end
end
