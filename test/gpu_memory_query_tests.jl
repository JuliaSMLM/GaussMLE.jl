@testset "GPU free-memory queries" begin
    @testset "NVML reports free memory" begin
        free, from_nvml = GaussMLE._gpu_free_memory(0)
        @test free isa Int
        @test free > 0
        @test from_nvml === true
    end

    @testset "Unsupported NVML query falls back to CUDA" begin
        queried_devices = Int[]
        unsupported = CUDA.NVML.NVMLError(CUDA.NVML.NVML_ERROR_NOT_SUPPORTED)
        nvml_free = device_idx -> begin
            push!(queried_devices, device_idx)
            throw(unsupported)
        end
        free, from_nvml = GaussMLE._gpu_free_memory(0; nvml_free = nvml_free)
        @test queried_devices == [0]
        @test free isa Int
        @test free > 0
        @test from_nvml === false
    end

    @testset "Other NVML errors propagate unchanged" begin
        failure = CUDA.NVML.NVMLError(CUDA.NVML.NVML_ERROR_UNKNOWN)
        nvml_free = _ -> throw(failure)
        @test_throws failure GaussMLE._gpu_free_memory(0; nvml_free = nvml_free)
    end

    @testset "GPU fit recovers x = 14 in a 15-pixel ROI" begin
        box_size = 15
        camera = SMLMData.IdealCamera(box_size, box_size, 0.1f0)
        model = GaussianXYNB(0.13f0)
        pixel_size = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
        psf_pixels = GaussMLE.to_pixel_units(model, pixel_size)
        parameters = GaussMLE.Params{4}(14.0f0, 8.0f0, 5000.0f0, 10.0f0)
        data = Array{Float32}(undef, box_size, box_size, 1)
        for j in 1:box_size, i in 1:box_size
            data[i, j, 1] = GaussMLE.evaluate_psf(psf_pixels, i, j, parameters)
        end
        batch = SMLMData.ROIBatch(data, Int32[1], Int32[1], Int32[1], camera)
        config = GaussMLEConfig(
            psf_model = model, backend = :gpu, gpu_timeout = 30.0
        )
        smld, info = fit(batch, config)
        emitter = only(smld.emitters)
        # Invert the same physical-coordinate and corner mapping as src/roi_batch.jl.
        x_camera = (emitter.x - camera.pixel_edges_x[1]) / pixel_size + 0.5f0
        x_roi = x_camera - batch.x_corners[1] + 1
        @test x_roi ≈ 14.0f0 atol = 0.01f0 rtol = 0
        @test info.backend == :gpu
    end
end
