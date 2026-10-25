# Regression tests for GPU free-memory queries and backend fallback behavior.
using Test, GaussMLE, CUDA, SMLMData

# The NVML-unsupported fallback is reachable only on a GPU without NVML memory info
# (the DGX Spark GB10), so _gpu_free_memory is the regression point for that bug.
@testset "GPU free-memory queries" begin
    @testset "NVML reports free memory" begin
        free, from_nvml = GaussMLE._gpu_free_memory(0; nvml_free = _ -> 123_456)
        @test free isa Int
        @test free == 123_456
        @test from_nvml === true
    end

    @testset "Live free-memory query" begin
        free, from_nvml = GaussMLE._gpu_free_memory(0)
        @test free isa Int
        @test free > 0
        @test from_nvml isa Bool
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

    @testset "Polling errors respect backend and interrupts" begin
        required_bytes = 10^15
        failure = ErrorException("poll callback failure")
        on_wait = (_, _, _) -> throw(failure)

        @testset "Automatic backend falls back with a warning" begin
            warning = r"GPU availability check failed, using CPU"
            device = @test_logs (:warn, warning) match_mode = :any begin
                GaussMLE.select_backend(
                    :auto, required_bytes; auto_timeout = 2.0, on_wait = on_wait
                )
            end
            @test device isa GaussMLE.CPU
        end

        @testset "Explicit GPU propagates polling failure" begin
            @test_throws failure GaussMLE.select_backend(
                :gpu, required_bytes; gpu_timeout = 2.0, on_wait = on_wait
            )
        end

        @testset "Automatic backend propagates interrupts" begin
            on_interrupt = (_, _, _) -> throw(InterruptException())
            @test_throws InterruptException GaussMLE.select_backend(
                :auto, required_bytes; auto_timeout = 2.0, on_wait = on_interrupt
            )
        end
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
        @testset "backend = $backend" for backend in (:gpu, :auto)
            config = GaussMLEConfig(
                psf_model = model, backend = backend,
                gpu_timeout = 30.0, auto_timeout = 30.0
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
end
