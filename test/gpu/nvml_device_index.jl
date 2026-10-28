# The NVML device read for a CUDA ordinal must be the GPU that ordinal names; under
# CUDA_VISIBLE_DEVICES the CUDA ordinals and NVML indices differ. Red-to-green check: a
# two-GPU host with CUDA_VISIBLE_DEVICES naming a GPU other than NVML index 0.
using Test, GaussMLE, CUDA

@testset "NVML device matches the CUDA device" begin
    for i in 0:(length(CUDA.devices()) - 1)
        @test CUDA.NVML.uuid(GaussMLE._nvml_device(i)) ==
            CUDA.parent_uuid(CUDA.CuDevice(i))
    end
end

@testset "Free memory is read from the visible GPU" begin
    CUDA.device!(0)
    CUDA.reclaim()
    _, from_nvml = GaussMLE._gpu_free_memory(0)

    # DGX Spark GB10 has no NVML memory query and shares host memory; filling uses host RAM.
    if from_nvml
        # Fill only half so another user of the same GPU keeps memory while this runs.
        chunk = CUDA.free_memory() ÷ 2
        filler = CuArray{UInt8}(undef, chunk)
        try
            CUDA.synchronize()
            cuda_free = CUDA.free_memory()
            free, from_nvml = GaussMLE._gpu_free_memory(0)
            @test from_nvml
            @test free < cuda_free + chunk ÷ 2

            device_idx, available = GaussMLE.wait_for_gpu_nvml(
                cuda_free; timeout = 1.0, poll = 0.1
            )
            @test !(available && device_idx == 0)
        finally
            CUDA.unsafe_free!(filler)
            CUDA.reclaim()
        end
    end
end
