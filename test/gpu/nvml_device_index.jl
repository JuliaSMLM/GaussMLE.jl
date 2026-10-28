# The NVML device read for a CUDA ordinal must be the GPU that ordinal names; under
# CUDA_VISIBLE_DEVICES the CUDA ordinals and NVML indices differ. Red-to-green check: a
# two-GPU host with CUDA_VISIBLE_DEVICES naming a GPU other than NVML index 0.
using Test, GaussMLE, CUDA

# One outer testset, so the memory checks still run and report when the first set fails.
@testset "NVML device index" begin
    @testset "Contention queries go to the GPU the ordinal names" begin
        for i in 0:(length(CUDA.devices()) - 1)
            expected_uuid = CUDA.parent_uuid(CUDA.CuDevice(i))
            process_devices = typeof(expected_uuid)[]
            utilization_devices = typeof(expected_uuid)[]
            processes = dev -> begin
                push!(process_devices, CUDA.NVML.uuid(dev))
                Dict((Int(getpid()) + 1) => (used_gpu_memory = 0,))
            end
            utilization = dev -> begin
                push!(utilization_devices, CUDA.NVML.uuid(dev))
                (compute = 0.0, memory = 0.0)
            end
            contended = GaussMLE._gpu_contended(
                i; processes = processes, utilization = utilization
            )
            @test process_devices == [expected_uuid]
            @test utilization_devices == [expected_uuid]
            @test !contended
        end
    end

    @testset "Free memory is read from the visible GPU" begin
        CUDA.device!(0)
        CUDA.reclaim()
        _, from_nvml = GaussMLE._gpu_free_memory(0)

        # DGX Spark GB10 has no NVML memory query and shares host memory; filling uses
        # host RAM.
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

                device_idx, available, failed = GaussMLE._poll_gpu(
                    :gpu, cuda_free; timeout = 1.0
                )
                @test !failed
                @test !(available && device_idx == 0)
            finally
                CUDA.unsafe_free!(filler)
                CUDA.reclaim()
            end
        end
    end
end
