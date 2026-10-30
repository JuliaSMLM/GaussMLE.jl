# Shared GPU test helpers; match fit's preference for CUDA when both vendors work.
using CUDA, AMDGPU

function gpu_backend()
    CUDA.functional() && return CUDA.CUDABackend()
    AMDGPU.functional() && return AMDGPU.ROCBackend()
    return nothing
end

to_gpu(a) = CUDA.functional() ? CUDA.CuArray(a) : AMDGPU.ROCArray(a)
gpu_name() = CUDA.functional() ? CUDA.name(CUDA.device()) : AMDGPU.HIP.name(AMDGPU.device())
