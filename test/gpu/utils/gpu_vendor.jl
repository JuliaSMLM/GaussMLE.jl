# Shared GPU test helpers; match fit's preference for CUDA when both vendors work.
# AMDGPU is optional: a KernelAbstractions 0.9 environment cannot hold it (AMDGPU 2.10
# needs 0.10).
using CUDA
const HAS_AMDGPU = Base.find_package("AMDGPU") !== nothing
HAS_AMDGPU && @eval using AMDGPU

function gpu_backend()
    CUDA.functional() && return CUDA.CUDABackend()
    HAS_AMDGPU && AMDGPU.functional() && return AMDGPU.ROCBackend()
    return nothing
end

to_gpu(a) = CUDA.functional() ? CUDA.CuArray(a) : AMDGPU.ROCArray(a)
gpu_name() = CUDA.functional() ? CUDA.name(CUDA.device()) : AMDGPU.HIP.name(AMDGPU.device())
