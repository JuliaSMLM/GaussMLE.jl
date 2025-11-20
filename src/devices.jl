"""
Device abstraction with automatic GPU detection
"""

# Device abstraction
abstract type ComputeDevice end
struct CPU <: ComputeDevice end
struct GPU <: ComputeDevice end

# Smart device selection
function auto_device()
    if CUDA.functional()
        @info "CUDA GPU detected, using GPU acceleration"
        return GPU()
    else
        @info "No CUDA GPU available, using CPU"
        return CPU()
    end
end

# Backend mapping for KernelAbstractions
backend(::CPU) = KernelAbstractions.CPU()
backend(::GPU) = CUDABackend()

# Allow explicit device selection with fallback
function select_device(device::Union{ComputeDevice, Nothing}=nothing)
    if isnothing(device)
        return auto_device()
    elseif device isa GPU && !CUDA.functional()
        @warn "GPU requested but not available, falling back to CPU"
        return CPU()
    else
        return device
    end
end