"""
Backend abstraction for different compute devices
"""

# Abstract type for all backends
abstract type FittingBackend end

# CPU Backend
struct CPUBackend <: FittingBackend
    nthreads::Int
    
    function CPUBackend(nthreads::Int=Threads.nthreads())
        new(nthreads)
    end
end

# CUDA Backend
struct CUDABackend <: FittingBackend
    device::CuDevice
    compute_capability::VersionNumber
    max_threads_per_block::Int
    max_blocks::Int
    shared_memory_size::Int
    
    function CUDABackend(device::CuDevice=CUDA.device(), 
                        compute_capability::VersionNumber=CUDA.capability(device))
        max_threads = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        max_blocks = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
        shared_mem = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
        new(device, compute_capability, max_threads, max_blocks, shared_mem)
    end
end

# Metal Backend (placeholder for now)
struct MetalBackend <: FittingBackend
    device::Any  # Will be MTLDevice when Metal.jl is loaded
    
    function MetalBackend(device=nothing)
        new(device)
    end
end

# Backend capability queries
backend_name(::CPUBackend) = "CPU"
backend_name(::CUDABackend) = "CUDA"
backend_name(::MetalBackend) = "Metal"

backend_available(::CPUBackend) = true
backend_available(::CUDABackend) = CUDA.functional()
backend_available(b::MetalBackend) = GaussGPU.metal_available()

supports_streaming(::CPUBackend) = false
supports_streaming(::CUDABackend) = true
supports_streaming(::MetalBackend) = true

max_batch_size(b::CPUBackend) = typemax(Int)  # No GPU memory limit
max_batch_size(b::CUDABackend) = estimate_max_batch_size(b)
max_batch_size(b::MetalBackend) = estimate_max_batch_size(b)

# Estimate maximum batch size based on available GPU memory
function estimate_max_batch_size(backend::CUDABackend, roi_size::Int=7, 
                               element_size::Int=sizeof(Float32))
    # Get available memory
    free_mem, total_mem = CUDA.available_memory()
    
    # Conservative estimate: use 80% of free memory
    usable_memory = floor(Int, 0.8 * free_mem)
    
    # Memory per ROI: input + output + workspace
    # Input: roi_size^2 * element_size
    # Output: ~5 parameters * element_size
    # Workspace: ~100 bytes for intermediate calculations
    memory_per_roi = roi_size^2 * element_size + 5 * element_size + 100
    
    # Calculate batch size
    batch_size = floor(Int, usable_memory / memory_per_roi)
    
    # Cap at reasonable maximum
    return min(batch_size, 1_000_000)
end

function estimate_max_batch_size(backend::MetalBackend, roi_size::Int=7,
                               element_size::Int=sizeof(Float32))
    # Placeholder - will implement when Metal support is added
    return 100_000
end

# Get optimal thread configuration for kernels
function optimal_thread_config(backend::CUDABackend, n_elements::Int)
    # Use warp-aligned thread counts for better performance
    threads_per_block = min(256, backend.max_threads_per_block)
    blocks = cld(n_elements, threads_per_block)
    
    # Ensure we don't exceed device limits
    blocks = min(blocks, backend.max_blocks)
    
    return (threads=threads_per_block, blocks=blocks)
end

function optimal_thread_config(backend::CPUBackend, n_elements::Int)
    # For CPU, use available threads
    return (threads=backend.nthreads, blocks=1)
end

function optimal_thread_config(backend::MetalBackend, n_elements::Int)
    # Placeholder for Metal
    return (threads=256, blocks=cld(n_elements, 256))
end