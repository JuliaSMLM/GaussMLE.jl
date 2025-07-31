"""
    GaussGPU

GPU acceleration module for GaussMLE.jl
Provides backend abstraction and GPU-accelerated fitting routines.
"""
module GaussGPU

using CUDA
using LinearAlgebra
using GaussMLE.GaussModel
using GaussMLE.GaussModel: MODEL_MAP, genargs, genθ, genΣ, compute_all!, update!
using GaussMLE.GaussModel: θ_xynb, θ_xynbs, Σ_xynb, Σ_xynbs, GaussMLEParams, GaussMLEΣ, GaussMLEArgs
using GaussMLE.GaussModel: initialize_parameters!, model
using GaussMLE.GaussLib
using GaussMLE.GaussFit

# Export backend types and functions
export FittingBackend, CPUBackend, CUDABackend, MetalBackend
export select_backend, backend_available, backend_name
export BatchConfig, fitstack_gpu

# Include sub-modules
include("backends.jl")
include("batching.jl")
include("dispatch.jl")
include("cuda_kernels.jl")

# Conditional Metal support
const METAL_AVAILABLE = try
    eval(:(using Metal))
    true
catch
    false
end

if METAL_AVAILABLE
    include("metal_kernels.jl")
end

"""
    select_backend(; force=nothing)

Automatically select the best available backend.
Use `force` parameter to override auto-selection.
"""
function select_backend(; force::Union{Nothing, Symbol}=nothing)
    if force !== nothing
        if force == :cuda
            CUDA.functional() || error("CUDA requested but not available")
            return CUDABackend()
        elseif force == :metal
            METAL_AVAILABLE || error("Metal requested but not available")
            return MetalBackend()
        elseif force == :cpu
            return CPUBackend()
        else
            error("Unknown backend: $force")
        end
    end
    
    # Auto-selection
    if CUDA.functional()
        device = CUDA.device()
        compute_capability = CUDA.capability(device)
        return CUDABackend(device, compute_capability)
    elseif METAL_AVAILABLE && metal_available()
        return MetalBackend(metal_device())
    else
        return CPUBackend(Threads.nthreads())
    end
end

"""
    fitstack_gpu(data, modelsymbol, backend=select_backend(); kwargs...)

GPU-accelerated version of fitstack.
Automatically handles batching for large datasets.
"""
function fitstack_gpu(data::AbstractArray{T,3}, modelsymbol::Symbol, 
                     backend::FittingBackend=select_backend();
                     batch_config::Union{Nothing,BatchConfig}=nothing,
                     variance::Union{Nothing,AbstractArray{T,3}}=nothing,
                     verbose::Bool=false) where T
    
    # Default batch configuration
    if batch_config === nothing
        batch_config = optimal_batch_config(backend, size(data))
    end
    
    # Dispatch to backend-specific implementation
    return fit_batched(backend, data, modelsymbol, batch_config, variance, verbose)
end

# Convenience function for checking Metal availability
function metal_available()
    METAL_AVAILABLE && Metal.functional()
end

function metal_device()
    METAL_AVAILABLE ? Metal.current_device() : nothing
end

end # module