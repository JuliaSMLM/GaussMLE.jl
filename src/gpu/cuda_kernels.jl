"""
CUDA kernel implementations for GPU-accelerated fitting
"""

# Include the actual kernel implementations
include("cuda_kernels_clean.jl")

# Main CUDA batch fitting function
function cuda_fit_batch(backend::CUDABackend, data::AbstractArray{T,3}, 
                       modeltype::Type, 
                       variance::Union{Nothing,AbstractArray{T,3}}) where T
    
    n_rois = size(data, 3)
    roi_size = size(data, 1)
    
    # Ensure data is Float32 for GPU efficiency
    FT = T <: Float32 ? T : Float32
    
    # Transfer data to GPU
    d_data = CuArray{FT}(data)
    
    # Allocate output arrays on GPU
    n_params = 4  # x, y, intensity, background
    d_params = CUDA.zeros(FT, n_params, n_rois)
    d_crlb = CUDA.zeros(FT, n_params, n_rois)
    
    # Launch kernel
    launch_gaussian_fit!(d_data, d_params, d_crlb)
    
    # Synchronize and check for errors
    CUDA.synchronize()
    
    # Transfer results back to CPU
    params_cpu = Array(d_params)
    crlb_cpu = Array(d_crlb)
    
    # Convert to expected format (array of structs)
    if modeltype == θ_xynb
        θ_result = [θ_xynb(params_cpu[1,i], params_cpu[2,i], 
                          params_cpu[3,i], params_cpu[4,i]) 
                    for i in 1:n_rois]
        Σ_result = [Σ_xynb(crlb_cpu[1,i], crlb_cpu[2,i], 
                          crlb_cpu[3,i], crlb_cpu[4,i], FT(0)) 
                    for i in 1:n_rois]
    else
        # Fall back to CPU for unsupported models
        @warn "Model type $modeltype not supported in CUDA, falling back to CPU"
        cpu_backend = CPUBackend(Threads.nthreads())
        return fit_batch(cpu_backend, data, modeltype, variance)
    end
    
    return θ_result, Σ_result
end

# Alternative entry point for direct CUDA array processing
function cuda_fit_batch(data::CuArray{T,3}, modeltype::Type, 
                       variance::Union{Nothing,CuArray{T,3}}=nothing) where T
    
    n_rois = size(data, 3)
    roi_size = size(data, 1)
    n_params = 4  # x, y, intensity, background
    
    # Allocate output arrays on GPU
    d_params = CUDA.zeros(T, n_params, n_rois)
    d_crlb = CUDA.zeros(T, n_params, n_rois)
    
    # Launch kernel
    launch_gaussian_fit!(data, d_params, d_crlb)
    
    # Synchronize
    CUDA.synchronize()
    
    return d_params, d_crlb
end