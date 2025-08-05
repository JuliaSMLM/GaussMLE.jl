"""
CUDA kernel implementations for GPU-accelerated fitting
"""

# Include the enhanced kernel implementations with proper CRLB
include("cuda_kernels_impl.jl")

# Include z-model specific kernels
include("cuda_kernels_z.jl")

# Main CUDA batch fitting function
function cuda_fit_batch(backend::CUDABackend, data::AbstractArray{T,3}, 
                       modeltype::Type, 
                       variance::Union{Nothing,AbstractArray{T,3}},
                       σ_PSF::Real=1.3,
                       calib::Union{Nothing,AstigmaticCalibration}=nothing) where T
    
    n_rois = size(data, 3)
    roi_size = size(data, 1)
    
    # Ensure data is Float32 for GPU efficiency
    FT = T <: Float32 ? T : Float32
    
    # Transfer data to GPU
    d_data = CuArray{FT}(data)
    
    # Determine number of parameters based on model type
    n_params = if modeltype == θ_xynb
        4  # x, y, intensity, background
    elseif modeltype == θ_xynbs
        5  # x, y, intensity, background, σ_PSF
    elseif modeltype == θ_xynbsxsy
        6  # x, y, intensity, background, σ_x, σ_y
    elseif modeltype == θ_xynbz
        5  # x, y, z, intensity, background
    else
        4  # Default fallback
    end
    
    # Allocate output arrays on GPU
    d_params = CUDA.zeros(FT, n_params, n_rois)
    d_crlb = CUDA.zeros(FT, n_params, n_rois)
    
    # Launch kernel with extended parameters for asymmetric PSF support
    if modeltype == θ_xynbsxsy
        # For asymmetric PSF, pass σ_PSF as both σ_x and σ_y initial values
        launch_gaussian_fit!(d_data, d_params, d_crlb, Int32(n_params), Float32(σ_PSF), Float32(σ_PSF), Float32(σ_PSF))
    elseif modeltype == θ_xynbz
        # For astigmatic z-model, use specialized kernel
        if calib === nothing
            calib = AstigmaticCalibration{FT}()
        end
        launch_gaussian_fit_z!(d_data, d_params, d_crlb, calib)
    else
        # For symmetric PSF models, use standard parameters
        launch_gaussian_fit!(d_data, d_params, d_crlb, Int32(n_params), Float32(σ_PSF))
    end
    
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
    elseif modeltype == θ_xynbs
        θ_result = [θ_xynbs(params_cpu[1,i], params_cpu[2,i], 
                           params_cpu[3,i], params_cpu[4,i], params_cpu[5,i]) 
                    for i in 1:n_rois]
        Σ_result = [Σ_xynbs(crlb_cpu[1,i], crlb_cpu[2,i], 
                           crlb_cpu[3,i], crlb_cpu[4,i], crlb_cpu[5,i], FT(0)) 
                    for i in 1:n_rois]
    elseif modeltype == θ_xynbsxsy
        θ_result = [θ_xynbsxsy(params_cpu[1,i], params_cpu[2,i], 
                              params_cpu[3,i], params_cpu[4,i], 
                              params_cpu[5,i], params_cpu[6,i]) 
                    for i in 1:n_rois]
        Σ_result = [Σ_xynbsxsy(crlb_cpu[1,i], crlb_cpu[2,i], 
                              crlb_cpu[3,i], crlb_cpu[4,i], 
                              crlb_cpu[5,i], crlb_cpu[6,i], FT(0)) 
                    for i in 1:n_rois]
    elseif modeltype == θ_xynbz
        θ_result = [θ_xynbz(params_cpu[1,i], params_cpu[2,i], 
                            params_cpu[3,i], params_cpu[4,i], params_cpu[5,i]) 
                    for i in 1:n_rois]
        Σ_result = [Σ_xynbz(crlb_cpu[1,i], crlb_cpu[2,i], 
                            crlb_cpu[3,i], crlb_cpu[4,i], crlb_cpu[5,i], FT(0)) 
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
                       variance::Union{Nothing,CuArray{T,3}}=nothing,
                       σ_PSF::Real=1.3,
                       calib::Union{Nothing,AstigmaticCalibration}=nothing) where T
    
    n_rois = size(data, 3)
    roi_size = size(data, 1)
    
    # Determine number of parameters based on model type
    n_params = if modeltype == θ_xynb
        4  # x, y, intensity, background
    elseif modeltype == θ_xynbs
        5  # x, y, intensity, background, σ_PSF
    elseif modeltype == θ_xynbsxsy
        6  # x, y, intensity, background, σ_x, σ_y
    elseif modeltype == θ_xynbz
        5  # x, y, z, intensity, background
    else
        4  # Default fallback
    end
    
    # Allocate output arrays on GPU
    d_params = CUDA.zeros(T, n_params, n_rois)
    d_crlb = CUDA.zeros(T, n_params, n_rois)
    
    # Launch kernel with extended parameters for asymmetric PSF support
    if modeltype == θ_xynbsxsy
        # For asymmetric PSF, pass σ_PSF as both σ_x and σ_y initial values
        launch_gaussian_fit!(data, d_params, d_crlb, Int32(n_params), Float32(σ_PSF), Float32(σ_PSF), Float32(σ_PSF))
    elseif modeltype == θ_xynbz
        # For astigmatic z-model, use specialized kernel
        if calib === nothing
            calib = AstigmaticCalibration{T}()
        end
        launch_gaussian_fit_z!(data, d_params, d_crlb, calib)
    else
        # For symmetric PSF models, use standard parameters
        launch_gaussian_fit!(data, d_params, d_crlb, Int32(n_params), Float32(σ_PSF))
    end
    
    # Synchronize
    CUDA.synchronize()
    
    return d_params, d_crlb
end