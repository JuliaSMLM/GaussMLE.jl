"""
High-level API for Gaussian MLE fitting
"""

using KernelAbstractions

# Include CPU kernel
include("cpu_kernel.jl")

# Main fitter type
struct GaussMLEFitter{D<:ComputeDevice, P<:PSFModel, C<:CameraModel, PC<:ParameterConstraints}
    device::D
    psf_model::P
    camera_model::C
    iterations::Int
    constraints::PC
    batch_size::Int
end

# Convenient constructor with smart defaults
function GaussMLEFitter(;
    psf_model = GaussianXYNB(1.3f0),
    camera_model = IdealCamera(),
    device = nothing,  # auto-detect if nothing
    iterations = 20,
    constraints = nothing,
    batch_size = 10_000
)
    # Handle symbol device specification
    device = if device == :cpu
        CPU()
    elseif device == :gpu
        GPU()
    elseif device == :auto || isnothing(device)
        auto_device()
    else
        device  # Already a ComputeDevice
    end
    device = select_device(device)
    
    # Default constraints based on typical box size
    if isnothing(constraints)
        constraints = default_constraints(psf_model, 11)  # typical 11x11 box
    end
    
    return GaussMLEFitter(device, psf_model, camera_model, 
                          iterations, constraints, batch_size)
end

# Main fitting function
function fit(fitter::GaussMLEFitter, data::AbstractArray{T,3}; 
             variance_map=nothing) where T
    
    n_fits = size(data, 3)
    n_params = length(fitter.psf_model)
    box_size = size(data, 1)
    
    # Update camera model if variance map provided
    camera = if !isnothing(variance_map) && fitter.camera_model isa IdealCamera
        @info "Variance map provided, switching to sCMOS noise model"
        SCMOSCamera(variance_map)
    else
        fitter.camera_model
    end
    
    # Convert data to Float32 if needed
    data_f32 = convert(Array{Float32,3}, data)
    
    # Allocate result arrays
    results = Matrix{Float32}(undef, n_params, n_fits)
    uncertainties = Matrix{Float32}(undef, n_params, n_fits)
    log_likelihoods = Vector{Float32}(undef, n_fits)
    
    # Use CPU or GPU kernel based on device
    if fitter.device isa CPU
        # Use CPU kernel
        cpu_fit_batch!(
            results, uncertainties, log_likelihoods,
            data_f32, fitter.psf_model, camera,
            fitter.constraints, fitter.iterations
        )
    else
        # Use GPU kernel (KernelAbstractions)
        # Process in batches for memory efficiency
        for batch_start in 1:fitter.batch_size:n_fits
            batch_end = min(batch_start + fitter.batch_size - 1, n_fits)
            batch_size = batch_end - batch_start + 1
            
            # Get batch data
            batch_data = data_f32[:, :, batch_start:batch_end]
            
            # Move batch to device using KernelAbstractions adapt
            d_batch_data = KernelAbstractions.allocate(backend(fitter.device), Float32, size(batch_data))
            copyto!(d_batch_data, batch_data)
            
            # Allocate device arrays for results
            d_results = KernelAbstractions.allocate(backend(fitter.device), Float32, (n_params, batch_size))
            d_uncertainties = KernelAbstractions.allocate(backend(fitter.device), Float32, (n_params, batch_size))
            d_log_likelihoods = KernelAbstractions.allocate(backend(fitter.device), Float32, batch_size)
            
            # Launch kernel
            kernel = gaussian_mle_kernel!(backend(fitter.device))
            kernel(d_results, d_uncertainties, d_log_likelihoods,
                   d_batch_data, fitter.psf_model, camera, 
                   fitter.constraints, fitter.iterations,
                   ndrange=batch_size)
            
            # Wait for kernel completion
            KernelAbstractions.synchronize(backend(fitter.device))
            
            # Copy results back
            copyto!(view(results, :, batch_start:batch_end), d_results)
            copyto!(view(uncertainties, :, batch_start:batch_end), d_uncertainties)
            copyto!(view(log_likelihoods, batch_start:batch_end), d_log_likelihoods)
        end
    end
    
    # Return structured results
    return GaussMLEResults(
        results, uncertainties, log_likelihoods, 
        fitter.psf_model, n_fits
    )
end

# Convenience function for single ROI fitting
function fit(fitter::GaussMLEFitter, roi::AbstractMatrix{T}) where T
    # Reshape to 3D array with single ROI
    data = reshape(roi, size(roi, 1), size(roi, 2), 1)
    results = fit(fitter, data)
    
    # Return vectors instead of matrices for single fit
    return results.parameters[:, 1], results.uncertainties[:, 1]
end

# Export API
export GaussMLEFitter, fit