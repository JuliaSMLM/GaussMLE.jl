"""
High-level API for Gaussian MLE fitting
"""

using KernelAbstractions
using CUDA
using SMLMData

# Input validation helpers
function validate_fit_input(data::AbstractArray{T,3}) where T
    # Check for empty data
    if isempty(data)
        throw(ArgumentError("Input data array is empty"))
    end

    # Check for square ROIs
    if size(data, 1) != size(data, 2)
        throw(ArgumentError("ROIs must be square, got size $(size(data, 1))×$(size(data, 2))"))
    end

    # Check for minimum ROI size
    if size(data, 1) < 3
        throw(ArgumentError("ROI size must be at least 3×3, got $(size(data, 1))×$(size(data, 1))"))
    end

    # Check for NaN or Inf values
    if any(!isfinite, data)
        throw(ArgumentError("Input data contains NaN or Inf values"))
    end

    # Check for negative values
    if any(<(0), data)
        @warn "Input data contains negative values, which may indicate preprocessing issues"
    end

    return true
end

# Camera parameter preparation for kernel dispatch
function prepare_camera_params(camera::IdealCamera, box_size, ::Type{T}) where T
    # Return compile-time flag and dummy variance map (will be ignored)
    return Val(false), ones(T, box_size, box_size)
end

function prepare_camera_params(camera::SCMOSCameraInternal, box_size, ::Type{T}) where T
    # Return compile-time flag and variance map from internal camera
    return Val(true), T.(camera.variance_map)
end

function prepare_camera_params(camera::SCMOSCamera, box_size, ::Type{T}) where T
    # Return compile-time flag and variance map from SMLMData camera
    # Note: SMLMData uses 'readnoise_variance' field name
    return Val(true), T.(camera.readnoise_variance)
end

"""
    GaussMLEFitter{D,P,C,PC}

Main type for configuring and performing Maximum Likelihood Estimation of Gaussian blob parameters.

# Fields
- `device::D<:ComputeDevice`: Compute device (CPU or GPU)
- `psf_model::P<:PSFModel`: Point spread function model
- `camera_model::C`: Camera noise model (IdealCamera or SCMOSCamera)
- `iterations::Int`: Number of Newton-Raphson iterations
- `constraints::PC<:ParameterConstraints`: Parameter bounds and step limits
- `batch_size::Int`: Batch size for GPU processing

# See also
[`fit`](@ref), [`GaussMLEResults`](@ref), [`PSFModel`](@ref), [`CameraModel`](@ref)
"""
struct GaussMLEFitter{D<:ComputeDevice, P<:PSFModel, C, PC<:ParameterConstraints}
    device::D
    psf_model::P
    camera_model::C  # Can be CameraModel or SMLMData.SCMOSCamera
    iterations::Int
    constraints::PC
    batch_size::Int
end

"""
    GaussMLEFitter(; kwargs...)

Create a fitter for Gaussian MLE with sensible defaults.

# Keyword Arguments
- `psf_model = GaussianXYNB(1.3f0)`: PSF model to use
- `camera_model = IdealCamera()`: Camera noise model
- `device = nothing`: Compute device (`:cpu`, `:gpu`, `:auto`, or nothing for auto-detect)
- `iterations = 20`: Number of Newton-Raphson iterations
- `constraints = nothing`: Parameter constraints (uses defaults if nothing)
- `batch_size = 10_000`: Number of ROIs to process per GPU batch

# Examples
```julia
# Simple usage with auto GPU detection
fitter = GaussMLEFitter()

# Force CPU execution
cpu_fitter = GaussMLEFitter(device = :cpu)

# Variable sigma model with sCMOS camera
fitter = GaussMLEFitter(
    psf_model = GaussianXYNBS(),
    camera_model = SCMOSCamera(variance_map),
    iterations = 30
)
```

# See also
[`fit`](@ref), [`GaussianXYNB`](@ref), [`GaussianXYNBS`](@ref), [`IdealCamera`](@ref)
"""
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
        if !CUDA.functional()
            @warn "GPU requested but CUDA is not available, falling back to CPU"
            CPU()
        else
            GPU()
        end
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

"""
    fit(fitter::GaussMLEFitter, data::AbstractArray{T,3}; variance_map=nothing) -> GaussMLEResults

Fit Gaussian blobs to a stack of ROIs using Maximum Likelihood Estimation.

# Arguments
- `fitter::GaussMLEFitter`: Configured fitter object
- `data::AbstractArray{T,3}`: ROI data as (roi_size, roi_size, n_rois) array

# Keyword Arguments
- `variance_map=nothing`: Optional sCMOS variance map (will override fitter's camera model)

# Returns
- `GaussMLEResults`: Fitted parameters, uncertainties, and log-likelihoods

# Examples
```julia
# Fit 1000 ROIs
data = zeros(Float32, 7, 7, 1000)  # Your ROI data here
fitter = GaussMLEFitter(psf_model = GaussianXYNB(1.3f0))
results = fit(fitter, data)

# Access results
println("Mean x position: ", mean(results.x))
println("Mean localization precision: ", mean(results.x_error))
```

# See also
[`GaussMLEFitter`](@ref), [`GaussMLEResults`](@ref)
"""
function fit(fitter::GaussMLEFitter, data::AbstractArray{T,3};
             variance_map=nothing) where T

    # Validate input
    validate_fit_input(data)

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
    
    # Prepare camera parameters for kernel
    use_scmos, variance_map = prepare_camera_params(camera, box_size, Float32)

    # Use unified kernel for both CPU and GPU
    if fitter.device isa CPU
        # Use unified kernel on CPU
        backend = KernelAbstractions.CPU()
        kernel = unified_gaussian_mle_kernel!(backend)
        kernel(results, uncertainties, log_likelihoods,
               data_f32, fitter.psf_model, use_scmos, variance_map,
               fitter.constraints, fitter.iterations,
               ndrange=n_fits)
        KernelAbstractions.synchronize(backend)
    else
        # Use unified kernel on GPU
        # Process in batches for memory efficiency
        for batch_start in 1:fitter.batch_size:n_fits
            batch_end = min(batch_start + fitter.batch_size - 1, n_fits)
            batch_size = batch_end - batch_start + 1
            
            # Get batch data
            batch_data = data_f32[:, :, batch_start:batch_end]
            
            # Move batch to device using KernelAbstractions adapt
            d_batch_data = KernelAbstractions.allocate(GaussMLE.backend(fitter.device), Float32, size(batch_data))
            copyto!(d_batch_data, batch_data)

            # Move variance map to device (same for all batches)
            d_variance_map = KernelAbstractions.allocate(GaussMLE.backend(fitter.device), Float32, size(variance_map))
            copyto!(d_variance_map, variance_map)

            # Allocate device arrays for results
            d_results = KernelAbstractions.allocate(GaussMLE.backend(fitter.device), Float32, (n_params, batch_size))
            d_uncertainties = KernelAbstractions.allocate(GaussMLE.backend(fitter.device), Float32, (n_params, batch_size))
            d_log_likelihoods = KernelAbstractions.allocate(GaussMLE.backend(fitter.device), Float32, batch_size)

            # Launch unified kernel (works on GPU!)
            kernel = unified_gaussian_mle_kernel!(GaussMLE.backend(fitter.device))
            kernel(d_results, d_uncertainties, d_log_likelihoods,
                   d_batch_data, fitter.psf_model, use_scmos, d_variance_map,
                   fitter.constraints, fitter.iterations,
                   ndrange=batch_size)
            
            # Wait for kernel completion
            KernelAbstractions.synchronize(GaussMLE.backend(fitter.device))
            
            # Copy results back (need to copy to host arrays first)
            results[:, batch_start:batch_end] = Array(d_results)
            uncertainties[:, batch_start:batch_end] = Array(d_uncertainties)
            log_likelihoods[batch_start:batch_end] = Array(d_log_likelihoods)
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

# Fit method for ROIBatch - dispatches based on camera type
function fit(fitter::GaussMLEFitter, roi_batch::ROIBatch{T,N,A,<:SMLMData.IdealCamera}) where {T,N,A}
    # For IdealCamera, use standard fitting
    n_fits = length(roi_batch)
    n_params = length(fitter.psf_model)
    
    # Use standard fit on the data
    results = fit(fitter, roi_batch.data)
    
    # Convert to LocalizationResult with camera coordinates
    return create_localization_result(
        results.parameters, 
        results.uncertainties,
        results.log_likelihoods,
        roi_batch, 
        fitter.psf_model
    )
end

# Fit method for ROIBatch with SCMOSCamera
function fit(fitter::GaussMLEFitter, roi_batch::ROIBatch{T,N,A,<:SCMOSCamera}) where {T,N,A}
    
    # Create a new fitter with the SCMOSCamera from the ROIBatch
    fitter_with_camera = GaussMLEFitter(
        fitter.device,
        fitter.psf_model,
        roi_batch.camera,  # Use the SCMOSCamera from ROIBatch
        fitter.iterations,
        fitter.constraints,
        fitter.batch_size
    )
    
    # Use standard fit with the updated fitter
    results = fit(fitter_with_camera, roi_batch.data)
    
    # Return LocalizationResult with camera coordinates
    return create_localization_result(
        results.parameters,
        results.uncertainties,
        results.log_likelihoods,
        roi_batch, 
        fitter.psf_model
    )
end


# Export API
export GaussMLEFitter, fit