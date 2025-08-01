"""
    fitstack(roi_stack::AbstractArray{T,3}, model::Symbol; 
             σ_PSF=1.3, backend=:cpu, variance=nothing, verbose=false) where T

Unified interface for fitting Gaussian models to a stack of ROIs.

# Arguments
- `roi_stack`: 3D array of ROIs (height × width × n_rois)
- `model`: Model type (:xynb, :xynbs, etc.)

# Keyword Arguments
- `σ_PSF`: PSF width in pixels (default: 1.3)
- `backend`: Computing backend (:cpu, :gpu, :auto)
  - `:cpu` (default): Use CPU backend for compatibility
  - `:gpu`: Force GPU backend (errors if unavailable)  
  - `:auto`: Use GPU if available, otherwise CPU
- `variance`: Variance image for sCMOS cameras (optional)
- `verbose`: Print progress information

# Returns
- `θ_found`: Array of fitted parameters
- `Σ_found`: Array of parameter uncertainties (CRLB)

# Examples
```julia
# Basic usage (auto-selects backend)
θ, Σ = fitstack(roi_stack, :xynb; σ_PSF=1.3)

# Force GPU
θ, Σ = fitstack(roi_stack, :xynb; σ_PSF=1.3, backend=:gpu)

# With variance map for sCMOS
θ, Σ = fitstack(roi_stack, :xynb; σ_PSF=1.3, variance=var_map)
```
"""
function fitstack(roi_stack::AbstractArray{T,3}, model::Symbol; 
                  σ_PSF::Real=1.3,
                  backend::Symbol=:cpu,
                  variance::Union{Nothing,AbstractArray{T,3}}=nothing,
                  verbose::Bool=false) where T <: Real
    
    # Validate backend choice
    if !(backend in [:auto, :cpu, :gpu])
        throw(ArgumentError("backend must be :auto, :cpu, or :gpu"))
    end
    
    # Check if GaussGPU module is available and CUDA is functional
    gpu_available = false
    try
        # GaussGPU is loaded at the parent module level
        gpu_available = GaussMLE.GaussGPU.cuda_available()
    catch
        # GaussGPU module might not be available or CUDA not functional
        gpu_available = false
    end
    
    # Determine actual backend to use
    use_gpu = false
    if backend == :gpu
        # Force GPU - check availability
        if !gpu_available
            error("GPU backend requested but CUDA is not available. " *
                  "Please install CUDA.jl and ensure you have a CUDA-capable GPU.")
        end
        use_gpu = true
        verbose && @info "Using GPU backend (forced)"
    elseif backend == :auto
        # Auto mode - try GPU, fall back to CPU
        if gpu_available
            use_gpu = true
            verbose && @info "Using GPU backend (auto-detected)"
        else
            verbose && @info "Using CPU backend (GPU not available)"
        end
    else
        # CPU explicitly requested
        verbose && @info "Using CPU backend (requested)"
    end
    
    # Create args structure based on model type
    args = if model == :xynb
        GaussMLE.GaussModel.Args_xynb(T(σ_PSF))
    elseif model == :xynbs
        GaussMLE.GaussModel.Args_xynbs(T(σ_PSF))
    else
        error("Unknown model type: $model")
    end
    
    # Call appropriate backend
    if use_gpu
        # GPU path - fitstack_gpu doesn't use args directly
        return GaussMLE.GaussGPU.fitstack_gpu(roi_stack, model;
                                             variance=variance,
                                             verbose=verbose)
    else
        # CPU path - use traditional fitstack
        return fitstack_old(roi_stack, model, args;
                           varimage=isnothing(variance) ? T(0) : variance)
    end
end