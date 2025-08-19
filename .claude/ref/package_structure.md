# GaussMLE.jl - Idiomatic Julia Design

## Core Architecture

### Device Abstraction with Auto-Detection

```julia
module GaussMLE

using KernelAbstractions
using CUDA
using StaticArrays
using LinearAlgebra

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
backend(::CPU) = CPU()
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
```

### Camera Noise Models

```julia
# Camera noise models
abstract type CameraModel end

# Ideal camera with only Poisson noise
struct IdealCamera <: CameraModel end

# sCMOS camera with Poisson noise + pixel-dependent readout noise
struct SCMOSCamera{T} <: CameraModel
    variance_map::T  # Pixel-wise variance (readout noise²)
    gain_map::T      # Optional pixel-wise gain calibration
    
    function SCMOSCamera(variance_map::T, gain_map=nothing) where T
        if isnothing(gain_map)
            gain_map = ones(eltype(variance_map), size(variance_map))
        end
        new{T}(variance_map, gain_map)
    end
end

# Noise model interface
@inline function compute_likelihood_terms(data::T, model::T, ::IdealCamera) where T
    cf = data / model - one(T)
    df = data / (model * model)
    return cf, df
end

@inline function compute_likelihood_terms(data::T, model::T, camera::SCMOSCamera, i, j) where T
    # Total variance = Poisson variance + readout variance
    total_var = model + camera.variance_map[i, j]
    cf = (data - model) / total_var
    df = (data + camera.variance_map[i, j]) / (total_var * total_var)
    return cf, df
end
```

### PSF Models as Parametric Types

```julia
# PSF Models with compile-time known parameter counts
abstract type PSFModel{NParams} end

# Fixed sigma Gaussian
struct GaussianXYNB{T} <: PSFModel{4}
    σ::T
end

# Variable sigma Gaussian
struct GaussianXYNBS <: PSFModel{5} end

# Anisotropic Gaussian
struct GaussianXYNBSXSY <: PSFModel{6} end

# Astigmatic 3D PSF
struct AstigmaticXYZNB{T} <: PSFModel{5}
    σx₀::T
    σy₀::T
    Ax::T; Ay::T
    Bx::T; By::T
    γ::T
    d::T
end

# Parameter type aliases
const Params{N} = SVector{N, Float32}

# Efficient small matrix inversion using LU decomposition
@inline function invert_small_matrix(M::SMatrix{N,N,T}) where {N,T}
    # For small matrices (N ≤ 6), LU decomposition is optimal
    return inv(lu(M))
end

# Efficient 1D integrated Gaussian
@inline function integrated_gaussian_1d(x::T, σ::T) where T
    norm = one(T) / (sqrt(T(2)) * σ)
    return T(0.5) * (erf((x + one(T)) * norm) - erf(x * norm))
end

# PSF evaluation interface
@inline function evaluate_psf(psf::GaussianXYNB, i, j, θ::Params{4})
    x, y, N, bg = θ
    psf_x = integrated_gaussian_1d(i - x, psf.σ)
    psf_y = integrated_gaussian_1d(j - y, psf.σ)
    return bg + N * psf_x * psf_y
end

@inline function evaluate_psf(::GaussianXYNBS, i, j, θ::Params{5})
    x, y, N, bg, σ = θ
    psf_x = integrated_gaussian_1d(i - x, σ)
    psf_y = integrated_gaussian_1d(j - y, σ)
    return bg + N * psf_x * psf_y
end
```

### Parameter Constraints

```julia
# Flexible constraint system
struct ParameterConstraints{N}
    lower::Params{N}
    upper::Params{N}
    max_step::Params{N}
end

# Default constraints for each model
function default_constraints(::GaussianXYNB, box_size)
    return ParameterConstraints(
        Params{4}(-2.0f0, -2.0f0, 1.0f0, 0.01f0),        # lower bounds
        Params{4}(box_size+2, box_size+2, Inf32, Inf32), # upper bounds
        Params{4}(1.0f0, 1.0f0, Inf32, Inf32)            # max step
    )
end

@inline function apply_constraints!(θ::Params{N}, Δθ::Params{N}, 
                                   constraints::ParameterConstraints{N}) where N
    # Apply step size limits and bounds
    θ_new = θ - clamp.(Δθ, -constraints.max_step, constraints.max_step)
    return clamp.(θ_new, constraints.lower, constraints.upper)
end
```

### Main Fitting Kernel

```julia
@kernel function gaussian_mle_kernel!(
    results::AbstractArray{T,2},
    uncertainties::AbstractArray{T,2},
    log_likelihoods::AbstractArray{T,1},
    @Const(data::AbstractArray{T,3}),
    @Const(psf_model),
    @Const(camera_model),
    @Const(constraints),
    iterations::Int
) where T
    idx = @index(Global)
    
    # Get the data for this fit
    box_size = size(data, 1)
    roi = @inbounds view(data, :, :, idx)
    
    # Initialize parameters
    θ = initialize_parameters(roi, psf_model)
    
    # Allocate small working arrays
    N = length(θ)
    ∇L = @MVector zeros(T, N)  # Gradient (numerator)
    H_diag = @MVector zeros(T, N)  # Diagonal Hessian (denominator) for NR
    H = @MMatrix zeros(T, N, N) # Full matrix for Fisher Information later
    
    # Newton-Raphson iterations
    # Note: This uses diagonal Newton-Raphson updates (element-wise division)
    # rather than full Newton method (matrix inversion) for stability
    for iter in 1:iterations
        fill!(∇L, zero(T))
        fill!(H_diag, zero(T))
        
        # Compute derivatives over all pixels
        for j in 1:box_size, i in 1:box_size
            # Model and derivatives at this pixel
            model, dudt, d2udt2 = compute_pixel_derivatives(i, j, θ, psf_model)
            
            # Likelihood terms
            data_ij = roi[i, j]
            cf, df = compute_likelihood_terms(data_ij, model, camera_model, i, j)
            
            # Accumulate gradient and diagonal Hessian for Newton-Raphson
            for k in 1:N
                ∇L[k] += dudt[k] * cf
                H_diag[k] += d2udt2[k,k] * cf - dudt[k] * dudt[k] * df
            end
        end
        
        # Newton-Raphson update (element-wise, not full Newton)
        θ_new = @MVector zeros(T, N)
        for k in 1:N
            if abs(H_diag[k]) > eps(T)
                Δθ_k = ∇L[k] / H_diag[k]
                θ_new[k] = θ[k] - clamp(Δθ_k, -constraints.max_step[k], constraints.max_step[k])
            else
                θ_new[k] = θ[k]
            end
        end
        θ = clamp.(θ_new, constraints.lower, constraints.upper)
    end
    
    # Compute final log-likelihood and CRLB
    log_likelihood = zero(T)
    fill!(H, zero(T))  # Now compute full Fisher matrix
    
    for j in 1:box_size, i in 1:box_size
        model, dudt, _ = compute_pixel_derivatives(i, j, θ, psf_model)
        data_ij = roi[i, j]
        
        # Log-likelihood contribution
        log_likelihood += compute_log_likelihood(data_ij, model, camera_model, i, j)
        
        # Fisher Information Matrix (full matrix needed for CRLB)
        for k in 1:N, l in k:N
            F_kl = dudt[k] * dudt[l] / model
            H[k,l] += F_kl
            k != l && (H[l,k] += F_kl)  # Symmetric
        end
    end
    
    # Invert Fisher matrix for uncertainties (CRLB) using LU decomposition
    # For small matrices (4-6 params), LU is efficient and stable
    H_inv = invert_small_matrix(SMatrix{N,N}(H))
    
    # Store results
    @inbounds results[:, idx] = θ
    @inbounds uncertainties[:, idx] = sqrt.(diag(H_inv))
    @inbounds log_likelihoods[idx] = log_likelihood
end
```

### High-Level API

```julia
# Main fitter type
struct GaussMLEFitter{D<:ComputeDevice, P<:PSFModel, C<:CameraModel}
    device::D
    psf_model::P
    camera_model::C
    iterations::Int
    constraints::ParameterConstraints
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
    
    # Update camera model if variance map provided
    camera = if !isnothing(variance_map) && fitter.camera_model isa IdealCamera
        @info "Variance map provided, switching to sCMOS noise model"
        SCMOSCamera(variance_map)
    else
        fitter.camera_model
    end
    
    # Allocate result arrays
    results = Matrix{T}(undef, n_params, n_fits)
    uncertainties = Matrix{T}(undef, n_params, n_fits)
    log_likelihoods = Vector{T}(undef, n_fits)
    
    # Process in batches for memory efficiency
    for batch_start in 1:fitter.batch_size:n_fits
        batch_end = min(batch_start + fitter.batch_size - 1, n_fits)
        batch_size = batch_end - batch_start + 1
        
        # Move batch to device
        batch_data = adapt(backend(fitter.device), 
                          data[:, :, batch_start:batch_end])
        
        # Allocate device arrays for results
        d_results = similar(batch_data, n_params, batch_size)
        d_uncertainties = similar(d_results)
        d_log_likelihoods = similar(batch_data, batch_size)
        
        # Launch kernel
        kernel = gaussian_mle_kernel!(backend(fitter.device))
        kernel(d_results, d_uncertainties, d_log_likelihoods,
               batch_data, fitter.psf_model, camera, 
               fitter.constraints, fitter.iterations,
               ndrange=batch_size)
        
        # Copy results back
        results[:, batch_start:batch_end] = Array(d_results)
        uncertainties[:, batch_start:batch_end] = Array(d_uncertainties)
        log_likelihoods[batch_start:batch_end] = Array(d_log_likelihoods)
    end
    
    # Return structured results
    return GaussMLEResults(
        results, uncertainties, log_likelihoods, 
        fitter.psf_model, n_fits
    )
end

# Results structure
struct GaussMLEResults{T, P<:PSFModel}
    parameters::Matrix{T}
    uncertainties::Matrix{T}
    log_likelihoods::Vector{T}
    psf_model::P
    n_fits::Int
end

# Convenient accessors
Base.getproperty(r::GaussMLEResults{T, <:PSFModel{4}}, s::Symbol) where T = 
    s === :x ? r.parameters[1, :] :
    s === :y ? r.parameters[2, :] :
    s === :photons ? r.parameters[3, :] :
    s === :background ? r.parameters[4, :] :
    s === :x_error ? r.uncertainties[1, :] :
    s === :y_error ? r.uncertainties[2, :] :
    getfield(r, s)

end # module
```

### Usage Examples

```julia
using GaussMLE

# Simple usage with auto-detection
fitter = GaussMLEFitter()
results = fit(fitter, data)

# Explicit CPU usage
cpu_fitter = GaussMLEFitter(device=CPU())

# sCMOS camera with variance map
scmos_fitter = GaussMLEFitter(
    camera_model = SCMOSCamera(variance_map),
    psf_model = GaussianXYNBS()  # variable sigma
)

# Access results
println("Found $(results.n_fits) molecules")
println("Mean localization precision: $(mean(results.x_error)) pixels")

# Custom constraints for challenging data
tight_constraints = ParameterConstraints(
    Params{4}(-1.0f0, -1.0f0, 10.0f0, 0.1f0),
    Params{4}(12.0f0, 12.0f0, 1e5f0, 100.0f0),
    Params{4}(0.5f0, 0.5f0, 1e4f0, 50.0f0)
)

custom_fitter = GaussMLEFitter(
    psf_model = GaussianXYNB(1.5f0),
    constraints = tight_constraints,
    iterations = 30
)
```

## Key Design Features

1. **Automatic Device Selection**: Detects GPU availability and falls back to CPU gracefully
2. **Type-Stable Architecture**: Uses parametric types for compile-time optimization
3. **Camera Model Abstraction**: Clean separation between ideal and sCMOS cameras
4. **Efficient Small Arrays**: StaticArrays for stack-allocated performance
5. **Batch Processing**: Handles large datasets without memory issues
6. **Extensible Design**: Easy to add new PSF models or camera types
7. **User-Friendly API**: Simple interface with sensible defaults
8. **KernelAbstractions**: Single kernel implementation for all devices