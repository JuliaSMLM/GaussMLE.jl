"""
Camera noise models for different detector types
"""

"""
    CameraModel

Abstract type for camera noise models used in likelihood calculations.
"""
abstract type CameraModel end

"""
    IdealCamera <: CameraModel

Ideal camera model with Poisson noise only.

Assumes photon counting statistics with no additional readout noise. Suitable for:
- EMCCD cameras in photon-counting mode
- Ideal simulations
- Quick fitting when readout noise is negligible

# Example
```julia
fitter = GaussMLEFitter(camera_model = IdealCamera())
```

# See also
[`SCMOSCamera`](@ref) for cameras with per-pixel readout noise
"""
struct IdealCamera <: CameraModel end

# sCMOS camera with Poisson noise + pixel-dependent readout noise
# Internal type for fitting - renamed to avoid conflict with SMLMData-compatible version
struct SCMOSCameraInternal{T} <: CameraModel
    variance_map::T  # Pixel-wise variance (readout noise²)
    gain_map::T      # Optional pixel-wise gain calibration
    
    function SCMOSCameraInternal(variance_map::T, gain_map=nothing) where T
        if isnothing(gain_map)
            gain_map = ones(eltype(variance_map), size(variance_map))
        end
        new{T}(variance_map, gain_map)
    end
end

# Noise model interface for likelihood calculations
@inline function compute_likelihood_terms(data::T, model::T, ::IdealCamera) where T
    # Poisson noise only
    # Cap values to prevent numerical instability (following SMITE)
    # Note: SMITE uses 10e-3 = 0.01, not 1e-3 = 0.001
    if model > T(0.01)
        cf = data / model - one(T)
        df = data / (model * model)
        # Cap at 10^4 to prevent instability
        cf = min(cf, T(1e4))
        df = min(df, T(1e4))
    else
        cf = zero(T)
        df = zero(T)
    end
    return cf, df
end

@inline function compute_likelihood_terms(data::T, model::T, camera::SCMOSCameraInternal, i, j) where T
    # Total variance = Poisson variance + readout variance
    total_var = model + camera.variance_map[i, j]
    cf = (data - model) / total_var
    df = one(T) / total_var  # Corrected: ∂²L/∂model² = -1/variance, but we need positive for Hessian
    return cf, df
end

# Log-likelihood computation
@inline function compute_log_likelihood(data::T, model::T, ::IdealCamera) where T
    if model > zero(T) && data > zero(T)
        # Simplified Poisson log-likelihood without lgamma term (constant w.r.t. parameters)
        return data * log(model) - model
    else
        return zero(T)
    end
end

@inline function compute_log_likelihood(data::T, model::T, camera::SCMOSCameraInternal, i, j) where T
    # Gaussian approximation for sCMOS noise
    total_var = model + camera.variance_map[i, j]
    if total_var > zero(T)
        residual = data - model
        return -T(0.5) * (log(T(2π) * total_var) + residual^2 / total_var)
    else
        return zero(T)
    end
end

# Support for SMLMData SCMOSCamera (which is not a CameraModel subtype)
@inline function compute_likelihood_terms(data::T, model::T, camera::SCMOSCamera, i, j) where T
    # Total variance = Poisson variance + readout variance
    total_var = model + camera.readnoise_variance[i, j]
    cf = (data - model) / total_var
    df = one(T) / total_var
    return cf, df
end

@inline function compute_log_likelihood(data::T, model::T, camera::SCMOSCamera, i, j) where T
    # Gaussian approximation for sCMOS noise
    total_var = model + camera.readnoise_variance[i, j]
    if total_var > zero(T)
        residual = data - model
        return -T(0.5) * (log(T(2π) * total_var) + residual^2 / total_var)
    else
        return zero(T)
    end
end

# Export types and functions
export CameraModel, IdealCamera, SCMOSCameraInternal
export compute_likelihood_terms, compute_log_likelihood