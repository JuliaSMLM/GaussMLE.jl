"""
Camera noise models for different detector types
"""

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

# Noise model interface for likelihood calculations
@inline function compute_likelihood_terms(data::T, model::T, ::IdealCamera) where T
    # Poisson noise only
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

# Log-likelihood computation
@inline function compute_log_likelihood(data::T, model::T, ::IdealCamera) where T
    if model > zero(T) && data > zero(T)
        # Simplified Poisson log-likelihood without lgamma term (constant w.r.t. parameters)
        return data * log(model) - model
    else
        return zero(T)
    end
end

@inline function compute_log_likelihood(data::T, model::T, camera::SCMOSCamera, i, j) where T
    # Gaussian approximation for sCMOS noise
    total_var = model + camera.variance_map[i, j]
    if total_var > zero(T)
        residual = data - model
        return -T(0.5) * (log(T(2π) * total_var) + residual^2 / total_var)
    else
        return zero(T)
    end
end

# Export types and functions
export CameraModel, IdealCamera, SCMOSCamera
export compute_likelihood_terms, compute_log_likelihood