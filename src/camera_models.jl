"""
Camera noise models for likelihood calculations

Note: This file defines INTERNAL types for likelihood dispatch only.
Users interact with SMLMData.IdealCamera and SMLMData.SCMOSCamera.
"""

# Lightweight marker type for Poisson-only likelihood dispatch
# Not exported - internal use only
struct IdealCameraInternal end

# Internal sCMOS representation with preprocessed variance
# Not exported - internal use only
struct SCMOSCameraInternal{T}
    variance_map::T  # Pixel-wise readout noise variance (e⁻²)
end

# Noise model interface for likelihood calculations
@inline function compute_likelihood_terms(data::T, model::T, ::IdealCameraInternal) where T
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

@inline function compute_likelihood_terms(data::T, model::T, variance_map::AbstractArray, i, j) where T
    # Total variance = Poisson variance + readout variance
    total_var = model + variance_map[i, j]
    cf = (data - model) / total_var
    df = one(T) / total_var  # Corrected: ∂²L/∂model² = -1/variance, but we need positive for Hessian
    return cf, df
end

# Scalar variance (uniform sCMOS readnoise)
@inline function compute_likelihood_terms(data::T, model::T, variance::T, i, j) where T
    # Total variance = Poisson variance + uniform readout variance
    total_var = model + variance
    cf = (data - model) / total_var
    df = one(T) / total_var
    return cf, df
end

# Legacy compatibility: accept SCMOSCameraInternal (for backward compatibility)
@inline function compute_likelihood_terms(data::T, model::T, camera::SCMOSCameraInternal, i, j) where T
    return compute_likelihood_terms(data, model, camera.variance_map, i, j)
end

# Log-likelihood ratio (LLR) computation: log L(fitted) - log L(saturated)
# For goodness-of-fit testing via χ² = -2×LLR ~ χ²(df)
@inline function compute_log_likelihood(data::T, model::T, ::IdealCameraInternal) where T
    if model > zero(T) && data > zero(T)
        # LLR for Poisson: data×log(model) - model - (data×log(data) - data)
        # Matches SMITE's Div calculation (smi_cuda_gaussMLEv2.cu)
        return data * log(model) - model - (data * log(data) - data)
    elseif model > zero(T) && data == zero(T)
        # Limit as data→0: saturated term = 0, fitted term = -model
        return -model
    else
        return zero(T)
    end
end

@inline function compute_log_likelihood(data::T, model::T, variance_map::AbstractArray, i, j) where T
    # LLR for Gaussian (sCMOS): log L(fitted) - log L(saturated)
    # L(saturated) has μ=data, so residual=0: log L = -0.5×log(2π×var)
    # L(fitted): -0.5×[log(2π×var) + residual²/var]
    # LLR = -0.5×residual²/var (constant terms cancel)
    total_var = model + variance_map[i, j]
    if total_var > zero(T)
        residual = data - model
        return -T(0.5) * (residual^2 / total_var)
    else
        return zero(T)
    end
end

# Scalar variance (uniform sCMOS readnoise)
@inline function compute_log_likelihood(data::T, model::T, variance::T, i, j) where T
    total_var = model + variance
    if total_var > zero(T)
        residual = data - model
        return -T(0.5) * (residual^2 / total_var)
    else
        return zero(T)
    end
end

# Legacy compatibility: accept SCMOSCameraInternal (for backward compatibility)
@inline function compute_log_likelihood(data::T, model::T, camera::SCMOSCameraInternal, i, j) where T
    return compute_log_likelihood(data, model, camera.variance_map, i, j)
end

# Preprocessing helpers for SMLMData.SCMOSCamera
"""
    to_electrons(data_adu, camera::SMLMData.SCMOSCamera)

Convert raw ADU data to electrons using camera calibration.
Applies: electrons = (ADU - offset) × gain
"""
function to_electrons(data_adu::AbstractArray, camera)
    # Broadcasting handles both scalar and per-pixel offset/gain
    return (data_adu .- camera.offset) .* camera.gain
end

"""
    get_variance_map(camera::SMLMData.SCMOSCamera)

Extract readout noise variance map in electrons² from camera calibration.
Returns: readnoise² (in e⁻²)
"""
function get_variance_map(camera)
    # Broadcasting handles both scalar and per-pixel readnoise
    return camera.readnoise .^ 2
end