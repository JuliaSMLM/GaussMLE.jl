"""
PSF Models with compile-time known parameter counts
"""

using StaticArrays
using SpecialFunctions: erf

# PSF Models with compile-time known parameter counts
abstract type PSFModel{NParams} end

# Fixed sigma Gaussian (x, y, N, background)
struct GaussianXYNB{T} <: PSFModel{4}
    σ::T
end

# Variable sigma Gaussian (x, y, N, background, σ)
struct GaussianXYNBS <: PSFModel{5} end

# Anisotropic Gaussian (x, y, N, background, σx, σy)
struct GaussianXYNBSXSY <: PSFModel{6} end

# Astigmatic 3D PSF (x, y, z, N, background)
struct AstigmaticXYZNB{T} <: PSFModel{5}
    σx₀::T
    σy₀::T
    Ax::T
    Ay::T
    Bx::T
    By::T
    γ::T
    d::T
    
    function AstigmaticXYZNB{T}(σx₀, σy₀, Ax, Ay, Bx, By, γ, d) where T
        new{T}(T(σx₀), T(σy₀), T(Ax), T(Ay), T(Bx), T(By), T(γ), T(d))
    end
end

# Parameter type aliases
const Params{N} = SVector{N, Float32}

# Get parameter count for a model
Base.length(::Type{<:PSFModel{N}}) where N = N
Base.length(::PSFModel{N}) where N = N

# Efficient 1D integrated Gaussian
@inline function integrated_gaussian_1d(x::T, σ::T) where T
    norm = one(T) / (sqrt(T(2)) * σ)
    return T(0.5) * (erf((x + one(T)) * norm) - erf(x * norm))
end

# PSF evaluation interface

# Fixed sigma model
@inline function evaluate_psf(psf::GaussianXYNB, i, j, θ::Params{4})
    x, y, N, bg = θ
    psf_x = integrated_gaussian_1d(Float32(i) - x, psf.σ)
    psf_y = integrated_gaussian_1d(Float32(j) - y, psf.σ)
    return bg + N * psf_x * psf_y
end

# Variable sigma model
@inline function evaluate_psf(::GaussianXYNBS, i, j, θ::Params{5})
    x, y, N, bg, σ = θ
    psf_x = integrated_gaussian_1d(Float32(i) - x, σ)
    psf_y = integrated_gaussian_1d(Float32(j) - y, σ)
    return bg + N * psf_x * psf_y
end

# Anisotropic model
@inline function evaluate_psf(::GaussianXYNBSXSY, i, j, θ::Params{6})
    x, y, N, bg, σx, σy = θ
    psf_x = integrated_gaussian_1d(Float32(i) - x, σx)
    psf_y = integrated_gaussian_1d(Float32(j) - y, σy)
    return bg + N * psf_x * psf_y
end

# Astigmatic model
@inline function evaluate_psf(psf::AstigmaticXYZNB, i, j, θ::Params{5})
    x, y, z, N, bg = θ
    
    # Width calculation based on z position
    σx = psf.σx₀ * sqrt(one(Float32) + ((z - psf.γ) / psf.d)^2 + psf.Ax * ((z - psf.γ) / psf.d)^3 + psf.Bx * ((z - psf.γ) / psf.d)^4)
    σy = psf.σy₀ * sqrt(one(Float32) + ((z + psf.γ) / psf.d)^2 + psf.Ay * ((z + psf.γ) / psf.d)^3 + psf.By * ((z + psf.γ) / psf.d)^4)
    
    psf_x = integrated_gaussian_1d(Float32(i) - x, σx)
    psf_y = integrated_gaussian_1d(Float32(j) - y, σy)
    return bg + N * psf_x * psf_y
end

# Compute pixel derivatives for Newton-Raphson
@inline function compute_pixel_derivatives(i, j, θ, psf_model::PSFModel{N}) where N
    # Use automatic differentiation or analytical derivatives
    # For now, we'll implement analytical derivatives for each model
    
    model_val = evaluate_psf(psf_model, i, j, θ)
    
    # Placeholder for derivatives - would implement analytical forms
    dudt = @SVector zeros(Float32, N)
    d2udt2 = @SMatrix zeros(Float32, N, N)
    
    return model_val, dudt, d2udt2
end

# Parameter initialization based on data
function initialize_parameters(roi::AbstractMatrix{T}, psf::GaussianXYNB) where T
    box_size = size(roi, 1)
    center = (box_size + 1) / 2
    
    # Simple COM initialization
    total = sum(roi)
    bg = minimum(roi)
    signal = roi .- bg
    
    x = sum((1:box_size) .* sum(signal, dims=2)[:]) / sum(signal)
    y = sum((1:box_size) .* sum(signal, dims=1)[:]) / sum(signal)
    N = sum(signal)
    
    return Params{4}(T(x), T(y), T(N), T(bg))
end

function initialize_parameters(roi::AbstractMatrix{T}, ::GaussianXYNBS) where T
    box_size = size(roi, 1)
    center = (box_size + 1) / 2
    
    total = sum(roi)
    bg = minimum(roi)
    signal = roi .- bg
    
    x = sum((1:box_size) .* sum(signal, dims=2)[:]) / sum(signal)
    y = sum((1:box_size) .* sum(signal, dims=1)[:]) / sum(signal)
    N = sum(signal)
    σ = T(1.3)  # Default starting value
    
    return Params{5}(T(x), T(y), T(N), T(bg), σ)
end

function initialize_parameters(roi::AbstractMatrix{T}, ::GaussianXYNBSXSY) where T
    box_size = size(roi, 1)
    center = (box_size + 1) / 2
    
    total = sum(roi)
    bg = minimum(roi)
    signal = roi .- bg
    
    x = sum((1:box_size) .* sum(signal, dims=2)[:]) / sum(signal)
    y = sum((1:box_size) .* sum(signal, dims=1)[:]) / sum(signal)
    N = sum(signal)
    σx = T(1.3)
    σy = T(1.3)
    
    return Params{6}(T(x), T(y), T(N), T(bg), σx, σy)
end

function initialize_parameters(roi::AbstractMatrix{T}, psf::AstigmaticXYZNB) where T
    box_size = size(roi, 1)
    center = (box_size + 1) / 2
    
    total = sum(roi)
    bg = minimum(roi)
    signal = roi .- bg
    
    x = sum((1:box_size) .* sum(signal, dims=2)[:]) / sum(signal)
    y = sum((1:box_size) .* sum(signal, dims=1)[:]) / sum(signal)
    N = sum(signal)
    z = T(0)  # Start at focal plane
    
    return Params{5}(T(x), T(y), z, T(N), T(bg))
end

# Export types and functions
export PSFModel, GaussianXYNB, GaussianXYNBS, GaussianXYNBSXSY, AstigmaticXYZNB
export Params, evaluate_psf, compute_pixel_derivatives, initialize_parameters
export integrated_gaussian_1d