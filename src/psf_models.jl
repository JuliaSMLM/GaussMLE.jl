"""
PSF Models with compile-time known parameter counts
"""

using StaticArrays
using SpecialFunctions: erf

"""
    PSFModel{NParams,T}

Abstract type for point spread function models.

The type parameter `NParams` specifies the number of fitting parameters at compile time,
enabling type-stable code generation. The type parameter `T` specifies the numeric type.
"""
abstract type PSFModel{NParams,T} end

"""
    GaussianXYNB{T} <: PSFModel{4,T}

2D Gaussian PSF with fixed width σ.

# Parameters (in order)
1. x: x-position (pixels)
2. y: y-position (pixels)
3. N: total photon count
4. bg: background per pixel

# Fields
- `σ::T`: Fixed Gaussian width (standard deviation in pixels)

# Example
```julia
psf = GaussianXYNB(1.3f0)  # Fixed σ = 1.3 pixels
fitter = GaussMLEFitter(psf_model = psf)
```
"""
struct GaussianXYNB{T} <: PSFModel{4,T}
    σ::T
end

"""
    GaussianXYNBS{T} <: PSFModel{5,T}

2D Gaussian PSF with variable width σ.

# Parameters (in order)
1. x: x-position (pixels)
2. y: y-position (pixels)
3. N: total photon count
4. bg: background per pixel
5. σ: Gaussian width (standard deviation in pixels)

# Example
```julia
psf = GaussianXYNBS()  # Variable sigma
fitter = GaussMLEFitter(psf_model = psf)
```
"""
struct GaussianXYNBS{T} <: PSFModel{5,T} end

# Default constructor for GaussianXYNBS
GaussianXYNBS() = GaussianXYNBS{Float32}()

"""
    GaussianXYNBSXSY{T} <: PSFModel{6,T}

2D Anisotropic Gaussian PSF with independent x and y widths.

# Parameters (in order)
1. x: x-position (pixels)
2. y: y-position (pixels)
3. N: total photon count
4. bg: background per pixel
5. σx: Gaussian width in x (standard deviation in pixels)
6. σy: Gaussian width in y (standard deviation in pixels)

# Example
```julia
psf = GaussianXYNBSXSY()
fitter = GaussMLEFitter(psf_model = psf)
```
"""
struct GaussianXYNBSXSY{T} <: PSFModel{6,T} end

# Default constructor for GaussianXYNBSXSY
GaussianXYNBSXSY() = GaussianXYNBSXSY{Float32}()

"""
    AstigmaticXYZNB{T} <: PSFModel{5,T}

3D astigmatic PSF for z-position estimation using engineered astigmatism.

The PSF width varies with z-position according to:
```
σx(z) = σx₀ * sqrt(1 + ((z-γ)/d)² + Ax*((z-γ)/d)³ + Bx*((z-γ)/d)⁴)
σy(z) = σy₀ * sqrt(1 + ((z+γ)/d)² + Ay*((z+γ)/d)³ + By*((z+γ)/d)⁴)
```

# Parameters (in order)
1. x: x-position (pixels)
2. y: y-position (pixels)
3. z: z-position (nm)
4. N: total photon count
5. bg: background per pixel

# Fields
- `σx₀, σy₀::T`: Nominal widths at z=0
- `Ax, Ay::T`: Cubic coefficients
- `Bx, By::T`: Quartic coefficients
- `γ::T`: Astigmatism offset (nm)
- `d::T`: Depth of focus (nm)

# Example
```julia
# Typical parameters from calibration
psf = AstigmaticXYZNB{Float32}(
    1.3f0, 1.3f0,    # σx₀, σy₀
    0.05f0, 0.05f0,  # Ax, Ay
    0.3f0, 0.3f0,    # Bx, By
    50.0f0,          # γ
    100.0f0          # d
)
fitter = GaussMLEFitter(psf_model = psf)
```

# See also
The astigmatic PSF model is described in Huang et al., Science 319, 810-813 (2008).
"""
struct AstigmaticXYZNB{T} <: PSFModel{5,T}
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
Base.length(::Type{<:PSFModel{N,T}}) where {N,T} = N
Base.length(::PSFModel{N,T}) where {N,T} = N

# Use the GaussLib implementation for consistency
using .GaussLib: integral_gaussian_1d

# PSF evaluation interface

# Fixed sigma model
@inline function evaluate_psf(psf::GaussianXYNB, i, j, θ::Params{4})
    x, y, N, bg = θ
    psf_x = integral_gaussian_1d(i, x, psf.σ)
    psf_y = integral_gaussian_1d(j, y, psf.σ)
    return bg + N * psf_x * psf_y
end

# Variable sigma model
@inline function evaluate_psf(::GaussianXYNBS, i, j, θ::Params{5})
    x, y, N, bg, σ = θ
    psf_x = integral_gaussian_1d(i, x, σ)
    psf_y = integral_gaussian_1d(j, y, σ)
    return bg + N * psf_x * psf_y
end

# Anisotropic model
@inline function evaluate_psf(::GaussianXYNBSXSY, i, j, θ::Params{6})
    x, y, N, bg, σx, σy = θ
    psf_x = integral_gaussian_1d(i, x, σx)
    psf_y = integral_gaussian_1d(j, y, σy)
    return bg + N * psf_x * psf_y
end

# Astigmatic model
@inline function evaluate_psf(psf::AstigmaticXYZNB, i, j, θ::Params{5})
    x, y, z, N, bg = θ
    
    # Width calculation based on z position using GaussLib
    # Ensure alpha is positive to avoid domain errors
    αx = max(0.1f0, GaussLib.compute_alpha((z - psf.γ), psf.Ax, psf.Bx, psf.d))
    αy = max(0.1f0, GaussLib.compute_alpha((z + psf.γ), psf.Ay, psf.By, psf.d))
    σx = psf.σx₀ * sqrt(αx)
    σy = psf.σy₀ * sqrt(αy)
    
    psf_x = integral_gaussian_1d(i, x, σx)
    psf_y = integral_gaussian_1d(j, y, σy)
    return bg + N * psf_x * psf_y
end

# Note: compute_pixel_derivatives is implemented in psf_derivatives.jl for each model type

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
    
    # Initialize z to zero - with proper gamma parameter, this is not a local minimum
    # The focal planes are separated by 2γ, so z=0 is a good starting point
    z = T(0)
    
    return Params{5}(T(x), T(y), z, T(N), T(bg))
end

# Export types and functions
export PSFModel, GaussianXYNB, GaussianXYNBS, GaussianXYNBSXSY, AstigmaticXYZNB
export Params, evaluate_psf, compute_pixel_derivatives, initialize_parameters
export integrated_gaussian_1d