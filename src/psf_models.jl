"""
PSF Models with compile-time known parameter counts
"""

"""
    PSFModel{NParams,T}

Abstract type for point spread function models.

The type parameter `NParams` specifies the number of fitting parameters at compile time,
enabling type-stable code generation. The type parameter `T` specifies the numeric type.

# Custom models
Since 0.4.3 the kernel receives every model through `Adapt.adapt_structure`, on the CPU
(POCL) as on a GPU, so a model defined outside this package must be `isbits` after Adapt:
numbers, `SVector`s, and arrays that `Adapt.adapt_structure` converts to device arrays, with
any host-only field (a callable, a heap array) dropped there, as `SplinePSFModel` drops its
point PSF. `simple_initialize` must return an `SVector`, not an `MVector`. A model that is
not `isbits` after Adapt fails with a `GPUCompiler.KernelError` ("passing non-bitstype
argument").
"""
abstract type PSFModel{NParams, T} end

"""
    GaussianXYNB{T} <: PSFModel{4,T}

2D Gaussian PSF with fixed width σ.

# Parameters (in order)
1. x: x-position (fitted in ROI pixels, output in microns)
2. y: y-position (fitted in ROI pixels, output in microns)
3. N: total signal (photons for IdealCamera, electrons for sCMOS)
4. bg: background per pixel

# Fields
- `σ::T`: Fixed Gaussian width (standard deviation in **microns**)

# Example
```jldoctest
julia> psf = GaussianXYNB(0.13f0)  # σ = 130 nm (typical ~500nm emission, 100nm pixels)
GaussianXYNB(σ=0.13)

julia> fitter = GaussMLEConfig(psf_model = psf);
```

# Note
PSF width is specified in physical units (microns) for camera-independence.
Internally converted to pixels based on camera pixel size during fitting.
"""
struct GaussianXYNB{T} <: PSFModel{4, T}
    σ::T  # Microns
end

"""
    GaussianXYNBS{T} <: PSFModel{5,T}

2D Gaussian PSF with variable width σ (fitted parameter).

# Parameters (in order)
1. x: x-position (fitted in ROI pixels, output in microns)
2. y: y-position (fitted in ROI pixels, output in microns)
3. N: total signal (photons for IdealCamera, electrons for sCMOS)
4. bg: background per pixel
5. σ: Gaussian width (fitted in pixels, output in microns)

# Example
```julia
psf = GaussianXYNBS()  # Variable sigma (no fixed value)
fitter = GaussMLEConfig(psf_model = psf)
```

# Note
The fitted σ parameter is stored in microns in Emitter2DFitSigma output.
"""
struct GaussianXYNBS{T} <: PSFModel{5, T} end

# Default constructor for GaussianXYNBS
GaussianXYNBS() = GaussianXYNBS{Float32}()

"""
    GaussianXYNBSXSY{T} <: PSFModel{6,T}

2D Anisotropic Gaussian PSF with independent x and y widths (both fitted parameters).

# Parameters (in order)
1. x: x-position (fitted in ROI pixels, output in microns)
2. y: y-position (fitted in ROI pixels, output in microns)
3. N: total signal (photons for IdealCamera, electrons for sCMOS)
4. bg: background per pixel
5. σx: Gaussian width in x (fitted in pixels, output in microns)
6. σy: Gaussian width in y (fitted in pixels, output in microns)

# Example
```julia
psf = GaussianXYNBSXSY()  # Variable σx, σy (no fixed values)
fitter = GaussMLEConfig(psf_model = psf)
```

# Note
The fitted σx and σy parameters are stored in microns in Emitter2DFitSigmaXY output.
"""
struct GaussianXYNBSXSY{T} <: PSFModel{6, T} end

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
1. x: x-position (fitted in ROI pixels, output in microns)
2. y: y-position (fitted in ROI pixels, output in microns)
3. z: z-position (microns throughout fitting and output)
4. N: total signal (photons for IdealCamera, electrons for sCMOS)
5. bg: background per pixel

# Fields (spatial calibration parameters in **microns**)
- `σx₀, σy₀::T`: In-focus PSF widths (microns)
- `Ax, Ay::T`: Cubic coefficients (dimensionless)
- `Bx, By::T`: Quartic coefficients (dimensionless)
- `γ::T`: Astigmatism offset (microns)
- `d::T`: Depth scaling parameter (microns)

# Example
```julia
# Typical parameters from calibration (all spatial params in microns)
psf = AstigmaticXYZNB{Float32}(
    0.13f0, 0.13f0,  # σx₀, σy₀ (130 nm)
    0.05f0, 0.05f0,  # Ax, Ay (dimensionless)
    0.3f0, 0.3f0,    # Bx, By (dimensionless)
    0.05f0,          # γ (50 nm astigmatism offset)
    0.4f0            # d (400 nm depth scale)
)
fitter = GaussMLEConfig(psf_model = psf)
```

# Note
Spatial calibration parameters are specified in microns for camera-independence.
Internally, x, y and lateral widths use pixels; z, γ and d remain in microns.
Ax, Ay, Bx and By are dimensionless.

# See also
The astigmatic PSF model is described in Huang et al., Science 319, 810-813 (2008).
"""
struct AstigmaticXYZNB{T} <: PSFModel{5, T}
    σx₀::T  # Microns
    σy₀::T  # Microns
    Ax::T   # Dimensionless
    Ay::T   # Dimensionless
    Bx::T   # Dimensionless
    By::T   # Dimensionless
    γ::T    # Microns
    d::T    # Microns

    function AstigmaticXYZNB{T}(σx₀, σy₀, Ax, Ay, Bx, By, γ, d) where {T}
        return new{T}(T(σx₀), T(σy₀), T(Ax), T(Ay), T(Bx), T(By), T(γ), T(d))
    end
end

# Parameter type aliases
const Params{N} = SVector{N, Float32}

# Get parameter count for a model
Base.length(::Type{<:PSFModel{N, T}}) where {N, T} = N
Base.length(::PSFModel{N, T}) where {N, T} = N

# Gaussian models contain only scalars. Tabulated models specialize these host helpers.
_prepare_psf_model(backend, model::PSFModel) = model
_psf_table_bytes(::PSFModel) = 0

# Whether parameter 3 is axial position (x, y, z, N, bg ordering).
_has_axial_position(::Type{<:PSFModel}) = false
_has_axial_position(::Type{<:AstigmaticXYZNB}) = true

# PSF evaluation interface

# Fixed sigma model
@inline function evaluate_psf(psf::GaussianXYNB, i, j, θ::Params{4})
    x, y, N, bg = θ
    psf_x = integral_gaussian_1d(j, x, psf.σ)
    psf_y = integral_gaussian_1d(i, y, psf.σ)
    return bg + N * psf_x * psf_y
end

# Variable sigma model
@inline function evaluate_psf(::GaussianXYNBS, i, j, θ::Params{5})
    x, y, N, bg, σ = θ
    psf_x = integral_gaussian_1d(j, x, σ)
    psf_y = integral_gaussian_1d(i, y, σ)
    return bg + N * psf_x * psf_y
end

# Anisotropic model
@inline function evaluate_psf(::GaussianXYNBSXSY, i, j, θ::Params{6})
    x, y, N, bg, σx, σy = θ
    psf_x = integral_gaussian_1d(j, x, σx)
    psf_y = integral_gaussian_1d(i, y, σy)
    return bg + N * psf_x * psf_y
end

# Astigmatic model
@inline function evaluate_psf(psf::AstigmaticXYZNB, i, j, θ::Params{5})
    x, y, z, N, bg = θ

    # Width calculation based on z position using GaussLib
    αx = GaussLib.compute_alpha((z - psf.γ), psf.Ax, psf.Bx, psf.d)
    αy = GaussLib.compute_alpha((z + psf.γ), psf.Ay, psf.By, psf.d)
    σx = psf.σx₀ * sqrt(αx)
    σy = psf.σy₀ * sqrt(αy)

    psf_x = integral_gaussian_1d(j, x, σx)
    psf_y = integral_gaussian_1d(i, y, σy)
    return bg + N * psf_x * psf_y
end

# GaussMLE does not call `initialize_parameters`; the generic function stays with no methods
# so packages that extend it still load. It goes in 0.5.
function initialize_parameters end

# Note: compute_pixel_derivatives is implemented in psf_derivatives.jl for each model type

# Pretty printing for PSF models
Base.show(io::IO, psf::GaussianXYNB) = print(io, "GaussianXYNB(σ=$(psf.σ))")
# No fields, just default
Base.show(io::IO, psf::GaussianXYNBS) = print(io, "GaussianXYNBS()")
Base.show(io::IO, psf::GaussianXYNBSXSY) = print(io, "GaussianXYNBSXSY()")  # No fields
function Base.show(io::IO, psf::AstigmaticXYZNB)
    print(io, "AstigmaticXYZNB(")
    print(io, "σx₀=", psf.σx₀, ", σy₀=", psf.σy₀, ", ")
    print(io, "γ=", psf.γ, ", d=", psf.d)
    return print(io, ")")
end
# ===================================================================
# Unit Conversion: Physical (microns) → Pixels
# ===================================================================

"""
    to_pixel_units(psf::PSFModel, pixel_size::Real)

Convert PSF lateral widths from microns to pixels for kernel computation.

User-facing PSF models store spatial calibration parameters in microns.
Only lateral widths are converted; astigmatic z, γ and d remain in microns,
and Ax, Ay, Bx and By are dimensionless.

# Arguments
- `psf`: PSF model with spatial calibration parameters in microns
- `pixel_size`: Camera pixel size in microns

# Returns
PSF model with lateral widths converted to pixels

# Example
```jldoctest
julia> psf_microns = GaussianXYNB(0.13f0);  # σ = 130 nm

julia> GaussMLE.to_pixel_units(psf_microns, 0.1f0)  # 100 nm pixels
GaussianXYNB(σ=1.3)
```
"""
function to_pixel_units(psf::GaussianXYNB{T}, pixel_size::Real) where {T}
    return GaussianXYNB{T}(psf.σ / T(pixel_size))
end

# GaussianXYNBS and GaussianXYNBSXSY have no fixed parameters - no conversion needed
to_pixel_units(psf::GaussianXYNBS{T}, pixel_size::Real) where {T} = psf
to_pixel_units(psf::GaussianXYNBSXSY{T}, pixel_size::Real) where {T} = psf

function to_pixel_units(psf::AstigmaticXYZNB{T}, pixel_size::Real) where {T}
    px = T(pixel_size)
    return AstigmaticXYZNB{T}(
        psf.σx₀ / px,  # Lateral width: microns → pixels
        psf.σy₀ / px,  # Lateral width: microns → pixels
        psf.Ax,        # Dimensionless
        psf.Ay,        # Dimensionless
        psf.Bx,        # Dimensionless
        psf.By,        # Dimensionless
        psf.γ,         # Keep in microns (z is physical, not pixels)
        psf.d          # Keep in microns (z is physical, not pixels)
    )
end
