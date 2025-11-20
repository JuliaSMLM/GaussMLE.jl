"""
Custom emitter types for GaussMLE that extend SMLMData.AbstractEmitter

These types store fitted PSF parameters (σ, σx, σy) in addition to the standard
localization parameters. All types are mutable structs subtying AbstractEmitter
for full ecosystem compatibility.
"""

using SMLMData

"""
    Emitter2DFitSigma{T} <: AbstractEmitter

2D emitter with fitted isotropic PSF width σ (for GaussianXYNBS).

# Fields
## Spatial (microns)
- `x::T`, `y::T`: Position in microns

## Photometry
- `photons::T`: Total photon count
- `bg::T`: Background level

## PSF Parameter
- `σ::T`: Fitted PSF width (microns)

## Uncertainties (CRLB)
- `σ_x::T`, `σ_y::T`: Position uncertainties (microns)
- `σ_photons::T`, `σ_bg::T`: Photometry uncertainties
- `σ_σ::T`: PSF width uncertainty (microns)

## Metadata
- `frame::Int`: Frame number
- `dataset::Int`: Dataset ID
- `track_id::Int`: Trajectory ID
- `id::Int`: Unique emitter ID
"""
mutable struct Emitter2DFitSigma{T} <: SMLMData.AbstractEmitter
    # Spatial (microns)
    x::T
    y::T

    # Photometry
    photons::T
    bg::T

    # PSF parameter (microns)
    σ::T

    # Uncertainties (microns for spatial, same units for photometry)
    σ_x::T
    σ_y::T
    σ_photons::T
    σ_bg::T
    σ_σ::T  # Uncertainty in fitted sigma

    # Metadata
    frame::Int
    dataset::Int
    track_id::Int
    id::Int
end

"""
    Emitter2DFitSigmaXY{T} <: AbstractEmitter

2D emitter with fitted anisotropic PSF widths σx, σy (for GaussianXYNBSXSY).

# Fields
## Spatial (microns)
- `x::T`, `y::T`: Position in microns

## Photometry
- `photons::T`: Total photon count
- `bg::T`: Background level

## PSF Parameters
- `σx::T`, `σy::T`: Fitted PSF widths in x and y (microns)

## Uncertainties (CRLB)
- `σ_x::T`, `σ_y::T`: Position uncertainties (microns)
- `σ_photons::T`, `σ_bg::T`: Photometry uncertainties
- `σ_σx::T`, `σ_σy::T`: PSF width uncertainties (microns)

## Metadata
- `frame::Int`: Frame number
- `dataset::Int`: Dataset ID
- `track_id::Int`: Trajectory ID
- `id::Int`: Unique emitter ID
"""
mutable struct Emitter2DFitSigmaXY{T} <: SMLMData.AbstractEmitter
    # Spatial (microns)
    x::T
    y::T

    # Photometry
    photons::T
    bg::T

    # PSF parameters (microns)
    σx::T
    σy::T

    # Uncertainties (microns for spatial, same units for photometry)
    σ_x::T
    σ_y::T
    σ_photons::T
    σ_bg::T
    σ_σx::T  # Uncertainty in fitted sigma_x
    σ_σy::T  # Uncertainty in fitted sigma_y

    # Metadata
    frame::Int
    dataset::Int
    track_id::Int
    id::Int
end
