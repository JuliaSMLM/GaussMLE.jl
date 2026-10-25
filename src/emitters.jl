"""
Custom emitter types for GaussMLE that extend SMLMData.AbstractEmitter

These types store fitted PSF parameters (σ, σx, σy) and goodness-of-fit metrics
in addition to standard localization parameters. All types are mutable structs
subtyping AbstractEmitter for full ecosystem compatibility.

All emitter types include:
- Position and photometry (standard)
- CRLB uncertainties (standard)
- Goodness-of-fit p-value (from χ² test)
- PSF-specific parameters (σ, σx, σy, z)

IdealCamera photometry uses photons and photons/pixel; sCMOS photometry uses electrons
and electrons/pixel without quantum-efficiency correction.
"""

"""
    Emitter2DFitGaussMLE{T} <: AbstractEmitter

2D emitter for fixed-σ Gaussian PSF (GaussianXYNB) with goodness-of-fit.

Equivalent localization fields to SMLMData.Emitter2DFit, with an additional `pvalue` field.

# Fields
## Spatial (microns)
- `x::T`, `y::T`: Position in microns

## Photometry
- `photons::T`: Total signal (photons for IdealCamera, electrons for sCMOS)
- `bg::T`: Background (photons/pixel for IdealCamera, electrons/pixel for sCMOS)

## Uncertainties (CRLB, microns for spatial)
- `σ_x::T`, `σ_y::T`: Position uncertainties (microns)
- `σ_xy::T`: Position covariance (microns²) - off-diagonal of Fisher matrix inverse
- `σ_photons::T`: Signal uncertainty (photons for IdealCamera, electrons for sCMOS)
- `σ_bg::T`: Background uncertainty (photons/pixel for IdealCamera,
  electrons/pixel for sCMOS)

## Goodness-of-Fit
- `pvalue::T`: P-value from χ² test (df = n_pixels - n_params)

## Metadata
- `frame::Int`: Frame number
- `dataset::Int`: Dataset ID
- `track_id::Int`: Trajectory ID
- `id::Int`: Unique emitter ID
"""
mutable struct Emitter2DFitGaussMLE{T} <: SMLMData.AbstractEmitter
    # Spatial (microns)
    x::T
    y::T

    # Photometry
    photons::T
    bg::T

    # Uncertainties
    σ_x::T
    σ_y::T
    σ_xy::T  # Position covariance (microns²)
    σ_photons::T
    σ_bg::T

    # Goodness-of-fit
    pvalue::T

    # Metadata
    frame::Int
    dataset::Int
    track_id::Int
    id::Int
end

"""
    Emitter2DFitSigma{T} <: AbstractEmitter

2D emitter with fitted isotropic PSF width σ (for GaussianXYNBS).

# Fields
## Spatial (microns)
- `x::T`, `y::T`: Position in microns

## Photometry
- `photons::T`: Total signal (photons for IdealCamera, electrons for sCMOS)
- `bg::T`: Background (photons/pixel for IdealCamera, electrons/pixel for sCMOS)

## PSF Parameter
- `σ::T`: Fitted PSF width (microns)

## Uncertainties (CRLB)
- `σ_x::T`, `σ_y::T`: Position uncertainties (microns)
- `σ_xy::T`: Position covariance (microns²) - off-diagonal of Fisher matrix inverse
- `σ_photons::T`: Signal uncertainty (photons for IdealCamera, electrons for sCMOS)
- `σ_bg::T`: Background uncertainty (photons/pixel for IdealCamera,
  electrons/pixel for sCMOS)
- `σ_σ::T`: PSF width uncertainty (microns)

## Goodness-of-Fit
- `pvalue::T`: P-value from χ² test (df = n_pixels - n_params)

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
    σ_xy::T  # Position covariance (microns²)
    σ_photons::T
    σ_bg::T
    σ_σ::T  # Uncertainty in fitted sigma

    # Goodness-of-fit
    pvalue::T

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
- `photons::T`: Total signal (photons for IdealCamera, electrons for sCMOS)
- `bg::T`: Background (photons/pixel for IdealCamera, electrons/pixel for sCMOS)

## PSF Parameters
- `σx::T`, `σy::T`: Fitted PSF widths in x and y (microns)

## Uncertainties (CRLB)
- `σ_x::T`, `σ_y::T`: Position uncertainties (microns)
- `σ_xy::T`: Position covariance (microns²) - off-diagonal of Fisher matrix inverse
- `σ_photons::T`: Signal uncertainty (photons for IdealCamera, electrons for sCMOS)
- `σ_bg::T`: Background uncertainty (photons/pixel for IdealCamera,
  electrons/pixel for sCMOS)
- `σ_σx::T`, `σ_σy::T`: PSF width uncertainties (microns)

## Goodness-of-Fit
- `pvalue::T`: P-value from χ² test (df = n_pixels - n_params)

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
    σ_xy::T  # Position covariance (microns²)
    σ_photons::T
    σ_bg::T
    σ_σx::T  # Uncertainty in fitted sigma_x
    σ_σy::T  # Uncertainty in fitted sigma_y

    # Goodness-of-fit
    pvalue::T

    # Metadata
    frame::Int
    dataset::Int
    track_id::Int
    id::Int
end

"""
    Emitter3DFitGaussMLE{T} <: AbstractEmitter

3D emitter for astigmatic PSF (AstigmaticXYZNB) with goodness-of-fit.

Equivalent localization fields to SMLMData.Emitter3DFit, with an additional `pvalue` field.

# Fields
## Spatial (microns)
- `x::T`, `y::T`, `z::T`: Position in microns

## Photometry
- `photons::T`: Total signal (photons for IdealCamera, electrons for sCMOS)
- `bg::T`: Background (photons/pixel for IdealCamera, electrons/pixel for sCMOS)

## Uncertainties (CRLB, microns for spatial)
- `σ_x::T`, `σ_y::T`, `σ_z::T`: Position uncertainties (microns)
- `σ_xy::T`, `σ_xz::T`, `σ_yz::T`: Position covariances (microns²) - off-diagonals of
  Fisher matrix inverse
- `σ_photons::T`: Signal uncertainty (photons for IdealCamera, electrons for sCMOS)
- `σ_bg::T`: Background uncertainty (photons/pixel for IdealCamera,
  electrons/pixel for sCMOS)

## Goodness-of-Fit
- `pvalue::T`: P-value from χ² test (df = n_pixels - n_params)

## Metadata
- `frame::Int`: Frame number
- `dataset::Int`: Dataset ID
- `track_id::Int`: Trajectory ID
- `id::Int`: Unique emitter ID
"""
mutable struct Emitter3DFitGaussMLE{T} <: SMLMData.AbstractEmitter
    # Spatial (microns)
    x::T
    y::T
    z::T

    # Photometry
    photons::T
    bg::T

    # Uncertainties
    σ_x::T
    σ_y::T
    σ_z::T
    σ_xy::T   # Position covariance x-y (microns²)
    σ_xz::T   # Position covariance x-z (microns²)
    σ_yz::T   # Position covariance y-z (microns²)
    σ_photons::T
    σ_bg::T

    # Goodness-of-fit
    pvalue::T

    # Metadata
    frame::Int
    dataset::Int
    track_id::Int
    id::Int
end
