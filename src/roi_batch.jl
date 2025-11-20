"""
GaussMLE-specific result types and conversion functions for SMLMData integration.

ROIBatch and SingleROI are imported from SMLMData.jl (the ecosystem standard).
This file contains only GaussMLE-specific types and conversions.
"""

# Extended results type with camera coordinates
struct LocalizationResult{T,P<:PSFModel}
    parameters::Matrix{T}       # ROI coordinates (as before)
    uncertainties::Matrix{T}     # Uncertainties
    log_likelihoods::Vector{T}  # Log-likelihood values

    # Camera coordinates (computed from ROI coords + corners)
    x_camera::Vector{T}         # X position in camera pixels
    y_camera::Vector{T}         # Y position in camera pixels

    # Context
    frame_indices::Vector{Int32}  # Frame index for each fit
    roi_corners::Matrix{Int32}  # For reference

    psf_model::P
    n_fits::Int
end

# Convert from ROI coordinates to camera coordinates
function roi_to_camera_coords(x_roi::T, y_roi::T, x_corner::Int32, y_corner::Int32) where T
    # ROI coordinates are 1-indexed (1 = first pixel in ROI)
    # Camera coordinates are 1-indexed (1 = first pixel in camera)
    # Corner is the camera pixel index of the top-left ROI pixel
    return x_corner + x_roi - one(T), y_corner + y_roi - one(T)
end

# Create LocalizationResult from fit results and ROIBatch
function create_localization_result(
    parameters::Matrix{T},
    uncertainties::Matrix{T},
    log_likelihoods::Vector{T},
    roi_batch::SMLMData.ROIBatch,
    psf_model::P
) where {T,P<:PSFModel}

    n_fits = size(parameters, 2)

    # Compute camera coordinates
    x_camera = Vector{T}(undef, n_fits)
    y_camera = Vector{T}(undef, n_fits)

    for i in 1:n_fits
        x_roi = parameters[1, i]
        y_roi = parameters[2, i]
        x_corner = roi_batch.corners[1, i]
        y_corner = roi_batch.corners[2, i]

        x_camera[i], y_camera[i] = roi_to_camera_coords(x_roi, y_roi, x_corner, y_corner)
    end

    LocalizationResult(
        parameters,
        uncertainties,
        log_likelihoods,
        x_camera,
        y_camera,
        roi_batch.frame_indices,
        roi_batch.corners,
        psf_model,
        n_fits
    )
end

# Iterator interface for LocalizationResult - enables `for fit in result`
function Base.iterate(r::LocalizationResult, state=1)
    if state > r.n_fits
        return nothing
    end
    fit = (
        parameters = r.parameters[:, state],
        uncertainties = r.uncertainties[:, state],
        log_likelihood = r.log_likelihoods[state],
        x_camera = r.x_camera[state],
        y_camera = r.y_camera[state],
        frame_index = r.frame_indices[state],
        roi_corner = r.roi_corners[:, state],
        index = state
    )
    return (fit, state + 1)
end

# Length and indexing support
Base.length(r::LocalizationResult) = r.n_fits
Base.size(r::LocalizationResult) = (r.n_fits,)

function Base.getindex(r::LocalizationResult, i::Int)
    if i < 1 || i > r.n_fits
        throw(BoundsError(r, i))
    end
    return (
        parameters = r.parameters[:, i],
        uncertainties = r.uncertainties[:, i],
        log_likelihood = r.log_likelihoods[i],
        x_camera = r.x_camera[i],
        y_camera = r.y_camera[i],
        frame_index = r.frame_indices[i],
        roi_corner = r.roi_corners[:, i],
        index = i
    )
end

# ===================================================================
# Emitter Conversion Functions - Dispatch on PSF Model Type
# ===================================================================

# Generic dispatcher - routes to appropriate emitter constructor based on PSF model
to_emitter(result::LocalizationResult, idx::Int, camera::SMLMData.AbstractCamera; kwargs...) =
    to_emitter(result.psf_model, result, idx, camera; kwargs...)

# ---- GaussianXYNB: Standard Emitter2DFit ----
function to_emitter(
    ::GaussianXYNB,
    result::LocalizationResult{T},
    idx::Int,
    camera::SMLMData.AbstractCamera;
    dataset::Int = 1,
    track_id::Int = 0,
    id::Int = idx
) where T
    pixel_size_x = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
    pixel_size_y = camera.pixel_edges_y[2] - camera.pixel_edges_y[1]

    # Convert to microns
    x_microns = (result.x_camera[idx] - 1) * pixel_size_x
    y_microns = (result.y_camera[idx] - 1) * pixel_size_y

    # Parameter order: [x, y, photons, bg]
    photons = result.parameters[3, idx]
    bg = result.parameters[4, idx]

    # Uncertainties (convert spatial to microns)
    σ_x = result.uncertainties[1, idx] * pixel_size_x
    σ_y = result.uncertainties[2, idx] * pixel_size_y
    σ_photons = result.uncertainties[3, idx]
    σ_bg = result.uncertainties[4, idx]

    SMLMData.Emitter2DFit{T}(
        T(x_microns), T(y_microns),
        photons, bg,
        T(σ_x), T(σ_y),
        σ_photons, σ_bg,
        Int(result.frame_indices[idx]),
        dataset, track_id, id
    )
end

# ---- GaussianXYNBS: Emitter2DFitSigma (with fitted σ) ----
function to_emitter(
    ::GaussianXYNBS,
    result::LocalizationResult{T},
    idx::Int,
    camera::SMLMData.AbstractCamera;
    dataset::Int = 1,
    track_id::Int = 0,
    id::Int = idx
) where T
    pixel_size_x = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
    pixel_size_y = camera.pixel_edges_y[2] - camera.pixel_edges_y[1]

    # Convert to microns
    x_microns = (result.x_camera[idx] - 1) * pixel_size_x
    y_microns = (result.y_camera[idx] - 1) * pixel_size_y

    # Parameter order: [x, y, photons, bg, σ]
    photons = result.parameters[3, idx]
    bg = result.parameters[4, idx]
    σ_pixels = result.parameters[5, idx]
    σ_microns = σ_pixels * pixel_size_x  # Convert to microns

    # Uncertainties
    σ_x = result.uncertainties[1, idx] * pixel_size_x
    σ_y = result.uncertainties[2, idx] * pixel_size_y
    σ_photons = result.uncertainties[3, idx]
    σ_bg = result.uncertainties[4, idx]
    σ_σ = result.uncertainties[5, idx] * pixel_size_x  # σ uncertainty in microns

    Emitter2DFitSigma{T}(
        T(x_microns), T(y_microns),
        photons, bg,
        T(σ_microns),
        T(σ_x), T(σ_y),
        σ_photons, σ_bg,
        T(σ_σ),
        Int(result.frame_indices[idx]),
        dataset, track_id, id
    )
end

# ---- GaussianXYNBSXSY: Emitter2DFitSigmaXY (with fitted σx, σy) ----
function to_emitter(
    ::GaussianXYNBSXSY,
    result::LocalizationResult{T},
    idx::Int,
    camera::SMLMData.AbstractCamera;
    dataset::Int = 1,
    track_id::Int = 0,
    id::Int = idx
) where T
    pixel_size_x = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
    pixel_size_y = camera.pixel_edges_y[2] - camera.pixel_edges_y[1]

    # Convert to microns
    x_microns = (result.x_camera[idx] - 1) * pixel_size_x
    y_microns = (result.y_camera[idx] - 1) * pixel_size_y

    # Parameter order: [x, y, photons, bg, σx, σy]
    photons = result.parameters[3, idx]
    bg = result.parameters[4, idx]
    σx_pixels = result.parameters[5, idx]
    σy_pixels = result.parameters[6, idx]
    σx_microns = σx_pixels * pixel_size_x
    σy_microns = σy_pixels * pixel_size_y

    # Uncertainties
    σ_x = result.uncertainties[1, idx] * pixel_size_x
    σ_y = result.uncertainties[2, idx] * pixel_size_y
    σ_photons = result.uncertainties[3, idx]
    σ_bg = result.uncertainties[4, idx]
    σ_σx = result.uncertainties[5, idx] * pixel_size_x
    σ_σy = result.uncertainties[6, idx] * pixel_size_y

    Emitter2DFitSigmaXY{T}(
        T(x_microns), T(y_microns),
        photons, bg,
        T(σx_microns), T(σy_microns),
        T(σ_x), T(σ_y),
        σ_photons, σ_bg,
        T(σ_σx), T(σ_σy),
        Int(result.frame_indices[idx]),
        dataset, track_id, id
    )
end

# ---- AstigmaticXYZNB: Emitter3DFit ----
function to_emitter(
    ::AstigmaticXYZNB,
    result::LocalizationResult{T},
    idx::Int,
    camera::SMLMData.AbstractCamera;
    dataset::Int = 1,
    track_id::Int = 0,
    id::Int = idx
) where T
    pixel_size_x = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
    pixel_size_y = camera.pixel_edges_y[2] - camera.pixel_edges_y[1]

    # Convert to microns
    x_microns = (result.x_camera[idx] - 1) * pixel_size_x
    y_microns = (result.y_camera[idx] - 1) * pixel_size_y
    z_pixels = result.parameters[3, idx]
    z_microns = z_pixels * pixel_size_x

    # Parameter order: [x, y, z, photons, bg]
    photons = result.parameters[4, idx]
    bg = result.parameters[5, idx]

    # Uncertainties
    σ_x = result.uncertainties[1, idx] * pixel_size_x
    σ_y = result.uncertainties[2, idx] * pixel_size_y
    σ_z = result.uncertainties[3, idx] * pixel_size_x
    σ_photons = result.uncertainties[4, idx]
    σ_bg = result.uncertainties[5, idx]

    SMLMData.Emitter3DFit{T}(
        T(x_microns), T(y_microns), T(z_microns),
        photons, bg,
        T(σ_x), T(σ_y), T(σ_z),
        σ_photons, σ_bg,
        Int(result.frame_indices[idx]),
        dataset, track_id, id
    )
end

# Legacy function names for backward compatibility
to_emitter2dfit(result::LocalizationResult, idx::Int, camera::SMLMData.AbstractCamera; kwargs...) =
    to_emitter(GaussianXYNB(1.0f0), result, idx, camera; kwargs...)

to_emitter3dfit(result::LocalizationResult{T, <:AstigmaticXYZNB}, idx::Int, camera::SMLMData.AbstractCamera; kwargs...) where T =
    to_emitter(result.psf_model, result, idx, camera; kwargs...)

# ===================================================================
# Unified to_smld - Uses dispatch automatically
# ===================================================================

"""
    to_smld(result::LocalizationResult, roi_batch::ROIBatch; kwargs...)

Convert LocalizationResult to SMLMData.BasicSMLD with appropriate emitter types.

Dispatches to correct emitter constructor based on PSF model:
- GaussianXYNB → Emitter2DFit
- GaussianXYNBS → Emitter2DFitSigma (with σ)
- GaussianXYNBSXSY → Emitter2DFitSigmaXY (with σx, σy)
- AstigmaticXYZNB → Emitter3DFit (with z)
"""
function to_smld(
    result::LocalizationResult,
    roi_batch::SMLMData.ROIBatch;
    dataset::Int = 1,
    metadata::Dict{String,Any} = Dict{String,Any}()
)
    # Create emitters - dispatch handles emitter type selection
    emitters = [
        to_emitter(result, i, roi_batch.camera; dataset=dataset, id=i)
        for i in 1:result.n_fits
    ]

    # Determine frame range
    n_frames = maximum(result.frame_indices)

    # Add fitting metadata
    metadata["psf_model"] = string(typeof(result.psf_model))
    metadata["n_fits"] = result.n_fits

    # Create BasicSMLD with mixed emitter types (all <: AbstractEmitter)
    SMLMData.BasicSMLD(
        emitters,
        roi_batch.camera,
        Int(n_frames),
        1,  # n_datasets
        metadata
    )
end
