"""
ROI batch data structure for efficient GPU processing with location tracking
Integrates with SMLMData.jl types for ecosystem compatibility
"""

using StaticArrays
using SMLMData
using KernelAbstractions

# ROI batch for efficient SoA GPU processing
struct ROIBatch{T,N,A<:AbstractArray{T,N},C<:SMLMData.AbstractCamera}
    data::A                        # (roi_size, roi_size, n_rois) 
    corners::Matrix{Int32}         # 2×n_rois for [x;y] pixel corners (1-indexed)
    frame_indices::Vector{Int32}   # Frame number for each ROI
    camera::C                      # SMLMData camera (IdealCamera or SCMOSCamera)
    roi_size::Int                  # Size of each ROI (assumed square)
    
    function ROIBatch(data::A, corners::Matrix{Int32}, frame_indices::Vector{Int32}, 
                     camera::C) where {T,A<:AbstractArray{T,3},C<:SMLMData.AbstractCamera}
        n_rois = size(data, 3)
        roi_size = size(data, 1)
        @assert size(data, 1) == size(data, 2) "ROIs must be square"
        @assert size(corners) == (2, n_rois) "Corners must be 2×n_rois"
        @assert length(frame_indices) == n_rois "Must have one frame index per ROI"
        new{T,3,A,C}(data, corners, frame_indices, camera, roi_size)
    end
end

# Convenience constructor from vectors of corners
function ROIBatch(data::AbstractArray{T,3}, x_corners::Vector, y_corners::Vector, 
                  frame_indices::Vector, camera::C) where {T,C<:SMLMData.AbstractCamera}
    corners = Matrix{Int32}(undef, 2, length(x_corners))
    corners[1, :] = x_corners
    corners[2, :] = y_corners
    ROIBatch(data, corners, Int32.(frame_indices), camera)
end

# Single ROI for convenient access
struct SingleROI{T}
    data::Matrix{T}
    corner::SVector{2,Int32}  # (x, y) pixel corner (1-indexed)
    frame_idx::Int32
end

# Convert single ROIs to batch - requires camera
function ROIBatch(rois::Vector{SingleROI{T}}, camera::C) where {T,C<:SMLMData.AbstractCamera}
    if isempty(rois)
        # Need a dummy camera for empty batch - use provided camera
        return ROIBatch(zeros(T, 0, 0, 0), Matrix{Int32}(undef, 2, 0), Int32[], camera)
    end
    
    roi_size = size(first(rois).data, 1)
    n_rois = length(rois)
    
    # Pre-allocate arrays
    data = zeros(T, roi_size, roi_size, n_rois)
    corners = Matrix{Int32}(undef, 2, n_rois)
    frame_indices = Vector{Int32}(undef, n_rois)
    
    for (i, roi) in enumerate(rois)
        data[:, :, i] = roi.data
        corners[:, i] = roi.corner
        frame_indices[i] = roi.frame_idx
    end
    
    ROIBatch(data, corners, frame_indices, camera)
end

# Indexing to get individual ROIs
Base.getindex(batch::ROIBatch, i::Int) = SingleROI(
    batch.data[:, :, i],
    SVector{2,Int32}(batch.corners[:, i]),
    batch.frame_indices[i]
)

Base.length(batch::ROIBatch) = size(batch.data, 3)
Base.size(batch::ROIBatch) = (length(batch),)

# Iteration support
Base.iterate(batch::ROIBatch, state=1) = state > length(batch) ? nothing : (batch[state], state + 1)

# Device transfer support for KernelAbstractions
import Adapt
function Adapt.adapt_structure(to, batch::ROIBatch)
    ROIBatch(
        Adapt.adapt(to, batch.data),
        Adapt.adapt(to, batch.corners),
        Adapt.adapt(to, batch.frame_indices),
        batch.camera  # Camera stays on host (contains variance map)
    )
end

# Extended results type with camera coordinates
struct LocalizationResult{T,P<:PSFModel}
    parameters::Matrix{T}       # ROI coordinates (as before)
    uncertainties::Matrix{T}     # Uncertainties
    log_likelihoods::Vector{T}  # Log-likelihood values
    
    # Camera coordinates (computed from ROI coords + corners)
    x_camera::Vector{T}         # X position in camera pixels
    y_camera::Vector{T}         # Y position in camera pixels
    
    # Context
    frame_indices::Vector{Int32}
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
    roi_batch::ROIBatch,
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

# Conversion to SMLMData Emitter2DFit (coordinates in microns)
function to_emitter2dfit(
    result::LocalizationResult,
    idx::Int,
    camera::SMLMData.AbstractCamera;
    dataset::Int = 1,
    track_id::Int = 0,
    id::Int = idx
)
    # Get pixel size from camera (assumes uniform pixels)
    pixel_size_x = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
    pixel_size_y = camera.pixel_edges_y[2] - camera.pixel_edges_y[1]
    
    # Convert camera pixel coordinates to physical microns
    # Camera pixels are 1-indexed, physical space has origin at (0,0)
    x_microns = (result.x_camera[idx] - 1) * pixel_size_x
    y_microns = (result.y_camera[idx] - 1) * pixel_size_y
    
    # Get fitting parameters
    photons = result.parameters[3, idx]
    bg = result.parameters[4, idx]
    
    # Convert uncertainties from pixels to microns
    σ_x = result.uncertainties[1, idx] * pixel_size_x
    σ_y = result.uncertainties[2, idx] * pixel_size_y
    σ_photons = result.uncertainties[3, idx]
    σ_bg = result.uncertainties[4, idx]
    
    # Create Emitter2DFit - need to match the Float type throughout
    T = eltype(result.parameters)
    SMLMData.Emitter2DFit{T}(
        T(x_microns), T(y_microns),
        photons, bg,
        T(σ_x), T(σ_y),
        σ_photons, σ_bg,
        Int(result.frame_indices[idx]),
        dataset,
        track_id,
        id
    )
end

# Batch conversion to SMLMData format
function to_smld(
    result::LocalizationResult,
    roi_batch::ROIBatch;
    dataset::Int = 1,
    metadata::Dict{String,Any} = Dict{String,Any}()
)
    # Create emitters using camera from ROIBatch
    emitters = [
        to_emitter2dfit(result, i, roi_batch.camera; dataset=dataset, id=i)
        for i in 1:result.n_fits
    ]
    
    # Determine frame range
    n_frames = maximum(result.frame_indices)
    
    # Add fitting metadata
    metadata["psf_model"] = string(typeof(result.psf_model))
    metadata["n_fits"] = result.n_fits
    
    # Create BasicSMLD
    SMLMData.BasicSMLD(
        emitters,
        roi_batch.camera,
        Int(n_frames),  # Convert to Int64
        1,  # n_datasets
        metadata
    )
end

# Export types and functions
export ROIBatch, SingleROI, LocalizationResult
export roi_to_camera_coords, create_localization_result
export to_emitter2dfit, to_smld