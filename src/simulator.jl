"""
Camera-aware simulator with PSF-specific defaults
Generates ROIBatch with appropriate noise models based on camera type
"""

using Random
using Distributions
using SMLMData
using StaticArrays
using .GaussLib: integral_gaussian_1d, compute_alpha

# PSF-specific default parameters
get_default_params(::GaussianXYNB) = Float32[6.0, 6.0, 1000.0, 10.0]
get_default_params(::GaussianXYNBS) = Float32[6.0, 6.0, 1000.0, 10.0, 1.3]
get_default_params(::GaussianXYNBSXSY) = Float32[6.0, 6.0, 1000.0, 10.0, 1.3, 1.3]
get_default_params(::AstigmaticXYZNB) = Float32[6.0, 6.0, 0.0, 1000.0, 10.0]

"""
    generate_roi_batch(camera, psf_model; kwargs...) → ROIBatch

Generate synthetic ROI data with camera-appropriate noise.

# Arguments
- `camera::AbstractCamera`: Camera model (IdealCamera or SCMOSCamera)
- `psf_model::PSFModel`: PSF model determining parameter structure

# Keywords
- `n_rois::Int = 100`: Number of ROIs to generate
- `roi_size::Int = 11`: Size of each ROI (square)
- `true_params::Union{Nothing, Matrix} = nothing`: Parameters or use PSF defaults
- `corners::Union{Nothing, Matrix{Int32}} = nothing`: ROI corners or auto-generate
- `frame_indices::Union{Nothing, Vector{Int32}} = nothing`: Frame indices or all frame 1
- `xy_variation::Float32 = 1.0`: Position variation (±pixels) when using defaults
- `corner_mode::Symbol = :random`: Corner generation mode (:random, :grid, :clustered)
- `min_spacing::Int = 20`: Minimum pixel spacing between ROIs
- `seed::Union{Nothing, Int} = nothing`: Random seed for reproducibility

# Returns
- `ROIBatch`: Complete ROI batch with camera attached, ready for fitting
"""
function generate_roi_batch(
    camera::SMLMData.AbstractCamera,
    psf_model::PSFModel;
    n_rois::Int = 100,
    roi_size::Int = 11,
    true_params::Union{Nothing, Matrix} = nothing,
    corners::Union{Nothing, Matrix{Int32}} = nothing,
    frame_indices::Union{Nothing, Vector{Int32}} = nothing,
    xy_variation::Float32 = 1.0f0,
    corner_mode::Symbol = :random,
    min_spacing::Int = 20,
    seed::Union{Nothing, Int} = nothing
)
    # Set random seed if provided
    !isnothing(seed) && Random.seed!(seed)
    
    # Generate or validate parameters
    if isnothing(true_params)
        true_params = _generate_default_params(psf_model, n_rois, xy_variation)
    else
        actual_n_rois = size(true_params, 2)
        if actual_n_rois != n_rois
            n_rois = actual_n_rois  # Use actual count from params
        end
    end
    
    # Generate or validate corners
    if isnothing(corners)
        camera_size = (
            length(camera.pixel_edges_x) - 1,
            length(camera.pixel_edges_y) - 1
        )
        corners = _generate_corners(n_rois, camera_size, roi_size, corner_mode, min_spacing)
    else
        @assert size(corners) == (2, n_rois) "Corners must be 2×n_rois"
    end
    
    # Generate or validate frame indices
    if isnothing(frame_indices)
        frame_indices = ones(Int32, n_rois)
    else
        @assert length(frame_indices) == n_rois "Must have one frame index per ROI"
    end
    
    # Generate the ROI data with appropriate noise
    data = _generate_roi_data(camera, psf_model, true_params, corners, roi_size)
    
    # Create and return ROIBatch
    ROIBatch(data, corners, frame_indices, camera)
end

"""
Generate default parameters with PSF-specific values and position variation
"""
function _generate_default_params(psf_model::PSFModel, n_rois::Int, xy_variation::Float32)
    base_params = get_default_params(psf_model)
    n_params = length(base_params)
    
    # Replicate for all ROIs
    params = Matrix{Float32}(undef, n_params, n_rois)
    for i in 1:n_rois
        params[:, i] = base_params
        
        # Add position variation only to x,y (first two parameters)
        if xy_variation > 0
            params[1, i] += (rand() - 0.5f0) * 2.0f0 * xy_variation  # x variation
            params[2, i] += (rand() - 0.5f0) * 2.0f0 * xy_variation  # y variation
        end
    end
    
    return params
end

"""
Generate corners based on specified mode
"""
function _generate_corners(n_rois::Int, camera_size::Tuple{Int,Int}, roi_size::Int,
                          mode::Symbol, min_spacing::Int)
    nx, ny = camera_size
    corners = Matrix{Int32}(undef, 2, n_rois)
    
    # Ensure ROIs fit within camera
    max_x = nx - roi_size + 1
    max_y = ny - roi_size + 1
    
    if mode == :random
        # Random placement with minimum spacing
        placed = Vector{Tuple{Int,Int}}()
        attempts = 0
        max_attempts = n_rois * 100
        
        for i in 1:n_rois
            valid = false
            while !valid && attempts < max_attempts
                attempts += 1
                x = rand(1:max_x)
                y = rand(1:max_y)
                
                # Check minimum spacing
                valid = true
                for (px, py) in placed
                    if abs(x - px) < min_spacing && abs(y - py) < min_spacing
                        valid = false
                        break
                    end
                end
                
                if valid
                    corners[1, i] = x
                    corners[2, i] = y
                    push!(placed, (x, y))
                end
            end
            
            # Fallback if spacing constraint too strict
            if !valid
                corners[1, i] = rand(1:max_x)
                corners[2, i] = rand(1:max_y)
            end
        end
        
    elseif mode == :grid
        # Regular grid pattern
        n_grid = ceil(Int, sqrt(n_rois))
        spacing_x = max(min_spacing, (max_x - 1) ÷ n_grid)
        spacing_y = max(min_spacing, (max_y - 1) ÷ n_grid)
        
        idx = 1
        for j in 1:spacing_y:max_y
            for i in 1:spacing_x:max_x
                if idx <= n_rois
                    corners[1, idx] = i
                    corners[2, idx] = j
                    idx += 1
                end
            end
        end
        
    elseif mode == :clustered
        # Clusters of ROIs (simulating multi-emitter regions)
        n_clusters = max(1, n_rois ÷ 10)
        rois_per_cluster = n_rois ÷ n_clusters
        
        idx = 1
        for c in 1:n_clusters
            # Random cluster center
            cx = rand(roi_size:max_x-roi_size)
            cy = rand(roi_size:max_y-roi_size)
            
            # Place ROIs around cluster center
            for _ in 1:min(rois_per_cluster, n_rois - idx + 1)
                offset_x = rand(-min_spacing:min_spacing)
                offset_y = rand(-min_spacing:min_spacing)
                
                corners[1, idx] = clamp(cx + offset_x, 1, max_x)
                corners[2, idx] = clamp(cy + offset_y, 1, max_y)
                idx += 1
            end
        end
    else
        error("Unknown corner_mode: $mode. Use :random, :grid, or :clustered")
    end
    
    return corners
end

"""
Generate ROI data with camera-specific noise - dispatches on camera type
"""
function _generate_roi_data(camera::SMLMData.IdealCamera, psf_model::PSFModel,
                           true_params::Matrix, corners::Matrix{Int32}, roi_size::Int)
    n_rois = size(true_params, 2)
    data = zeros(Float32, roi_size, roi_size, n_rois)
    
    for k in 1:n_rois
        roi = @view data[:, :, k]
        params = @view true_params[:, k]
        _generate_single_roi!(roi, psf_model, params, camera, corners[:, k])
    end
    
    return data
end

function _generate_roi_data(camera::SMLMData.SCMOSCamera, psf_model::PSFModel,
                           true_params::Matrix, corners::Matrix{Int32}, roi_size::Int)
    n_rois = size(true_params, 2)
    data = zeros(Float32, roi_size, roi_size, n_rois)

    for k in 1:n_rois
        roi = @view data[:, :, k]
        params = @view true_params[:, k]
        _generate_single_roi!(roi, psf_model, params, camera, corners[:, k])
    end

    return data
end

"""
Generate single ROI with IdealCamera (Poisson noise only)
"""
function _generate_single_roi!(roi::AbstractMatrix, psf_model::PSFModel,
                              params::AbstractVector, ::SMLMData.IdealCamera,
                              corner::AbstractVector)
    roi_size = size(roi, 1)
    
    # Generate expected signal for each pixel
    for j in 1:roi_size, i in 1:roi_size
        expected = _evaluate_psf_pixel(psf_model, i, j, params)
        
        # Poisson noise
        if expected > 0
            roi[i, j] = rand(Poisson(expected))
        else
            roi[i, j] = 0
        end
    end
end

"""
Generate single ROI with SMLMData.SCMOSCamera (Poisson + readout noise + ADU conversion)
Simulates full camera pipeline: photons → electrons → ADU
"""
function _generate_single_roi!(roi::AbstractMatrix, psf_model::PSFModel,
                              params::AbstractVector, camera::SMLMData.SCMOSCamera,
                              corner::AbstractVector)
    roi_size = size(roi, 1)
    x_corner, y_corner = corner

    # Generate expected signal for each pixel
    for j in 1:roi_size, i in 1:roi_size
        expected_photons = _evaluate_psf_pixel(psf_model, i, j, params)

        # Apply QE and Poisson noise (photons → electrons)
        qe = camera.qe isa Number ? camera.qe : camera.qe[i + x_corner - 1, j + y_corner - 1]
        expected_electrons = expected_photons * qe
        signal_electrons = expected_electrons > 0 ? Float32(rand(Poisson(expected_electrons))) : 0.0f0

        # Add readout noise (Gaussian, in electrons)
        cam_i = i + x_corner - 1
        cam_j = j + y_corner - 1
        readnoise = camera.readnoise isa Number ? camera.readnoise : camera.readnoise[cam_i, cam_j]
        total_electrons = signal_electrons + randn(Float32) * readnoise

        # Convert to ADU
        gain = camera.gain isa Number ? camera.gain : camera.gain[cam_i, cam_j]
        offset = camera.offset isa Number ? camera.offset : camera.offset[cam_i, cam_j]
        roi[i, j] = total_electrons / gain + offset
    end
end

"""
Evaluate PSF at a single pixel - dispatches on PSF model type
"""
function _evaluate_psf_pixel(psf::GaussianXYNB, i::Int, j::Int, params::AbstractVector)
    x, y, photons, bg = params
    psf_x = integral_gaussian_1d(i, x, psf.σ)
    psf_y = integral_gaussian_1d(j, y, psf.σ)
    return bg + photons * psf_x * psf_y
end

function _evaluate_psf_pixel(::GaussianXYNBS, i::Int, j::Int, params::AbstractVector)
    x, y, photons, bg, σ = params
    psf_x = integral_gaussian_1d(i, x, σ)
    psf_y = integral_gaussian_1d(j, y, σ)
    return bg + photons * psf_x * psf_y
end

function _evaluate_psf_pixel(::GaussianXYNBSXSY, i::Int, j::Int, params::AbstractVector)
    x, y, photons, bg, σx, σy = params
    psf_x = integral_gaussian_1d(i, x, σx)
    psf_y = integral_gaussian_1d(j, y, σy)
    return bg + photons * psf_x * psf_y
end

function _evaluate_psf_pixel(psf::AstigmaticXYZNB, i::Int, j::Int, params::AbstractVector)
    x, y, z, photons, bg = params

    # Width calculation based on z position
    αx = compute_alpha((z - psf.γ), psf.Ax, psf.Bx, psf.d)
    αy = compute_alpha((z + psf.γ), psf.Ay, psf.By, psf.d)
    σx = psf.σx₀ * sqrt(αx)
    σy = psf.σy₀ * sqrt(αy)

    psf_x = integral_gaussian_1d(i, x, σx)
    psf_y = integral_gaussian_1d(j, y, σy)
    return bg + photons * psf_x * psf_y
end

# Export the main function
export generate_roi_batch