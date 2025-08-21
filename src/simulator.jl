"""
Data simulator for generating synthetic blobs with integrated Gaussian PSFs
Provides consistent data generation for tests and examples
"""

using Random
using Distributions
using .GaussLib: integral_gaussian_1d, compute_alpha

"""
    SimulatedBlob

Parameters for a single simulated blob
"""
struct SimulatedBlob{T}
    x::T
    y::T
    photons::T
    background::T
    # Optional parameters for different models
    sigma::Union{T, Nothing}
    sigma_x::Union{T, Nothing}
    sigma_y::Union{T, Nothing}
    z::Union{T, Nothing}
end

# Convenience constructors
function SimulatedBlob(x::T, y::T, photons::T, background::T) where T
    SimulatedBlob{T}(x, y, photons, background, nothing, nothing, nothing, nothing)
end

function SimulatedBlob(x::T, y::T, photons::T, background::T, sigma::T) where T
    SimulatedBlob{T}(x, y, photons, background, sigma, nothing, nothing, nothing)
end

"""
    generate_blobs_data(psf_model, blobs, roi_size; camera_model=IdealCamera(), seed=nothing)

Generate synthetic blob data using integrated Gaussian PSFs

# Arguments
- `psf_model`: PSF model to use for generation (must match fitting model)
- `blobs`: Vector of SimulatedBlob or single blob parameters
- `roi_size`: Size of each ROI (roi_size × roi_size pixels)
- `camera_model`: Camera noise model (IdealCamera or SCMOSCamera)
- `seed`: Random seed for reproducibility

# Returns
- `data`: Array of size (roi_size, roi_size, n_blobs) with synthetic data
- `true_params`: Dictionary with true parameter values for validation
"""
function generate_blobs_data(
    psf_model::PSFModel,
    blobs::Vector{<:SimulatedBlob},
    roi_size::Int;
    camera_model::CameraModel = IdealCamera(),
    seed::Union{Int, Nothing} = nothing
)
    !isnothing(seed) && Random.seed!(seed)
    
    n_blobs = length(blobs)
    T = eltype(blobs[1].x)
    data = zeros(T, roi_size, roi_size, n_blobs)
    
    # Store true parameters for validation
    true_params = Dict{Symbol, Vector{T}}()
    true_params[:x] = [s.x for s in blobs]
    true_params[:y] = [s.y for s in blobs]
    true_params[:photons] = [s.photons for s in blobs]
    true_params[:background] = [s.background for s in blobs]
    
    # Add model-specific parameters
    if psf_model isa GaussianXYNBS
        true_params[:sigma] = [s.sigma for s in blobs]
    elseif psf_model isa GaussianXYNBSXSY
        true_params[:sigma_x] = [s.sigma_x for s in blobs]
        true_params[:sigma_y] = [s.sigma_y for s in blobs]
    elseif psf_model isa AstigmaticXYZNB
        true_params[:z] = [s.z for s in blobs]
    end
    
    # Generate data for each blob
    for (k, blob) in enumerate(blobs)
        roi = @view data[:, :, k]
        generate_single_blob!(roi, psf_model, blob, camera_model)
    end
    
    return data, true_params
end

"""
    generate_single_blob!(roi, psf_model, blob, camera_model)

Generate data for a single blob into a pre-allocated ROI
"""
function generate_single_blob!(
    roi::AbstractMatrix{T},
    psf_model::GaussianXYNB,
    blob::SimulatedBlob{T},
    camera_model::CameraModel
) where T
    roi_size = size(roi, 1)
    
    for j in 1:roi_size, i in 1:roi_size
        # Integrated Gaussian PSF
        psf_x = integral_gaussian_1d(i, blob.x, psf_model.σ)
        psf_y = integral_gaussian_1d(j, blob.y, psf_model.σ)
        
        # Expected photon count
        mu = blob.background + blob.photons * psf_x * psf_y
        
        # Generate noisy data based on camera model
        roi[i, j] = generate_pixel_value(mu, camera_model, i, j)
    end
end

function generate_single_blob!(
    roi::AbstractMatrix{T},
    psf_model::GaussianXYNBS,
    blob::SimulatedBlob{T},
    camera_model::CameraModel
) where T
    roi_size = size(roi, 1)
    sigma = isnothing(blob.sigma) ? T(1.3) : blob.sigma
    
    for j in 1:roi_size, i in 1:roi_size
        psf_x = integral_gaussian_1d(i, blob.x, sigma)
        psf_y = integral_gaussian_1d(j, blob.y, sigma)
        mu = blob.background + blob.photons * psf_x * psf_y
        roi[i, j] = generate_pixel_value(mu, camera_model, i, j)
    end
end

function generate_single_blob!(
    roi::AbstractMatrix{T},
    psf_model::GaussianXYNBSXSY,
    blob::SimulatedBlob{T},
    camera_model::CameraModel
) where T
    roi_size = size(roi, 1)
    sigma_x = isnothing(blob.sigma_x) ? T(1.3) : blob.sigma_x
    sigma_y = isnothing(blob.sigma_y) ? T(1.3) : blob.sigma_y
    
    for j in 1:roi_size, i in 1:roi_size
        psf_x = integral_gaussian_1d(i, blob.x, sigma_x)
        psf_y = integral_gaussian_1d(j, blob.y, sigma_y)
        mu = blob.background + blob.photons * psf_x * psf_y
        roi[i, j] = generate_pixel_value(mu, camera_model, i, j)
    end
end

function generate_single_blob!(
    roi::AbstractMatrix{T},
    psf_model::AstigmaticXYZNB,
    blob::SimulatedBlob{T},
    camera_model::CameraModel
) where T
    roi_size = size(roi, 1)
    z = isnothing(blob.z) ? zero(T) : blob.z
    
    # Compute z-dependent sigmas
    zx = z - psf_model.γ
    zy = z + psf_model.γ
    σx = psf_model.σx₀ * sqrt(compute_alpha(zx, psf_model.Ax, psf_model.Bx, psf_model.d))
    σy = psf_model.σy₀ * sqrt(compute_alpha(zy, psf_model.Ay, psf_model.By, psf_model.d))
    
    for j in 1:roi_size, i in 1:roi_size
        psf_x = integral_gaussian_1d(i, blob.x, σx)
        psf_y = integral_gaussian_1d(j, blob.y, σy)
        mu = blob.background + blob.photons * psf_x * psf_y
        roi[i, j] = generate_pixel_value(mu, camera_model, i, j)
    end
end

"""
    generate_pixel_value(mu, camera_model, i, j)

Generate a pixel value based on expected photons and camera noise model
"""
function generate_pixel_value(mu::T, ::IdealCamera, i, j) where T
    # Poisson noise only
    return mu > 0 ? T(rand(Poisson(mu))) : zero(T)
end

function generate_pixel_value(mu::T, camera::SCMOSCamera, i, j) where T
    # Poisson noise + Gaussian readout noise
    if mu > 0
        poisson_value = T(rand(Poisson(mu)))
        readout_std = sqrt(camera.variance_map[i, j])
        return poisson_value + randn(T) * readout_std
    else
        return randn(T) * sqrt(camera.variance_map[i, j])
    end
end

"""
    generate_random_blobs(n_blobs, roi_size; kwargs...)

Generate random blobs with variations for testing

# Keyword arguments
- `x_range`: Range for x positions (default: centered with 0.5 pixel std)
- `y_range`: Range for y positions (default: centered with 0.5 pixel std)
- `photons_range`: Range for photon counts (default: 800-1200)
- `background_range`: Range for background (default: 8-12)
- `sigma_range`: Range for PSF width (for variable sigma models)
- `z_range`: Range for z position (for 3D models)
- `seed`: Random seed
"""
function generate_random_blobs(
    n_blobs::Int,
    roi_size::Int;
    x_mean = roi_size/2 + 0.5,
    y_mean = roi_size/2 + 0.5,
    x_std = 0.5f0,
    y_std = 0.5f0,
    photons_mean = 1000.0f0,
    photons_std = 100.0f0,
    background_mean = 10.0f0,
    background_std = 1.0f0,
    sigma_mean = 1.3f0,
    sigma_std = 0.1f0,
    z_mean = 0.0f0,
    z_std = 200.0f0,
    model_type = :xynb,
    seed = nothing
)
    !isnothing(seed) && Random.seed!(seed)
    
    T = Float32
    blobs = SimulatedBlob{T}[]
    
    for k in 1:n_blobs
        x = T(x_mean + x_std * randn())
        y = T(y_mean + y_std * randn())
        photons = T(max(100, photons_mean + photons_std * randn()))
        background = T(max(0, background_mean + background_std * randn()))
        
        if model_type == :xynb
            push!(blobs, SimulatedBlob(x, y, photons, background))
        elseif model_type == :xynbs
            sigma = T(max(0.5, sigma_mean + sigma_std * randn()))
            push!(blobs, SimulatedBlob{T}(x, y, photons, background, sigma, nothing, nothing, nothing))
        elseif model_type == :xynbsxsy
            sigma_x = T(max(0.5, sigma_mean + sigma_std * randn()))
            sigma_y = T(max(0.5, sigma_mean + sigma_std * randn()))
            push!(blobs, SimulatedBlob{T}(x, y, photons, background, nothing, sigma_x, sigma_y, nothing))
        elseif model_type == :xynbz
            z = T(z_mean + z_std * randn())
            push!(blobs, SimulatedBlob{T}(x, y, photons, background, nothing, nothing, nothing, z))
        end
    end
    
    return blobs
end

# Export functions
export SimulatedBlob, generate_blobs_data, generate_single_blob!, generate_random_blobs, generate_pixel_value