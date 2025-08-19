"""
Data simulator for generating synthetic spots with integrated Gaussian PSFs
Provides consistent data generation for tests and examples
"""

using Random
using Distributions
using .GaussLib: integral_gaussian_1d, compute_alpha

"""
    SimulatedSpot

Parameters for a single simulated spot
"""
struct SimulatedSpot{T}
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
function SimulatedSpot(x::T, y::T, photons::T, background::T) where T
    SimulatedSpot{T}(x, y, photons, background, nothing, nothing, nothing, nothing)
end

function SimulatedSpot(x::T, y::T, photons::T, background::T, sigma::T) where T
    SimulatedSpot{T}(x, y, photons, background, sigma, nothing, nothing, nothing)
end

"""
    generate_spots_data(psf_model, spots, roi_size; camera_model=IdealCamera(), seed=nothing)

Generate synthetic spot data using integrated Gaussian PSFs

# Arguments
- `psf_model`: PSF model to use for generation (must match fitting model)
- `spots`: Vector of SimulatedSpot or single spot parameters
- `roi_size`: Size of each ROI (roi_size × roi_size pixels)
- `camera_model`: Camera noise model (IdealCamera or SCMOSCamera)
- `seed`: Random seed for reproducibility

# Returns
- `data`: Array of size (roi_size, roi_size, n_spots) with synthetic data
- `true_params`: Dictionary with true parameter values for validation
"""
function generate_spots_data(
    psf_model::PSFModel,
    spots::Vector{<:SimulatedSpot},
    roi_size::Int;
    camera_model::CameraModel = IdealCamera(),
    seed::Union{Int, Nothing} = nothing
)
    !isnothing(seed) && Random.seed!(seed)
    
    n_spots = length(spots)
    T = eltype(spots[1].x)
    data = zeros(T, roi_size, roi_size, n_spots)
    
    # Store true parameters for validation
    true_params = Dict{Symbol, Vector{T}}()
    true_params[:x] = [s.x for s in spots]
    true_params[:y] = [s.y for s in spots]
    true_params[:photons] = [s.photons for s in spots]
    true_params[:background] = [s.background for s in spots]
    
    # Add model-specific parameters
    if psf_model isa GaussianXYNBS
        true_params[:sigma] = [s.sigma for s in spots]
    elseif psf_model isa GaussianXYNBSXSY
        true_params[:sigma_x] = [s.sigma_x for s in spots]
        true_params[:sigma_y] = [s.sigma_y for s in spots]
    elseif psf_model isa AstigmaticXYZNB
        true_params[:z] = [s.z for s in spots]
    end
    
    # Generate data for each spot
    for (k, spot) in enumerate(spots)
        roi = @view data[:, :, k]
        generate_single_spot!(roi, psf_model, spot, camera_model)
    end
    
    return data, true_params
end

"""
    generate_single_spot!(roi, psf_model, spot, camera_model)

Generate data for a single spot into a pre-allocated ROI
"""
function generate_single_spot!(
    roi::AbstractMatrix{T},
    psf_model::GaussianXYNB,
    spot::SimulatedSpot{T},
    camera_model::CameraModel
) where T
    roi_size = size(roi, 1)
    
    for j in 1:roi_size, i in 1:roi_size
        # Integrated Gaussian PSF
        psf_x = integral_gaussian_1d(i, spot.x, psf_model.σ)
        psf_y = integral_gaussian_1d(j, spot.y, psf_model.σ)
        
        # Expected photon count
        mu = spot.background + spot.photons * psf_x * psf_y
        
        # Generate noisy data based on camera model
        roi[i, j] = generate_pixel_value(mu, camera_model, i, j)
    end
end

function generate_single_spot!(
    roi::AbstractMatrix{T},
    psf_model::GaussianXYNBS,
    spot::SimulatedSpot{T},
    camera_model::CameraModel
) where T
    roi_size = size(roi, 1)
    sigma = isnothing(spot.sigma) ? T(1.3) : spot.sigma
    
    for j in 1:roi_size, i in 1:roi_size
        psf_x = integral_gaussian_1d(i, spot.x, sigma)
        psf_y = integral_gaussian_1d(j, spot.y, sigma)
        mu = spot.background + spot.photons * psf_x * psf_y
        roi[i, j] = generate_pixel_value(mu, camera_model, i, j)
    end
end

function generate_single_spot!(
    roi::AbstractMatrix{T},
    psf_model::GaussianXYNBSXSY,
    spot::SimulatedSpot{T},
    camera_model::CameraModel
) where T
    roi_size = size(roi, 1)
    sigma_x = isnothing(spot.sigma_x) ? T(1.3) : spot.sigma_x
    sigma_y = isnothing(spot.sigma_y) ? T(1.3) : spot.sigma_y
    
    for j in 1:roi_size, i in 1:roi_size
        psf_x = integral_gaussian_1d(i, spot.x, sigma_x)
        psf_y = integral_gaussian_1d(j, spot.y, sigma_y)
        mu = spot.background + spot.photons * psf_x * psf_y
        roi[i, j] = generate_pixel_value(mu, camera_model, i, j)
    end
end

function generate_single_spot!(
    roi::AbstractMatrix{T},
    psf_model::AstigmaticXYZNB,
    spot::SimulatedSpot{T},
    camera_model::CameraModel
) where T
    roi_size = size(roi, 1)
    z = isnothing(spot.z) ? zero(T) : spot.z
    
    # Compute z-dependent sigmas
    zx = z - psf_model.γ
    zy = z + psf_model.γ
    σx = psf_model.σx₀ * sqrt(compute_alpha(zx, psf_model.Ax, psf_model.Bx, psf_model.d))
    σy = psf_model.σy₀ * sqrt(compute_alpha(zy, psf_model.Ay, psf_model.By, psf_model.d))
    
    for j in 1:roi_size, i in 1:roi_size
        psf_x = integral_gaussian_1d(i, spot.x, σx)
        psf_y = integral_gaussian_1d(j, spot.y, σy)
        mu = spot.background + spot.photons * psf_x * psf_y
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
    generate_random_spots(n_spots, roi_size; kwargs...)

Generate random spots with variations for testing

# Keyword arguments
- `x_range`: Range for x positions (default: centered with 0.5 pixel std)
- `y_range`: Range for y positions (default: centered with 0.5 pixel std)
- `photons_range`: Range for photon counts (default: 800-1200)
- `background_range`: Range for background (default: 8-12)
- `sigma_range`: Range for PSF width (for variable sigma models)
- `z_range`: Range for z position (for 3D models)
- `seed`: Random seed
"""
function generate_random_spots(
    n_spots::Int,
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
    spots = SimulatedSpot{T}[]
    
    for k in 1:n_spots
        x = T(x_mean + x_std * randn())
        y = T(y_mean + y_std * randn())
        photons = T(max(100, photons_mean + photons_std * randn()))
        background = T(max(0, background_mean + background_std * randn()))
        
        if model_type == :xynb
            push!(spots, SimulatedSpot(x, y, photons, background))
        elseif model_type == :xynbs
            sigma = T(max(0.5, sigma_mean + sigma_std * randn()))
            push!(spots, SimulatedSpot{T}(x, y, photons, background, sigma, nothing, nothing, nothing))
        elseif model_type == :xynbsxsy
            sigma_x = T(max(0.5, sigma_mean + sigma_std * randn()))
            sigma_y = T(max(0.5, sigma_mean + sigma_std * randn()))
            push!(spots, SimulatedSpot{T}(x, y, photons, background, nothing, sigma_x, sigma_y, nothing))
        elseif model_type == :xynbz
            z = T(z_mean + z_std * randn())
            push!(spots, SimulatedSpot{T}(x, y, photons, background, nothing, nothing, nothing, z))
        end
    end
    
    return spots
end

# Export functions
export SimulatedSpot, generate_spots_data, generate_single_spot!, generate_random_spots, generate_pixel_value