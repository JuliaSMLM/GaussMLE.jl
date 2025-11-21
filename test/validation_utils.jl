"""
Utilities for model validation testing
"""

using Random
using Statistics
using Distributions

"""
    extract_roi_coords(smld::SMLMData.BasicSMLD, corners::Matrix{Int32}, roi_size::Int, pixel_size::Real)

Extract ROI-local coordinates from BasicSMLD using actual ROI corners.

Converts fitted results (in camera/micron coordinates) back to ROI-local coordinates
using the true corner positions from the ROIBatch.

# Arguments
- `smld`: Fitted localization results
- `corners`: Actual ROI corners from ROIBatch (2×n_rois matrix, 1-indexed camera positions)
- `roi_size`: Size of ROI (pixels)
- `pixel_size`: Camera pixel size (microns)

# Returns
NamedTuple with ROI-local parameters (in pixels):
- x_roi, y_roi: Positions within ROI (1-indexed pixels)
- photons, bg: Photometry (unchanged)
- σ_x, σ_y: Uncertainties (in pixels)
"""
function extract_roi_coords(smld::SMLMData.BasicSMLD, corners::Matrix{Int32}, roi_size::Int, pixel_size::Real)
    n = length(smld.emitters)
    @assert size(corners, 2) == n "Number of corners must match number of emitters"

    x_roi = Vector{Float32}(undef, n)
    y_roi = Vector{Float32}(undef, n)
    photons = Vector{Float32}(undef, n)
    bg = Vector{Float32}(undef, n)
    σ_x = Vector{Float32}(undef, n)
    σ_y = Vector{Float32}(undef, n)

    for (i, e) in enumerate(smld.emitters)
        # Use actual corner from ROIBatch
        corner_x = Float32(corners[1, i])
        corner_y = Float32(corners[2, i])

        # Forward transform (see roi_batch.jl):
        #   x_camera = corner_x + x_roi - 1
        #   x_microns = (x_camera - 1) * pixel_size
        # Combined: x_microns = (corner_x + x_roi - 2) * pixel_size
        # Reverse transform:
        #   x_roi = (x_microns / pixel_size) - corner_x + 2
        x_roi[i] = (e.x / pixel_size) - corner_x + 2.0f0
        y_roi[i] = (e.y / pixel_size) - corner_y + 2.0f0

        photons[i] = e.photons
        bg[i] = e.bg
        σ_x[i] = e.σ_x / pixel_size  # Convert to pixels
        σ_y[i] = e.σ_y / pixel_size
    end

    return (x_roi=x_roi, y_roi=y_roi, photons=photons, bg=bg, σ_x=σ_x, σ_y=σ_y)
end

"""
    generate_test_data(model_type, n_blobs, box_size; psf_model=nothing, kwargs...)

Generate synthetic data with known ground truth for testing
"""
function generate_test_data(
    model_type::Symbol,
    n_blobs::Int,
    box_size::Int;
    psf_model::Union{Nothing, GaussMLE.PSFModel} = nothing,
    n_photons::Float32 = 1000.0f0,
    background::Float32 = 5.0f0,  # Optimal background for unbiased estimation
    sigma::Float32 = 1.3f0,
    position_std::Float32 = 0.5f0,
    seed::Int = 42
)
    Random.seed!(seed)
    
    # Initialize data array
    data = zeros(Float32, box_size, box_size, n_blobs)
    
    # Store true parameters
    true_params = Dict{Symbol, Vector{Float32}}()
    
    # Generate blobs
    for k in 1:n_blobs
        # True parameters with small variations
        x_true = Float32(box_size/2 + position_std * randn())
        y_true = Float32(box_size/2 + position_std * randn())
        n_true = n_photons * Float32(0.8 + 0.4 * rand())  # ±20% variation
        bg_true = background * Float32(0.8 + 0.4 * rand())
        
        # Model-specific parameters
        if model_type == :xynb
            # Fixed sigma model
            true_params[:x] = push!(get(true_params, :x, Float32[]), x_true)
            true_params[:y] = push!(get(true_params, :y, Float32[]), y_true)
            true_params[:photons] = push!(get(true_params, :photons, Float32[]), n_true)
            true_params[:background] = push!(get(true_params, :background, Float32[]), bg_true)
            
            # Generate data
            for j in 1:box_size, i in 1:box_size
                mu = generate_pixel_value(i, j, x_true, y_true, n_true, bg_true, sigma, sigma)
                data[i, j, k] = Float32(rand(Poisson(mu)))
            end
            
        elseif model_type == :xynbs
            # Variable sigma model
            sigma_true = sigma * Float32(0.8 + 0.4 * rand())
            
            true_params[:x] = push!(get(true_params, :x, Float32[]), x_true)
            true_params[:y] = push!(get(true_params, :y, Float32[]), y_true)
            true_params[:photons] = push!(get(true_params, :photons, Float32[]), n_true)
            true_params[:background] = push!(get(true_params, :background, Float32[]), bg_true)
            true_params[:sigma] = push!(get(true_params, :sigma, Float32[]), sigma_true)
            
            for j in 1:box_size, i in 1:box_size
                mu = generate_pixel_value(i, j, x_true, y_true, n_true, bg_true, sigma_true, sigma_true)
                data[i, j, k] = Float32(rand(Poisson(mu)))
            end
            
        elseif model_type == :xynbsxsy
            # Anisotropic model
            sigma_x_true = sigma * Float32(0.8 + 0.4 * rand())
            sigma_y_true = sigma * Float32(0.8 + 0.4 * rand())
            
            true_params[:x] = push!(get(true_params, :x, Float32[]), x_true)
            true_params[:y] = push!(get(true_params, :y, Float32[]), y_true)
            true_params[:photons] = push!(get(true_params, :photons, Float32[]), n_true)
            true_params[:background] = push!(get(true_params, :background, Float32[]), bg_true)
            true_params[:sigma_x] = push!(get(true_params, :sigma_x, Float32[]), sigma_x_true)
            true_params[:sigma_y] = push!(get(true_params, :sigma_y, Float32[]), sigma_y_true)
            
            for j in 1:box_size, i in 1:box_size
                mu = generate_pixel_value(i, j, x_true, y_true, n_true, bg_true, sigma_x_true, sigma_y_true)
                data[i, j, k] = Float32(rand(Poisson(mu)))
            end
            
        elseif model_type == :xynbz
            # Astigmatic 3D model - use actual PSF model for data generation
            @assert !isnothing(psf_model) "psf_model required for :xynbz model type"
            @assert psf_model isa GaussMLE.AstigmaticXYZNB "psf_model must be AstigmaticXYZNB for :xynbz"

            z_true = Float32(-250.0 + 500.0 * rand())  # Uniform between -250 and +250 nm

            true_params[:x] = push!(get(true_params, :x, Float32[]), x_true)
            true_params[:y] = push!(get(true_params, :y, Float32[]), y_true)
            true_params[:z] = push!(get(true_params, :z, Float32[]), z_true)
            true_params[:photons] = push!(get(true_params, :photons, Float32[]), n_true)
            true_params[:background] = push!(get(true_params, :background, Float32[]), bg_true)

            # Use actual PSF model to compute widths (matches implementation exactly)
            αx = GaussMLE.GaussLib.compute_alpha((z_true - psf_model.γ), psf_model.Ax, psf_model.Bx, psf_model.d)
            αy = GaussMLE.GaussLib.compute_alpha((z_true + psf_model.γ), psf_model.Ay, psf_model.By, psf_model.d)
            sigma_x_z = psf_model.σx₀ * sqrt(αx)
            sigma_y_z = psf_model.σy₀ * sqrt(αy)

            for j in 1:box_size, i in 1:box_size
                mu = generate_pixel_value(i, j, x_true, y_true, n_true, bg_true, sigma_x_z, sigma_y_z)
                data[i, j, k] = Float32(rand(Poisson(mu)))
            end
        end
    end
    
    return data, true_params
end

"""
    generate_pixel_value(i, j, x, y, n, bg, sigma_x, sigma_y)

Generate expected pixel value for an integrated Gaussian PSF
"""
function generate_pixel_value(i, j, x, y, n, bg, sigma_x, sigma_y)
    # Integrated Gaussian model (matches fitting code)
    dx = Float32(j) - x
    dy = Float32(i) - y

    # Use the same integrated Gaussian as in fitting (from GaussLib)
    psf_x = GaussMLE.GaussLib.integral_gaussian_1d(j, x, sigma_x)
    psf_y = GaussMLE.GaussLib.integral_gaussian_1d(i, y, sigma_y)

    return bg + n * psf_x * psf_y
end

"""
    validate_fitting_results(smld, true_params, param_name; bias_tol, std_tol, roi_size)

Validate that fitted parameters match ground truth within tolerances.
Uses extract_roi_coords() to properly convert from camera coordinates to ROI-local coordinates.
"""
function validate_fitting_results(
    smld::SMLMData.BasicSMLD,
    true_params::Dict{Symbol, Vector{Float32}},
    param_name::Symbol;
    bias_tol::Float32 = 0.1f0,
    std_tol::Float32 = 0.2f0,  # 20% tolerance on std matching
    roi_size::Int = 11,  # Default ROI size for coordinate extraction
    verbose::Bool = false
)
    # Extract ROI-local coordinates
    # For fit(Array), use dummy corners [1, 12, 23, ...] (matches interface.jl)
    n = length(smld.emitters)
    dummy_corners = zeros(Int32, 2, n)
    for i in 1:n
        dummy_corners[1, i] = Int32(1 + (i-1) * roi_size)
        dummy_corners[2, i] = Int32(1)
    end

    pixel_size = smld.camera.pixel_edges_x[2] - smld.camera.pixel_edges_x[1]
    coords = extract_roi_coords(smld, dummy_corners, roi_size, pixel_size)

    # All parameters are now accessible via custom emitter types!
    # - sigma: Emitter2DFitSigma
    # - sigma_x, sigma_y: Emitter2DFitSigmaXY
    # - z: Emitter3DFit

    # Get fitted values and uncertainties based on parameter name
    # Uses dispatch on emitter type to access PSF-specific fields
    fitted_vals = if param_name == :x
        coords.x_roi
    elseif param_name == :y
        coords.y_roi
    elseif param_name == :z
        # z is in microns for Emitter3DFit, convert to pixels for comparison
        Float32[e.z / pixel_size for e in smld.emitters]
    elseif param_name == :sigma
        # σ in microns for Emitter2DFitSigma, convert to pixels
        Float32[e.σ / pixel_size for e in smld.emitters]
    elseif param_name == :sigma_x
        # σx in microns for Emitter2DFitSigmaXY, convert to pixels
        Float32[e.σx / pixel_size for e in smld.emitters]
    elseif param_name == :sigma_y
        # σy in microns for Emitter2DFitSigmaXY, convert to pixels
        Float32[e.σy / pixel_size for e in smld.emitters]
    elseif param_name == :photons
        coords.photons
    elseif param_name == :background
        coords.bg
    else
        error("Unknown parameter name: $param_name")
    end

    fitted_errors = if param_name == :x
        coords.σ_x
    elseif param_name == :y
        coords.σ_y
    elseif param_name == :z
        # σ_z is in microns for Emitter3DFit, convert to pixels
        Float32[e.σ_z / pixel_size for e in smld.emitters]
    elseif param_name == :sigma
        # σ_σ in microns for Emitter2DFitSigma, convert to pixels
        Float32[e.σ_σ / pixel_size for e in smld.emitters]
    elseif param_name == :sigma_x
        # σ_σx in microns for Emitter2DFitSigmaXY, convert to pixels
        Float32[e.σ_σx / pixel_size for e in smld.emitters]
    elseif param_name == :sigma_y
        # σ_σy in microns for Emitter2DFitSigmaXY, convert to pixels
        Float32[e.σ_σy / pixel_size for e in smld.emitters]
    elseif param_name == :photons
        Float32[e.σ_photons for e in smld.emitters]
    elseif param_name == :background
        Float32[e.σ_bg for e in smld.emitters]
    else
        error("Unknown parameter name: $param_name")
    end

    # Get true values
    true_vals = true_params[param_name]

    # Compute statistics
    # For spatial parameters (x, y, z), compute bias properly
    # For photometry (photons, background), compute actual errors
    if param_name in [:x, :y, :z]
        errors = fitted_vals .- true_vals
        bias = mean(errors)
        empirical_std = std(errors)
        mean_reported_std = mean(fitted_errors)
        std_ratio = empirical_std / mean_reported_std
        bias_pass = abs(bias) < bias_tol
        std_pass = abs(1.0 - std_ratio) < std_tol
    else
        # For photons/background, compute actual errors
        errors = fitted_vals .- true_vals
        bias = mean(errors)
        empirical_std = std(errors)
        mean_reported_std = mean(fitted_errors)
        std_ratio = empirical_std / mean_reported_std
        bias_pass = abs(bias) < bias_tol
        std_pass = abs(1.0 - std_ratio) < std_tol
    end

    if verbose
        println("\nParameter: $param_name")
        println("  Bias: $bias (tolerance: ±$bias_tol)")
        println("  Empirical STD: $empirical_std")
        println("  Mean reported STD: $mean_reported_std")
        println("  STD ratio: $std_ratio (should be ≈1.0)")
        println("  Bias test: $(bias_pass ? "PASS" : "FAIL")")
        println("  STD test: $(std_pass ? "PASS" : "FAIL")")
    end

    return (
        bias = bias,
        empirical_std = empirical_std,
        mean_reported_std = mean_reported_std,
        std_ratio = std_ratio,
        bias_pass = bias_pass,
        std_pass = std_pass,
        overall_pass = bias_pass && std_pass
    )
end

"""
    run_model_validation(model_type, psf_model, n_blobs; kwargs...)

Run complete validation for a model configuration
"""
function run_model_validation(
    model_type::Symbol,
    psf_model::GaussMLE.PSFModel,
    n_blobs::Int = 1000;
    box_size::Int = 11,
    device = GaussMLE.CPU(),
    verbose::Bool = false,
    kwargs...
)
    # Generate test data (pass psf_model for astigmatic model)
    data, true_params = generate_test_data(model_type, n_blobs, box_size; psf_model=psf_model, kwargs...)
    
    # Create fitter
    fitter = GaussMLE.GaussMLEFitter(
        psf_model = psf_model,
        device = device,
        iterations = 20
    )
    
    # Fit the data
    smld = GaussMLE.fit(fitter, data)

    # Validate each parameter
    validation_results = Dict{Symbol, Any}()
    all_pass = true
    
    # Define parameters to validate based on model type
    params_to_validate = if model_type == :xynb
        [:x, :y, :photons, :background]
    elseif model_type == :xynbs
        [:x, :y, :photons, :background, :sigma]
    elseif model_type == :xynbsxsy
        [:x, :y, :photons, :background, :sigma_x, :sigma_y]
    elseif model_type == :xynbz
        [:x, :y, :z, :photons, :background]
    else
        error("Unknown model type: $model_type")
    end
    
    # Define tolerances for each parameter
    tolerances = Dict(
        :x => (bias_tol=0.15f0, std_tol=0.1f0),  # 10% tolerance for std/CRLB ratio, 0.15 pix bias tolerance
        :y => (bias_tol=0.15f0, std_tol=0.1f0),  # Increased from 0.05 to account for coordinate conversion
        :z => (bias_tol=30.0f0, std_tol=0.1f0),  # 10% tolerance for std/CRLB ratio
        :photons => (bias_tol=100.0f0, std_tol=0.1f0),  # 10% tolerance
        :background => (bias_tol=2.0f0, std_tol=0.1f0),  # 10% tolerance
        :sigma => (bias_tol=0.05f0, std_tol=0.1f0),
        :sigma_x => (bias_tol=0.05f0, std_tol=0.1f0),
        :sigma_y => (bias_tol=0.05f0, std_tol=0.1f0),
    )
    
    if verbose
        println("\n" * "="^60)
        println("Model Validation: $model_type")
        println("Device: $(typeof(device))")
        println("Number of blobs: $n_blobs")
        println("="^60)
    end
    
    for param in params_to_validate
        tol = tolerances[param]
        result = validate_fitting_results(
            smld, true_params, param;
            bias_tol = tol.bias_tol,
            std_tol = tol.std_tol,
            roi_size = box_size,
            verbose = verbose
        )
        validation_results[param] = result
        all_pass = all_pass && result.overall_pass
    end
    
    if verbose
        println("\n" * "="^60)
        println("Overall result: $(all_pass ? "PASS ✓" : "FAIL ✗")")
        println("="^60)
    end
    
    return all_pass, validation_results
end

# Export functions
export generate_test_data, validate_fitting_results, run_model_validation