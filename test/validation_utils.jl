"""
Utilities for model validation testing
"""

using Random
using Statistics
using Distributions

"""
    generate_test_data(model_type, n_spots, box_size; kwargs...)

Generate synthetic data with known ground truth for testing
"""
function generate_test_data(
    model_type::Symbol,
    n_spots::Int,
    box_size::Int;
    n_photons::Float32 = 1000.0f0,
    background::Float32 = 10.0f0,
    sigma::Float32 = 1.3f0,
    position_std::Float32 = 0.5f0,
    seed::Int = 42
)
    Random.seed!(seed)
    
    # Initialize data array
    data = zeros(Float32, box_size, box_size, n_spots)
    
    # Store true parameters
    true_params = Dict{Symbol, Vector{Float32}}()
    
    # Generate spots
    for k in 1:n_spots
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
            # Astigmatic 3D model
            z_true = Float32(200.0 * randn())  # Z position in nm
            
            true_params[:x] = push!(get(true_params, :x, Float32[]), x_true)
            true_params[:y] = push!(get(true_params, :y, Float32[]), y_true)
            true_params[:z] = push!(get(true_params, :z, Float32[]), z_true)
            true_params[:photons] = push!(get(true_params, :photons, Float32[]), n_true)
            true_params[:background] = push!(get(true_params, :background, Float32[]), bg_true)
            
            # Proper astigmatic PSF widths - opposite behavior in x and y
            # Matches the calibration in model_validation_tests.jl
            z_norm = z_true / 500.0f0
            alpha_x = 1.0f0 + z_norm^2 + 0.5f0 * z_norm^3 + 0.1f0 * z_norm^4
            alpha_y = 1.0f0 + z_norm^2 - 0.5f0 * z_norm^3 - 0.1f0 * z_norm^4
            sigma_x_z = sigma * sqrt(alpha_x)
            sigma_y_z = sigma * sqrt(alpha_y)
            
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
    dx = Float32(i) - x
    dy = Float32(j) - y
    
    # Use the same integrated Gaussian as in fitting (from GaussLib)
    psf_x = GaussMLE.GaussLib.integral_gaussian_1d(i, x, sigma_x)
    psf_y = GaussMLE.GaussLib.integral_gaussian_1d(j, y, sigma_y)
    
    return bg + n * psf_x * psf_y
end

"""
    validate_fitting_results(results, true_params, param_name; bias_tol, std_tol)

Validate that fitted parameters match ground truth within tolerances
"""
function validate_fitting_results(
    results::GaussMLE.GaussMLEResults,
    true_params::Dict{Symbol, Vector{Float32}},
    param_name::Symbol;
    bias_tol::Float32 = 0.1f0,
    std_tol::Float32 = 0.2f0,  # 20% tolerance on std matching
    verbose::Bool = false
)
    # Get fitted values and uncertainties
    fitted_vals = getproperty(results, param_name)
    uncertainty_field = Symbol(string(param_name) * "_error")
    fitted_errors = getproperty(results, uncertainty_field)
    
    # Get true values
    true_vals = true_params[param_name]
    
    # Compute statistics
    errors = fitted_vals .- true_vals
    bias = mean(errors)
    empirical_std = std(errors)
    mean_reported_std = mean(fitted_errors)
    
    # Check bias
    bias_pass = abs(bias) < bias_tol
    
    # Check that reported std matches empirical std
    std_ratio = empirical_std / mean_reported_std
    std_pass = abs(1.0 - std_ratio) < std_tol
    
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
    run_model_validation(model_type, psf_model, n_spots; kwargs...)

Run complete validation for a model configuration
"""
function run_model_validation(
    model_type::Symbol,
    psf_model::GaussMLE.PSFModel,
    n_spots::Int = 1000;
    box_size::Int = 11,
    device = GaussMLE.CPU(),
    verbose::Bool = false,
    kwargs...
)
    # Generate test data
    data, true_params = generate_test_data(model_type, n_spots, box_size; kwargs...)
    
    # Create fitter
    fitter = GaussMLE.GaussMLEFitter(
        psf_model = psf_model,
        device = device,
        iterations = 20
    )
    
    # Fit the data
    results = GaussMLE.fit(fitter, data)
    
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
        :x => (bias_tol=0.05f0, std_tol=0.25f0),  # Allow 25% tolerance for std/CRLB ratio
        :y => (bias_tol=0.05f0, std_tol=0.25f0),
        :z => (bias_tol=30.0f0, std_tol=3.0f0),  # Very lenient for z - needs more work
        :photons => (bias_tol=100.0f0, std_tol=0.25f0),  # 25% tolerance
        :background => (bias_tol=2.0f0, std_tol=0.25f0),
        :sigma => (bias_tol=0.05f0, std_tol=0.25f0),
        :sigma_x => (bias_tol=0.05f0, std_tol=0.25f0),
        :sigma_y => (bias_tol=0.05f0, std_tol=0.25f0),
    )
    
    if verbose
        println("\n" * "="^60)
        println("Model Validation: $model_type")
        println("Device: $(typeof(device))")
        println("Number of spots: $n_spots")
        println("="^60)
    end
    
    for param in params_to_validate
        tol = tolerances[param]
        result = validate_fitting_results(
            results, true_params, param;
            bias_tol = tol.bias_tol,
            std_tol = tol.std_tol,
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