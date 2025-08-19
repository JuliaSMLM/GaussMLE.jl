using Pkg
Pkg.activate("examples")

using GaussMLE
using Random
using Statistics
using Printf
using Distributions

# Example parameters (users can adjust these)
roi_size = 7
n_samples = 1_000  # Smaller for comprehensive testing
true_sigma = 1.3f0
verbose = true
output_format = "txt"

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)  # Create if it doesn't exist

# Fixed ground truth parameters for all tests
true_x = 4.0f0
true_y = 4.0f0
true_photons = 1000.0f0
true_bg = 10.0f0

println("=== Example: Comprehensive Model Comparison ===")
println("This example demonstrates comparing all PSF models and configurations")
println("Parameters: n_samples=$n_samples, roi_size=$roi_size, verbose=$verbose")
println()

struct BenchmarkResult
    psf_model::String
    device::Symbol
    camera::Symbol
    fits_per_second::Float64
    param_stats::Dict{Symbol, NamedTuple{(:bias, :std, :crlb), Tuple{Float32, Float32, Float32}}}
end

function generate_data(n_samples, roi_size, psf_model)
    Random.seed!(42)
    data = zeros(Float32, roi_size, roi_size, n_samples)
    true_params = Dict{Symbol, Vector{Float32}}()
    
    for k in 1:n_samples
        # Generate with small variations to test precision
        x = true_x + 0.5f0 * randn(Float32)  # Match validation test variation
        y = true_y + 0.5f0 * randn(Float32)
        photons = true_photons * (0.8f0 + 0.4f0 * rand(Float32))  # ±20% variation
        bg = true_bg * (0.8f0 + 0.4f0 * rand(Float32))
        
        # Store actual true values for each spot
        push!(get!(true_params, :x, Float32[]), x)
        push!(get!(true_params, :y, Float32[]), y)
        push!(get!(true_params, :photons, Float32[]), photons)
        push!(get!(true_params, :background, Float32[]), bg)
        
        # Adjust sigma for different models
        sigma_x = true_sigma
        sigma_y = true_sigma
        if psf_model isa GaussMLE.GaussianXYNBS
            sigma_actual = true_sigma * (0.8f0 + 0.4f0 * rand(Float32))
            sigma_x = sigma_y = sigma_actual
            push!(get!(true_params, :sigma, Float32[]), sigma_actual)
        elseif psf_model isa GaussMLE.GaussianXYNBSXSY
            sigma_x = true_sigma * (0.8f0 + 0.4f0 * rand(Float32))
            sigma_y = true_sigma * (0.8f0 + 0.4f0 * rand(Float32))
            push!(get!(true_params, :sigma_x, Float32[]), sigma_x)
            push!(get!(true_params, :sigma_y, Float32[]), sigma_y)
        elseif psf_model isa GaussMLE.AstigmaticXYZNB
            z = 200.0f0 * randn(Float32)  # Z variation
            push!(get!(true_params, :z, Float32[]), z)
            # Compute PSF widths based on z (matching validation test calibration)
            z_norm = z / 500.0f0
            alpha_x = 1.0f0 + z_norm^2 + 0.5f0 * z_norm^3 + 0.1f0 * z_norm^4
            alpha_y = 1.0f0 + z_norm^2 - 0.5f0 * z_norm^3 - 0.1f0 * z_norm^4
            sigma_x = true_sigma * sqrt(alpha_x)
            sigma_y = true_sigma * sqrt(alpha_y)
        end
        
        for j in 1:roi_size, i in 1:roi_size
            # Use the SAME integrated Gaussian model that the fitter uses
            psf_x = GaussMLE.GaussLib.integral_gaussian_1d(i, x, sigma_x)
            psf_y = GaussMLE.GaussLib.integral_gaussian_1d(j, y, sigma_y)
            expected = bg + photons * psf_x * psf_y
            
            # Poisson noise
            data[i, j, k] = expected > 0 ? rand(Poisson(expected)) : 0
        end
    end
    
    return data, true_params
end

function run_single_benchmark(psf_model, device, camera_type)
    # Generate data with true parameters
    data, true_params = generate_data(n_samples, roi_size, psf_model)
    
    # Setup variance map for sCMOS
    variance_map = if camera_type == :scmos
        2.0f0 * ones(Float32, roi_size, roi_size)
    else
        nothing
    end
    
    # Setup camera model
    camera_model = if camera_type == :ideal
        GaussMLE.IdealCamera()
    else
        GaussMLE.SCMOSCamera(variance_map)
    end
    
    # Create fitter
    fitter = GaussMLE.GaussMLEFitter(
        psf_model = psf_model,
        camera_model = camera_model,
        device = device,
        iterations = 20,
        batch_size = n_samples  # Process all at once for consistency
    )
    
    # Run fitting
    t_start = time()
    
    try
        results = GaussMLE.fit(fitter, data; variance_map=variance_map)
        
        t_elapsed = time() - t_start
        
        # Extract and analyze results
        params = results.parameters
        uncertainties = results.uncertainties
        
        # Parameter names based on model
        param_names = if psf_model isa GaussMLE.GaussianXYNB
            [:x, :y, :photons, :background]
        elseif psf_model isa GaussMLE.GaussianXYNBS
            [:x, :y, :photons, :background, :sigma]
        elseif psf_model isa GaussMLE.GaussianXYNBSXSY
            [:x, :y, :photons, :background, :sigma_x, :sigma_y]
        elseif psf_model isa GaussMLE.AstigmaticXYZNB
            [:x, :y, :photons, :background, :z]
        end
        
        # Calculate statistics using actual true values
        param_stats = Dict{Symbol, NamedTuple}()
        for (i, name) in enumerate(param_names)
            fitted = params[i, :]
            uncertainty = uncertainties[i, :]
            true_vals = get(true_params, name, nothing)
            
            if true_vals !== nothing
                # Calculate errors relative to actual true values
                errors = fitted .- true_vals
                bias = mean(errors)
                std_dev = std(errors)  # This is the correct empirical uncertainty
                mean_crlb = mean(uncertainty)
                
                param_stats[name] = (bias=bias, std=std_dev, crlb=mean_crlb)
            else
                # Fallback for parameters not in true_params (shouldn't happen)
                param_stats[name] = (bias=0.0f0, std=0.0f0, crlb=mean(uncertainty))
            end
        end
        
        model_name = split(string(typeof(psf_model)), ".")[end]
        return BenchmarkResult(model_name, device, camera_type, n_samples/t_elapsed, param_stats)
        
    catch e
        # Return nothing if benchmark fails
        verbose && println("    Failed: $e")
        return nothing
    end
end

# Define test configurations
psf_models = [
    GaussMLE.GaussianXYNB(true_sigma),
    GaussMLE.GaussianXYNBS(),
    GaussMLE.GaussianXYNBSXSY(),
    GaussMLE.AstigmaticXYZNB{Float32}(true_sigma, true_sigma, 0.5f0, -0.5f0, 0.1f0, -0.1f0, 0.0f0, 500.0f0)
]

devices = [:cpu]  # Add :gpu if available
cameras = [:ideal, :scmos]

results = BenchmarkResult[]

println("Running comprehensive comparison...")
println("Models: ", length(psf_models))
println("Devices: ", join(devices, ", "))
println("Cameras: ", join(cameras, ", "))
println("\nProgress:")

total_configs = length(psf_models) * length(devices) * length(cameras)
current = 0

for psf_model in psf_models
    for device in devices
        for camera in cameras
            global current += 1
            model_name = split(string(typeof(psf_model)), ".")[end]
            print("  [$current/$total_configs] $model_name-$device-$camera... ")
            
            result = run_single_benchmark(psf_model, device, camera)
            if result !== nothing
                push!(results, result)
                println("✓ ($(round(Int, result.fits_per_second)) fits/s)")
            else
                println("✗ (skipped)")
            end
        end
    end
end

# Build output table
output_lines = String[]

# Summary table
push!(output_lines, "\n" * "="^120)
push!(output_lines, "COMPREHENSIVE COMPARISON RESULTS")
push!(output_lines, "="^120)
push!(output_lines, @sprintf("%-25s %-7s %-6s %10s | %-20s %-20s %-20s", 
                "Model", "Device", "Camera", "Fits/s", 
                "X (bias/std/crlb)", "Y (bias/std/crlb)", "Photons (bias/std/crlb)"))
push!(output_lines, "-"^120)

for r in results
    x_stats = get(r.param_stats, :x, (bias=NaN32, std=NaN32, crlb=NaN32))
    y_stats = get(r.param_stats, :y, (bias=NaN32, std=NaN32, crlb=NaN32))
    n_stats = get(r.param_stats, :photons, (bias=NaN32, std=NaN32, crlb=NaN32))
    
    x_str = @sprintf("%5.3f/%5.3f/%5.3f", x_stats.bias, x_stats.std, x_stats.crlb)
    y_str = @sprintf("%5.3f/%5.3f/%5.3f", y_stats.bias, y_stats.std, y_stats.crlb)
    n_str = @sprintf("%5.1f/%5.1f/%5.1f", n_stats.bias, n_stats.std, n_stats.crlb)
    
    push!(output_lines, @sprintf("%-25s %-7s %-6s %10d | %-20s %-20s %-20s",
                    r.psf_model, r.device, r.camera, round(Int, r.fits_per_second),
                    x_str, y_str, n_str))
end

push!(output_lines, "="^120)

# Additional parameter table for models with extra parameters
push!(output_lines, "\nAdditional Parameters:")
push!(output_lines, "-"^80)

for r in results
    extra_params = setdiff(keys(r.param_stats), [:x, :y, :photons, :background])
    if !isempty(extra_params)
        for param in extra_params
            stats = r.param_stats[param]
            push!(output_lines, @sprintf("%-25s %-7s %-6s | %-12s: bias=%6.3f  std=%6.3f  crlb=%6.3f",
                            r.psf_model, r.device, r.camera, param, stats.bias, stats.std, stats.crlb))
        end
    end
end

push!(output_lines, "="^80)

# Print to console
for line in output_lines
    println(line)
end

# Save results to file
output_file = joinpath(output_dir, "comprehensive_comparison.$output_format")
open(output_file, "w") do io
    for line in output_lines
        println(io, line)
    end
end
verbose && println("\nResults saved to $output_file")