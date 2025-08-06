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
    
    for k in 1:n_samples
        x = true_x + 0.1f0 * randn(Float32)
        y = true_y + 0.1f0 * randn(Float32)
        
        # Adjust sigma for asymmetric models
        sigma_x = true_sigma
        sigma_y = true_sigma
        if psf_model isa GaussMLE.GaussianXYNBSXSY
            sigma_x *= 1.1f0  # Make slightly asymmetric for testing
            sigma_y *= 0.9f0
        end
        
        for j in 1:roi_size, i in 1:roi_size
            dx = Float32(i) - x
            dy = Float32(j) - y
            gaussian = true_photons * exp(-(dx^2/(2*sigma_x^2) + dy^2/(2*sigma_y^2)))
            expected = true_bg + gaussian / (2π * sigma_x * sigma_y)
            
            # Poisson noise
            data[i, j, k] = expected > 0 ? rand(Poisson(expected)) : 0
        end
    end
    
    return data
end

function run_single_benchmark(psf_model, device, camera_type)
    # Generate data
    data = generate_data(n_samples, roi_size, psf_model)
    
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
        
        # Parameter names and true values
        param_info = if psf_model isa GaussMLE.GaussianXYNB
            [(:x, true_x), (:y, true_y), (:photons, true_photons), (:background, true_bg)]
        elseif psf_model isa GaussMLE.GaussianXYNBS
            [(:x, true_x), (:y, true_y), (:photons, true_photons), (:background, true_bg), (:sigma, true_sigma)]
        elseif psf_model isa GaussMLE.GaussianXYNBSXSY
            [(:x, true_x), (:y, true_y), (:photons, true_photons), (:background, true_bg), 
             (:sigma_x, true_sigma * 1.1f0), (:sigma_y, true_sigma * 0.9f0)]
        elseif psf_model isa GaussMLE.AstigmaticXYZNB
            [(:x, true_x), (:y, true_y), (:photons, true_photons), (:background, true_bg), (:z, 0.0f0)]
        end
        
        # Calculate statistics
        param_stats = Dict{Symbol, NamedTuple}()
        for (i, (name, true_val)) in enumerate(param_info)
            fitted = params[i, :]
            uncertainty = uncertainties[i, :]
            
            bias = mean(fitted) - true_val
            std_dev = std(fitted)
            mean_crlb = mean(uncertainty)
            
            param_stats[name] = (bias=bias, std=std_dev, crlb=mean_crlb)
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