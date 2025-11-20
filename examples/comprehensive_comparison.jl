using Pkg
Pkg.activate(".")

using GaussMLE
using SMLMData
using Random
using Statistics
using Printf

# Example parameters (users can adjust these)
roi_size = 11
n_samples = 500  # Moderate size for comprehensive testing
verbose = true
output_format = "txt"

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)  # Create if it doesn't exist

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
    convergence_rate::Float64
end

function run_single_benchmark(psf_model, device, camera_type)
    try
        # Create camera
        camera = if camera_type == :ideal
            SMLMData.IdealCamera(512, 512, 0.1)
        else  # :scmos
            # 5.0 e⁻ rms readnoise (variance = 25 e⁻²)
            readnoise = 5.0f0
            SMLMData.SCMOSCamera(
                256, 256, 0.1f0, readnoise,
                offset = 100.0f0,
                gain = 0.5f0,
                qe = 0.82f0
            )
        end
        
        # Generate known true parameters with some variation
        Random.seed!(42)
        true_params = Float32[
            6.0 .+ 0.5f0 * randn(Float32, n_samples)';  # x
            6.0 .+ 0.5f0 * randn(Float32, n_samples)';  # y
            1000.0 .+ 200.0f0 * randn(Float32, n_samples)';  # photons
            10.0 .+ 2.0f0 * randn(Float32, n_samples)'   # background
        ]
        
        # Add model-specific parameters
        if psf_model isa GaussMLE.GaussianXYNBS
            sigma_vals = 1.3f0 .+ 0.2f0 * randn(Float32, n_samples)'
            true_params = vcat(true_params, sigma_vals)
        elseif psf_model isa GaussMLE.GaussianXYNBSXSY
            sigma_x_vals = 1.3f0 .+ 0.15f0 * randn(Float32, n_samples)'
            sigma_y_vals = 1.3f0 .+ 0.15f0 * randn(Float32, n_samples)'
            true_params = vcat(true_params, sigma_x_vals, sigma_y_vals)
        elseif psf_model isa GaussMLE.AstigmaticXYZNB
            z_vals = 200.0f0 * randn(Float32, n_samples)'
            # Insert z after y (x, y, z, photons, background)
            true_params = vcat(true_params[1:2, :], z_vals, true_params[3:4, :])
        end
        
        # Generate ROI batch using new simulator
        batch = GaussMLE.generate_roi_batch(camera, psf_model; 
                                           n_rois=n_samples,
                                           roi_size=roi_size,
                                           true_params=true_params,
                                           seed=42)
        
        # Create fitter
        device_obj = device == :cpu ? GaussMLE.CPU() : GaussMLE.GPU()
        fitter = GaussMLE.GaussMLEFitter(
            psf_model = psf_model,
            device = device_obj,
            iterations = 20
        )
        
        # Run fitting with timing
        t_start = time()
        results = GaussMLE.fit(fitter, batch)
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
            [:x, :y, :z, :photons, :background]
        end
        
        # Calculate statistics
        param_stats = Dict{Symbol, NamedTuple}()
        for (i, name) in enumerate(param_names)
            fitted = params[i, :]
            uncertainty = uncertainties[i, :]
            true_vals = true_params[i, :]
            
            # Calculate errors
            errors = fitted .- true_vals
            bias = mean(errors)
            std_dev = std(errors)  # Empirical uncertainty
            mean_crlb = mean(uncertainty)  # Theoretical uncertainty
            
            param_stats[name] = (bias=bias, std=std_dev, crlb=mean_crlb)
        end
        
        # Calculate convergence rate (no infinite uncertainties)
        n_converged = sum(isfinite.(uncertainties[1, :]))
        convergence_rate = n_converged / n_samples
        
        model_name = split(string(typeof(psf_model)), ".")[end]
        return BenchmarkResult(model_name, device, camera_type, n_samples/t_elapsed, param_stats, convergence_rate)
        
    catch e
        # Return nothing if benchmark fails
        verbose && println("    Failed: $e")
        return nothing
    end
end

# Define test configurations
psf_models = [
    GaussMLE.GaussianXYNB(0.13f0),
    GaussMLE.GaussianXYNBS(),
    GaussMLE.GaussianXYNBSXSY(),
    GaussMLE.AstigmaticXYZNB{Float32}(
        0.13f0, 0.13f0,  # σx₀, σy₀ (microns)
        0.0f0, 0.0f0,    # Ax, Ay
        0.0f0, 0.0f0,    # Bx, By
        250.0f0,         # γ
        400.0f0          # d
    )
]

devices = [:cpu]  # Add :gpu if available
cameras = [:ideal, :scmos]

results = BenchmarkResult[]

println("Running comprehensive comparison...")
println("Models: ", [split(string(typeof(p)), ".")[end] for p in psf_models])
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
                println("✓ ($(round(Int, result.fits_per_second)) fits/s, $(round(result.convergence_rate*100, digits=1))% converged)")
            else
                println("✗ (failed)")
            end
        end
    end
end

# Build output table
output_lines = String[]

# Summary table
push!(output_lines, "\n" * "="^130)
push!(output_lines, "COMPREHENSIVE COMPARISON RESULTS")
push!(output_lines, "="^130)
push!(output_lines, @sprintf("%-20s %-7s %-6s %10s %8s | %-20s %-20s %-20s", 
                "Model", "Device", "Camera", "Fits/s", "Conv%", 
                "X (bias/std/crlb)", "Y (bias/std/crlb)", "Photons (bias/std/crlb)"))
push!(output_lines, "-"^130)

for r in results
    x_stats = get(r.param_stats, :x, (bias=NaN32, std=NaN32, crlb=NaN32))
    y_stats = get(r.param_stats, :y, (bias=NaN32, std=NaN32, crlb=NaN32))
    n_stats = get(r.param_stats, :photons, (bias=NaN32, std=NaN32, crlb=NaN32))
    
    x_str = @sprintf("%5.3f/%5.3f/%5.3f", x_stats.bias, x_stats.std, x_stats.crlb)
    y_str = @sprintf("%5.3f/%5.3f/%5.3f", y_stats.bias, y_stats.std, y_stats.crlb)
    n_str = @sprintf("%5.1f/%5.1f/%5.1f", n_stats.bias, n_stats.std, n_stats.crlb)
    
    push!(output_lines, @sprintf("%-20s %-7s %-6s %10d %7.1f%% | %-20s %-20s %-20s",
                    r.psf_model, r.device, r.camera, round(Int, r.fits_per_second),
                    r.convergence_rate*100, x_str, y_str, n_str))
end

push!(output_lines, "="^130)

# Additional parameter table for models with extra parameters
push!(output_lines, "\nAdditional Parameters:")
push!(output_lines, "-"^100)
push!(output_lines, @sprintf("%-20s %-7s %-6s | %-12s %12s %12s %12s %12s", 
                    "Model", "Device", "Camera", "Parameter", "Bias", "Emp.STD", "CRLB", "Ratio"))
push!(output_lines, "-"^100)

for r in results
    extra_params = setdiff(keys(r.param_stats), [:x, :y, :photons, :background])
    if !isempty(extra_params)
        for param in extra_params
            stats = r.param_stats[param]
            ratio = stats.std / stats.crlb
            push!(output_lines, @sprintf("%-20s %-7s %-6s | %-12s %12.4f %12.4f %12.4f %12.3f",
                            r.psf_model, r.device, r.camera, param, 
                            stats.bias, stats.std, stats.crlb, ratio))
        end
    end
end

push!(output_lines, "="^100)

# Analysis section
push!(output_lines, "\nPERFORMANCE ANALYSIS:")
push!(output_lines, "-"^50)

# Find best and worst performers
if !isempty(results)
    # Speed analysis
    speed_sorted = sort(results, by=r->r.fits_per_second, rev=true)
    push!(output_lines, @sprintf("Fastest: %s-%s-%s (%.0f fits/s)", 
                        speed_sorted[1].psf_model, speed_sorted[1].device, speed_sorted[1].camera,
                        speed_sorted[1].fits_per_second))
    push!(output_lines, @sprintf("Slowest: %s-%s-%s (%.0f fits/s)", 
                        speed_sorted[end].psf_model, speed_sorted[end].device, speed_sorted[end].camera,
                        speed_sorted[end].fits_per_second))
    
    # Accuracy analysis (X position bias)
    accuracy_sorted = sort(results, by=r->abs(get(r.param_stats, :x, (bias=Inf32,)).bias))
    push!(output_lines, @sprintf("Most accurate (X bias): %s-%s-%s (%.4f pixels)", 
                        accuracy_sorted[1].psf_model, accuracy_sorted[1].device, accuracy_sorted[1].camera,
                        get(accuracy_sorted[1].param_stats, :x, (bias=NaN32,)).bias))
    
    # Precision analysis (X position empirical std)
    precision_sorted = sort(results, by=r->get(r.param_stats, :x, (std=Inf32,)).std)
    push!(output_lines, @sprintf("Most precise (X std): %s-%s-%s (%.4f pixels)", 
                        precision_sorted[1].psf_model, precision_sorted[1].device, precision_sorted[1].camera,
                        get(precision_sorted[1].param_stats, :x, (std=NaN32,)).std))
    
    # CRLB matching analysis
    crlb_ratios = []
    for r in results
        x_stats = get(r.param_stats, :x, nothing)
        if x_stats !== nothing && isfinite(x_stats.std) && isfinite(x_stats.crlb) && x_stats.crlb > 0
            push!(crlb_ratios, (abs(1.0 - x_stats.std/x_stats.crlb), r))
        end
    end
    
    if !isempty(crlb_ratios)
        sort!(crlb_ratios)
        best_crlb = crlb_ratios[1][2]
        x_stats = get(best_crlb.param_stats, :x, (std=NaN32, crlb=NaN32))
        push!(output_lines, @sprintf("Best CRLB match (X): %s-%s-%s (ratio=%.3f)", 
                            best_crlb.psf_model, best_crlb.device, best_crlb.camera,
                            x_stats.std/x_stats.crlb))
    end
end

push!(output_lines, "="^50)

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

println("\nResults saved to $output_file")
println("✓ Comprehensive comparison complete!")