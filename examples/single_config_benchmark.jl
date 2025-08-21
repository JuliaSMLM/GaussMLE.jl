using Pkg
Pkg.activate("examples")

using GaussMLE
using Random
using Statistics
using Printf
using Distributions

# Example parameters (users can adjust these)
psf_model = GaussMLE.GaussianXYNB(1.3f0)      # Can also use GaussianXYNBS(), AstigmaticXYZNB(), etc.
device = :cpu                                  # :cpu, :gpu, or :auto
camera_type = :ideal                          # :ideal or :scmos
roi_size = 7
n_samples = 10_000
verbose = true

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)  # Create if it doesn't exist

# Ground truth parameters
nominal_x = 4.0f0
nominal_y = 4.0f0  
nominal_photons = 1000.0f0
nominal_bg = 5.0f0  # Optimal background level for unbiased estimation

println("=== Example: Single Configuration Benchmark ===")
println("This example demonstrates benchmarking GaussMLE fitting performance")
println("Parameters: n_samples=$n_samples, device=$device, camera=$camera_type")
println()

function generate_data_with_truth(n_samples, roi_size)
    # Generate random blobs using the simulator
    blobs = GaussMLE.generate_random_blobs(
        n_samples, 
        roi_size;
        x_mean = nominal_x,
        y_mean = nominal_y,
        x_std = 0.1f0,
        y_std = 0.1f0,
        photons_mean = nominal_photons,
        photons_std = 0.0f0,  # No variation for benchmark
        background_mean = nominal_bg,
        background_std = 0.0f0,  # No variation for benchmark
        model_type = :xynb,
        seed = 42
    )
    
    # Generate data using the simulator with integrated Gaussians
    data, true_positions = GaussMLE.generate_blobs_data(
        psf_model,
        blobs,
        roi_size;
        camera_model = camera_model,
        seed = 42
    )
    
    return data, true_positions
end

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

# Generate synthetic data with ground truth tracking
verbose && println("Generating $(n_samples) ROIs...")
data, true_positions = generate_data_with_truth(n_samples, roi_size)

# Create fitter object
verbose && println("Creating fitter with $(typeof(psf_model))...")
fitter = GaussMLE.GaussMLEFitter(
    psf_model = psf_model,
    camera_model = camera_model,
    device = device,
    iterations = 20,
    batch_size = 10_000
)

# Run fitting
verbose && println("Fitting with $(device) device...")
t_start = time()

results = GaussMLE.fit(fitter, data; variance_map=variance_map)

t_elapsed = time() - t_start
fits_per_second = n_samples / t_elapsed

# Extract results
params = results.parameters
uncertainties = results.uncertainties

# Parameter names based on PSF model
param_names = if psf_model isa GaussMLE.GaussianXYNB
    [:x, :y, :photons, :background]
elseif psf_model isa GaussMLE.GaussianXYNBS
    [:x, :y, :photons, :background, :sigma]
elseif psf_model isa GaussMLE.AstigmaticXYZNB
    [:x, :y, :photons, :background, :z]
else
    [:x, :y, :photons, :background, :sigma_x, :sigma_y]
end

# Build results table
results_table = String[]
push!(results_table, "="^80)
push!(results_table, "BENCHMARK RESULTS")
push!(results_table, "="^80)
push!(results_table, "Configuration:")
push!(results_table, "  PSF Model:  $(typeof(psf_model))")
push!(results_table, "  Device:     $(device)")
push!(results_table, "  Camera:     $(camera_type)")
push!(results_table, "  ROI Size:   $(roi_size)×$(roi_size)")
push!(results_table, "  Samples:    $(n_samples)")
push!(results_table, "")
push!(results_table, "Performance:")
push!(results_table, "  Time:       $(round(t_elapsed, digits=2)) seconds")
push!(results_table, "  Speed:      $(round(Int, fits_per_second)) fits/second")
push!(results_table, "")
push!(results_table, "-"^80)
push!(results_table, @sprintf("%-12s %12s %12s %12s %12s", "Parameter", "Bias", "Std Dev", "Mean CRLB", "Std/CRLB"))
push!(results_table, "-"^80)

for (i, name) in enumerate(param_names)
    if haskey(true_positions, name)
        fitted = params[i, :]
        uncertainty = uncertainties[i, :]
        true_vals = true_positions[name]
        
        # Calculate bias and std using individual true values
        errors = fitted .- true_vals
        bias = mean(errors)
        std_dev = std(errors)
        mean_crlb = mean(uncertainty)
        ratio = std_dev / mean_crlb
        
        push!(results_table, @sprintf("%-12s %12.4f %12.4f %12.4f %12.2f", 
                        name, bias, std_dev, mean_crlb, ratio))
    elseif name == :sigma && psf_model isa GaussMLE.GaussianXYNBS
        # For sigma parameter when fitting
        fitted = params[i, :]
        uncertainty = uncertainties[i, :]
        
        bias = mean(fitted) - 1.3f0
        std_dev = std(fitted)
        mean_crlb = mean(uncertainty)
        ratio = std_dev / mean_crlb
        
        push!(results_table, @sprintf("%-12s %12.4f %12.4f %12.4f %12.2f", 
                        name, bias, std_dev, mean_crlb, ratio))
    end
end
push!(results_table, "="^80)

# Add note about expected ratios
push!(results_table, "")
push!(results_table, "Note: Std/CRLB ratio should be ~1.0 for optimal estimators.")
push!(results_table, "      Bias should be ~CRLB/√n_samples = ~$(round(0.056/sqrt(n_samples), digits=4)) for positions")

# Print results to console
for line in results_table
    println(line)
end

# Save results to file
output_file = joinpath(output_dir, "benchmark_results.txt")
open(output_file, "w") do io
    for line in results_table
        println(io, line)
    end
end
verbose && println("\nResults saved to $output_file")