using GaussMLE
using Statistics
using Printf

# Test with bg=10 and 100 iterations
println("Testing with Background = 10, Iterations = 100")
println("="^60)

# Parameters
n_samples = 10000
roi_size = 7
nominal_photons = 1000.0f0
nominal_bg = 10.0f0
iterations = 100  # Increased from default 20

println("Parameters:")
println("  N samples: $n_samples")
println("  ROI size: $(roi_size)×$(roi_size)")
println("  True photons: $nominal_photons")
println("  True background: $nominal_bg")
println("  Iterations: $iterations")
println()

# Setup
psf_model = GaussMLE.GaussianXYNB(1.3f0)
camera_model = GaussMLE.IdealCamera()

# Generate data
spots = GaussMLE.generate_random_spots(
    n_samples, 
    roi_size;
    x_mean = 4.0f0,
    y_mean = 4.0f0,
    x_std = 0.1f0,
    y_std = 0.1f0,
    photons_mean = nominal_photons,
    photons_std = 0.0f0,
    background_mean = nominal_bg,
    background_std = 0.0f0,
    model_type = :xynb,
    seed = 42
)

data, true_positions = GaussMLE.generate_spots_data(
    psf_model,
    spots,
    roi_size;
    camera_model = camera_model,
    seed = 42
)

# Create fitter with more iterations
fitter = GaussMLE.GaussMLEFitter(
    psf_model = psf_model,
    camera_model = camera_model,
    device = :cpu,
    iterations = iterations,  # 100 iterations
    batch_size = 10_000
)

# Fit
println("Fitting $n_samples spots with $iterations iterations...")
t_start = time()
results = GaussMLE.fit(fitter, data)
t_elapsed = time() - t_start
println(@sprintf("Time elapsed: %.2f seconds", t_elapsed))

# Extract results
params = results.parameters
uncertainties = results.uncertainties

# Analyze all parameters
param_names = ["X", "Y", "Photons", "Background"]
println("\nResults Summary (100 iterations):")
println("-"^60)
println(@sprintf("%-12s %12s %12s %12s %12s", "Parameter", "Bias", "Std Dev", "Mean CRLB", "Std/CRLB"))
println("-"^60)

for (i, name) in enumerate(param_names)
    fitted = params[i, :]
    uncertainty = uncertainties[i, :]
    
    if i <= 2  # X, Y
        true_vals = (i == 1) ? true_positions[:x] : true_positions[:y]
    elseif i == 3  # Photons
        true_vals = true_positions[:photons]
    else  # Background
        true_vals = true_positions[:background]
    end
    
    errors = fitted .- true_vals
    bias = mean(errors)
    std_dev = std(errors)
    mean_crlb = mean(uncertainty)
    ratio = std_dev / mean_crlb
    
    println(@sprintf("%-12s %12.4f %12.4f %12.4f %12.3f", 
                    name, bias, std_dev, mean_crlb, ratio))
end

println("="^60)

# Now test with standard 20 iterations for comparison
fitter_20 = GaussMLE.GaussMLEFitter(
    psf_model = psf_model,
    camera_model = camera_model,
    device = :cpu,
    iterations = 20,  # Standard iterations
    batch_size = 10_000
)

println("\nFitting same data with 20 iterations for comparison...")
t_start = time()
results_20 = GaussMLE.fit(fitter_20, data)
t_elapsed_20 = time() - t_start
println(@sprintf("Time elapsed: %.2f seconds", t_elapsed_20))

params_20 = results_20.parameters
uncertainties_20 = results_20.uncertainties

println("\nResults Summary (20 iterations):")
println("-"^60)
println(@sprintf("%-12s %12s %12s %12s %12s", "Parameter", "Bias", "Std Dev", "Mean CRLB", "Std/CRLB"))
println("-"^60)

for (i, name) in enumerate(param_names)
    fitted = params_20[i, :]
    uncertainty = uncertainties_20[i, :]
    
    if i <= 2  # X, Y
        true_vals = (i == 1) ? true_positions[:x] : true_positions[:y]
    elseif i == 3  # Photons
        true_vals = true_positions[:photons]
    else  # Background
        true_vals = true_positions[:background]
    end
    
    errors = fitted .- true_vals
    bias = mean(errors)
    std_dev = std(errors)
    mean_crlb = mean(uncertainty)
    ratio = std_dev / mean_crlb
    
    println(@sprintf("%-12s %12.4f %12.4f %12.4f %12.3f", 
                    name, bias, std_dev, mean_crlb, ratio))
end

println("="^60)

# Compare the two
println("\nComparison:")
println("-"^60)

fitted_N_100 = params[3, :]
fitted_N_20 = params_20[3, :]
crlb_N_100 = uncertainties[3, :]
crlb_N_20 = uncertainties_20[3, :]

errors_N_100 = fitted_N_100 .- true_positions[:photons]
errors_N_20 = fitted_N_20 .- true_positions[:photons]

println("Photons (N):")
println(@sprintf("  20 iter:  Bias=%.2f, Std=%.2f, CRLB=%.2f, Std/CRLB=%.3f", 
                mean(errors_N_20), std(errors_N_20), mean(crlb_N_20), 
                std(errors_N_20)/mean(crlb_N_20)))
println(@sprintf("  100 iter: Bias=%.2f, Std=%.2f, CRLB=%.2f, Std/CRLB=%.3f", 
                mean(errors_N_100), std(errors_N_100), mean(crlb_N_100), 
                std(errors_N_100)/mean(crlb_N_100)))

# Check convergence by looking at log-likelihood
ll_100 = results.log_likelihoods
ll_20 = results_20.log_likelihoods
println("\nLog-Likelihood Statistics:")
println(@sprintf("  20 iter:  Mean=%.2f, Std=%.2f", mean(ll_20), std(ll_20)))
println(@sprintf("  100 iter: Mean=%.2f, Std=%.2f", mean(ll_100), std(ll_100)))
println(@sprintf("  Improvement: %.2f", mean(ll_100) - mean(ll_20)))

# Check how many fits changed significantly
significant_changes = sum(abs.(fitted_N_100 .- fitted_N_20) .> 1.0)
println(@sprintf("\nFits with >1 photon change: %d (%.1f%%)", 
                significant_changes, 100*significant_changes/n_samples))

# Speed comparison
println("\nSpeed:")
println(@sprintf("  20 iter:  %.0f fits/sec", n_samples/t_elapsed_20))
println(@sprintf("  100 iter: %.0f fits/sec", n_samples/t_elapsed))
println(@sprintf("  Slowdown: %.1fx", t_elapsed/t_elapsed_20))