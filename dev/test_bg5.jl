using GaussMLE
using Statistics
using Printf

# Test with moderate background (bg = 5)
println("Testing with Background = 5")
println("="^60)

# Parameters
n_samples = 10000
roi_size = 7
nominal_photons = 1000.0f0
nominal_bg = 5.0f0  # Moderate background

println("Parameters:")
println("  N samples: $n_samples")
println("  ROI size: $(roi_size)×$(roi_size)")
println("  True photons: $nominal_photons")
println("  True background: $nominal_bg")
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

# Create fitter
fitter = GaussMLE.GaussMLEFitter(
    psf_model = psf_model,
    camera_model = camera_model,
    device = :cpu,
    iterations = 20,
    batch_size = 10_000
)

# Fit
println("Fitting $n_samples spots...")
results = GaussMLE.fit(fitter, data)

# Extract results
params = results.parameters
uncertainties = results.uncertainties

# Analyze all parameters
param_names = ["X", "Y", "Photons", "Background"]
println("\nResults Summary:")
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

# Detailed analysis for photons
fitted_N = params[3, :]
crlb_N = uncertainties[3, :]
true_N = true_positions[:photons]
errors_N = fitted_N .- true_N

println("\nDetailed Photon Analysis:")
println("-"^40)
println(@sprintf("Mean fitted N:      %.2f", mean(fitted_N)))
println(@sprintf("Median fitted N:    %.2f", median(fitted_N)))
println(@sprintf("True N:             %.2f", nominal_photons))
println(@sprintf("Bias:               %.2f", mean(errors_N)))
println(@sprintf("Std deviation:      %.2f", std(errors_N)))
println(@sprintf("Mean CRLB:          %.2f", mean(crlb_N)))
println(@sprintf("Std/CRLB ratio:     %.3f", std(errors_N)/mean(crlb_N)))

# Compare with different background levels
println("\nComparison with Previous Results:")
println("-"^40)
println("bg=0.01: Bias=-5.62,  Std/CRLB=0.946")
println("bg=5.00: Bias=$(Printf.@sprintf("%.2f", mean(errors_N))), Std/CRLB=$(Printf.@sprintf("%.3f", std(errors_N)/mean(crlb_N)))")
println("bg=10.0: Bias=+1.62,  Std/CRLB=1.21")

# Check error distribution
println("\nError Distribution Percentiles:")
println("-"^40)
percentiles = [1, 5, 25, 50, 75, 95, 99]
for p in percentiles
    println(@sprintf("%3d%%: %8.2f", p, quantile(errors_N, p/100)))
end

# Background analysis
fitted_bg = params[4, :]
crlb_bg = uncertainties[4, :]
errors_bg = fitted_bg .- true_positions[:background]

println("\nDetailed Background Analysis:")
println("-"^40)
println(@sprintf("Mean fitted bg:     %.3f", mean(fitted_bg)))
println(@sprintf("True bg:            %.3f", nominal_bg))
println(@sprintf("Bias:               %.3f", mean(errors_bg)))
println(@sprintf("Std deviation:      %.3f", std(errors_bg)))
println(@sprintf("Mean CRLB:          %.3f", mean(crlb_bg)))
println(@sprintf("Std/CRLB ratio:     %.3f", std(errors_bg)/mean(crlb_bg)))

# Check correlation
correlation = cor(errors_N, errors_bg)
println(@sprintf("\nCorrelation(N,bg):  %.3f", correlation))

# Theoretical CRLB
theoretical_crlb_N = sqrt(nominal_photons)
println("\nTheoretical Analysis:")
println("-"^40)
println(@sprintf("Theoretical √N:           %.2f", theoretical_crlb_N))
println(@sprintf("Actual mean CRLB:         %.2f", mean(crlb_N)))
println(@sprintf("Ratio (actual/theory):    %.3f", mean(crlb_N)/theoretical_crlb_N))

# Check for outliers
function count_outliers(data, factor=3.0)
    m = median(data)
    mad = median(abs.(data .- m))
    threshold = factor * mad * 1.4826  # MAD to std conversion
    return sum(abs.(data .- m) .> threshold)
end

n_outliers = count_outliers(errors_N)
println(@sprintf("\nOutliers (>3σ):           %d (%.2f%%)", n_outliers, 100*n_outliers/n_samples))