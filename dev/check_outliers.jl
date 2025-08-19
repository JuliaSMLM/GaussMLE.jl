using GaussMLE
using Statistics
using Printf

# Check for outliers in N estimates and CRLB
println("Checking for Outliers in Estimates")
println("="^60)

# Parameters
n_samples = 10000
roi_size = 7
nominal_photons = 1000.0f0
nominal_bg = 10.0f0

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

# Fit
fitter = GaussMLE.GaussMLEFitter(
    psf_model = psf_model,
    camera_model = camera_model,
    device = :cpu,
    iterations = 20,
    batch_size = 10_000
)

println("Fitting $n_samples spots...")
results = GaussMLE.fit(fitter, data)

# Extract photon estimates and uncertainties
fitted_N = results.parameters[3, :]
crlb_N = results.uncertainties[3, :]
true_N = true_positions[:photons]
errors_N = fitted_N .- true_N

# Basic statistics
println("\nBasic Statistics:")
println("-"^40)
println(@sprintf("Mean fitted N:      %.2f", mean(fitted_N)))
println(@sprintf("Median fitted N:    %.2f", median(fitted_N)))
println(@sprintf("Std of fitted N:    %.2f", std(fitted_N)))
println(@sprintf("Min fitted N:       %.2f", minimum(fitted_N)))
println(@sprintf("Max fitted N:       %.2f", maximum(fitted_N)))
println()
println(@sprintf("Mean CRLB:          %.2f", mean(crlb_N)))
println(@sprintf("Median CRLB:        %.2f", median(crlb_N)))
println(@sprintf("Std of CRLB:        %.2f", std(crlb_N)))
println(@sprintf("Min CRLB:           %.2f", minimum(crlb_N)))
println(@sprintf("Max CRLB:           %.2f", maximum(crlb_N)))

# Check for outliers using IQR method
function find_outliers(data, factor=1.5)
    q1 = quantile(data, 0.25)
    q3 = quantile(data, 0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    outliers = findall(x -> x < lower || x > upper, data)
    return outliers, lower, upper
end

# Find outliers in fitted N
outliers_N, lower_N, upper_N = find_outliers(fitted_N)
println("\nOutlier Analysis for Fitted N:")
println("-"^40)
println(@sprintf("IQR bounds: [%.2f, %.2f]", lower_N, upper_N))
println(@sprintf("Number of outliers: %d (%.2f%%)", length(outliers_N), 100*length(outliers_N)/n_samples))

if length(outliers_N) > 0
    println("Outlier values:")
    for i in 1:min(10, length(outliers_N))
        idx = outliers_N[i]
        println(@sprintf("  Spot %d: N=%.2f (true=%.2f, error=%.2f)", 
                        idx, fitted_N[idx], true_N[idx], errors_N[idx]))
    end
    if length(outliers_N) > 10
        println("  ... and $(length(outliers_N)-10) more")
    end
end

# Find outliers in CRLB
outliers_crlb, lower_crlb, upper_crlb = find_outliers(crlb_N)
println("\nOutlier Analysis for CRLB:")
println("-"^40)
println(@sprintf("IQR bounds: [%.2f, %.2f]", lower_crlb, upper_crlb))
println(@sprintf("Number of outliers: %d (%.2f%%)", length(outliers_crlb), 100*length(outliers_crlb)/n_samples))

if length(outliers_crlb) > 0
    println("Outlier CRLB values:")
    for i in 1:min(10, length(outliers_crlb))
        idx = outliers_crlb[i]
        println(@sprintf("  Spot %d: CRLB=%.2f, N=%.2f", 
                        idx, crlb_N[idx], fitted_N[idx]))
    end
    if length(outliers_crlb) > 10
        println("  ... and $(length(outliers_crlb)-10) more")
    end
end

# Calculate statistics without outliers
mask = trues(n_samples)
mask[outliers_N] .= false
mask[outliers_crlb] .= false
n_clean = sum(mask)

if n_clean > 0
    clean_N = fitted_N[mask]
    clean_crlb = crlb_N[mask]
    clean_true = true_N[mask]
    clean_errors = clean_N .- clean_true
    
    println("\nStatistics After Removing Outliers:")
    println("-"^40)
    println(@sprintf("Samples retained: %d (%.1f%%)", n_clean, 100*n_clean/n_samples))
    println(@sprintf("Mean fitted N:    %.2f", mean(clean_N)))
    println(@sprintf("Bias in N:        %.2f", mean(clean_errors)))
    println(@sprintf("Std of N:         %.2f", std(clean_errors)))
    println(@sprintf("Mean CRLB:        %.2f", mean(clean_crlb)))
    println(@sprintf("Std/CRLB ratio:   %.3f", std(clean_errors)/mean(clean_crlb)))
end

# Check for systematic patterns in outliers
println("\nChecking for Patterns in Outliers:")
println("-"^40)

# Check if outliers correlate with position
if length(outliers_N) > 0
    x_outliers = results.parameters[1, outliers_N]
    y_outliers = results.parameters[2, outliers_N]
    println(@sprintf("Mean X position of outliers: %.2f (overall mean: %.2f)", 
                    mean(x_outliers), mean(results.parameters[1, :])))
    println(@sprintf("Mean Y position of outliers: %.2f (overall mean: %.2f)", 
                    mean(y_outliers), mean(results.parameters[2, :])))
end

# Check distribution of errors
println("\nError Distribution Percentiles:")
println("-"^40)
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles
    println(@sprintf("  %3d%%: %.2f", p, quantile(errors_N, p/100)))
end

# Check if CRLB values are reasonable
println("\nCRLB Sanity Check:")
println("-"^40)
theoretical_crlb = sqrt(nominal_photons)
println(@sprintf("Theoretical √N:        %.2f", theoretical_crlb))
println(@sprintf("Mean CRLB:             %.2f", mean(crlb_N)))
println(@sprintf("Ratio (actual/theory): %.3f", mean(crlb_N)/theoretical_crlb))

# Check for infinite or NaN values
n_inf = sum(isinf.(crlb_N))
n_nan = sum(isnan.(crlb_N))
if n_inf > 0 || n_nan > 0
    println("\nWARNING: Found invalid CRLB values!")
    println(@sprintf("  Infinite: %d", n_inf))
    println(@sprintf("  NaN: %d", n_nan))
end