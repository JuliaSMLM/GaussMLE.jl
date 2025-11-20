"""
sCMOS Camera Example
Demonstrates fitting with sCMOS camera noise model
Shows how pixel-dependent variance improves localization
"""

using Pkg
Pkg.activate(@__DIR__)  # Activate examples environment

using GaussMLE
using Statistics
using Random

Random.seed!(42)  # Reproducible results

println("=== sCMOS Camera Example ===\n")

# Generate synthetic data with sCMOS-like noise
# (pixel-dependent readout noise)
println("Generating synthetic sCMOS data...")
roi_size = 11
n_rois = 100

# Create data with spatially varying noise
# (simulates sCMOS pixel-dependent readout noise)
data = zeros(Float32, roi_size, roi_size, n_rois)
variance_map = zeros(Float32, roi_size, roi_size)

# Create realistic variance pattern
# (sCMOS sensors have pixel-dependent readout noise)
for i in 1:roi_size, j in 1:roi_size
    # Variance increases toward edges (typical sCMOS pattern)
    dist_from_center = sqrt((i - 6)^2 + (j - 6)^2)
    variance_map[i, j] = 1.0f0 + 0.5f0 * dist_from_center
end

# Generate data with varying noise
for k in 1:n_rois
    for i in 1:roi_size, j in 1:roi_size
        # Signal: Gaussian blob
        signal = 100.0f0 * exp(-((i-5.5f0)^2 + (j-5.5f0)^2)/(2*1.3f0^2))
        background = 10.0f0

        # Noise: Poisson + pixel-dependent readout
        mean_val = signal + background
        noise_std = sqrt(mean_val + variance_map[i, j])

        data[i, j, k] = mean_val + randn(Float32) * noise_std
    end
end

println("  Generated $n_rois ROIs with sCMOS-like noise")
println("  Variance map range: $(round(minimum(variance_map), digits=2)) - $(round(maximum(variance_map), digits=2))")

# Fit WITHOUT sCMOS model (ignores pixel-dependent variance)
println("\n--- Fit 1: Ideal Camera (uniform noise) ---")
fitter_ideal = GaussMLEFitter(
    psf_model = GaussianXYNB(1.3f0),
    iterations = 25
)
smld_ideal = fit(fitter_ideal, data)
x_ideal = [e.x for e in smld_ideal.emitters]
σ_ideal = [e.σ_x for e in smld_ideal.emitters]
println("  Mean x: $(round(mean(x_ideal), digits=2)) μm")
println("  Mean precision: $(round(mean(σ_ideal)*1000, digits=1)) nm")

# Fit WITH sCMOS model (accounts for pixel-dependent variance)
println("\n--- Fit 2: sCMOS Camera (pixel-dependent noise) ---")
fitter_scmos = GaussMLEFitter(
    psf_model = GaussianXYNB(1.3f0),
    iterations = 25
)

# Fit with variance map
smld_scmos = fit(fitter_scmos, data, variance_map = variance_map)
x_scmos = [e.x for e in smld_scmos.emitters]
σ_scmos = [e.σ_x for e in smld_scmos.emitters]
println("  Mean x: $(round(mean(x_scmos), digits=2)) μm")
println("  Mean precision: $(round(mean(σ_scmos)*1000, digits=1)) nm")

# Compare results
println("\n=== Comparison ===")
improvement = (mean(σ_ideal) - mean(σ_scmos)) / mean(σ_ideal) * 100
println("Precision improvement: $(round(improvement, digits=1))%")
println("\nsCMOS model benefits:")
println("  ✓ Accounts for pixel-dependent readout noise")
println("  ✓ Weights pixels by their reliability")
println("  ✓ Better localization precision")

println("\n✓ sCMOS camera example complete!")
println("\nNote: Real sCMOS cameras have calibration files with")
println("      per-pixel offset, gain, and variance maps")
println("      Load these with SMLMData.SCMOSCamera()")
