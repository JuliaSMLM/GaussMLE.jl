"""
sCMOS Camera Example
Demonstrates fitting with sCMOS camera noise model
Shows how pixel-dependent variance affects localization uncertainties
"""

# Run from the repository root: julia --project examples/scmos_camera.jl

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
sensor_size = 64
x_corners = rand(Int32(1):Int32(sensor_size - roi_size + 1), n_rois)
y_corners = rand(Int32(1):Int32(sensor_size - roi_size + 1), n_rois)
frame_indices = ones(Int32, n_rois)

# Create data with spatially varying noise
# (simulates sCMOS pixel-dependent readout noise)
data = zeros(Float32, roi_size, roi_size, n_rois)
variance_map = zeros(Float32, sensor_size, sensor_size)

# Create realistic variance pattern
# (sCMOS sensors have pixel-dependent readout noise)
for i in 1:sensor_size, j in 1:sensor_size
    # Variance increases toward edges (typical sCMOS pattern)
    center = (sensor_size + 1) / 2.0f0
    dist_from_center = sqrt((i - center)^2 + (j - center)^2)
    variance_map[i, j] = 1.0f0 + 0.05f0 * dist_from_center
end

# Data are already in electrons, so use offset=0 and gain=1 calibration.
camera = SCMOSCamera(
    sensor_size, sensor_size, 0.1f0, sqrt.(variance_map); offset = 0.0f0, gain = 1.0f0
)

# Generate data with varying noise
for k in 1:n_rois
    for i in 1:roi_size, j in 1:roi_size
        # Signal: Gaussian blob
        signal = 100.0f0 * exp(-((i - 5.5f0)^2 + (j - 5.5f0)^2) / (2 * 1.3f0^2))
        background = 10.0f0

        # Noise: Poisson + pixel-dependent readout
        mean_val = signal + background
        cam_y = i + y_corners[k] - 1
        cam_x = j + x_corners[k] - 1
        noise_std = sqrt(mean_val + variance_map[cam_y, cam_x])

        data[i, j, k] = mean_val + randn(Float32) * noise_std
    end
end

println("  Generated $n_rois ROIs with sCMOS-like noise")
println("  Variance map range: $(round(minimum(variance_map), digits = 2)) - \
    $(round(maximum(variance_map), digits = 2))")

# Fit WITHOUT sCMOS model (ignores pixel-dependent variance)
println("\n--- Fit 1: Ideal Camera (Poisson noise only) ---")
fitter_ideal = GaussMLEConfig(
    psf_model = GaussianXYNB(0.13f0),
    iterations = 25
)
ideal_camera = IdealCamera(sensor_size, sensor_size, 0.1f0)
ideal_batch = ROIBatch(data, x_corners, y_corners, frame_indices, ideal_camera)
smld_ideal, info_ideal = fit(ideal_batch, fitter_ideal)
x_ideal = [e.x for e in smld_ideal.emitters]
σ_ideal = [e.σ_x for e in smld_ideal.emitters]
println("  Mean x: $(round(mean(x_ideal), digits = 2)) μm")
println("  Mean precision: $(round(mean(σ_ideal) * 1000, digits = 1)) nm")

# Fit WITH sCMOS model (accounts for pixel-dependent variance)
println("\n--- Fit 2: sCMOS Camera (pixel-dependent noise) ---")
fitter_scmos = GaussMLEConfig(
    psf_model = GaussianXYNB(0.13f0),
    iterations = 25
)

# Camera readnoise supplies the full-sensor variance map at each ROI's corners.
batch = ROIBatch(data, x_corners, y_corners, frame_indices, camera)
smld_scmos, info_scmos = fit(batch, fitter_scmos)
x_scmos = [e.x for e in smld_scmos.emitters]
σ_scmos = [e.σ_x for e in smld_scmos.emitters]
println("  Mean x: $(round(mean(x_scmos), digits = 2)) μm")
println("  Mean precision: $(round(mean(σ_scmos) * 1000, digits = 1)) nm")

# Compare results
println("\n=== Comparison ===")
change = (mean(σ_scmos) / mean(σ_ideal) - 1) * 100
println("Change in reported uncertainty: $(round(change, digits = 1))%")
println("\nsCMOS model benefits:")
println("  ✓ Accounts for pixel-dependent readout noise")
println("  ✓ Weights pixels by their reliability")
println("  ✓ Reports uncertainty including readout noise")

println("\n✓ sCMOS camera example complete!")
println("\nNote: Real sCMOS cameras have calibration files with")
println("      per-pixel offset, gain, and readnoise (σ, electrons) maps")
println("      Pass these to SCMOSCamera(nx, ny, pixel_size, readnoise; offset, gain)")
