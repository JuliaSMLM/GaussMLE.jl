"""
Test astigmatic with asymmetric off-center position and high photons
to reveal any x/y coordinate swap bugs
"""

using Pkg
Pkg.activate("dev")

using GaussMLE
using SMLMData
using Random
using Statistics
using Distributions: Poisson

println("="^70)
println("Asymmetric Position Test (x≠y, High Photons)")
println("="^70)

psf = AstigmaticXYZNB{Float32}(1.3f0,1.3f0,0.05f0,-0.05f0,0.01f0,-0.01f0,200.0f0,500.0f0)
box_size = 15
pixel_size = 0.1f0

# ASYMMETRIC position (different x and y)
x_true_roi = 10.3f0  # Far right
y_true_roi = 5.7f0   # Above center
z_true = 0.0f0
n_photons = 10000.0f0  # High SNR
bg = 1.0f0

println("\nGround Truth (ROI coordinates):")
println("  x = $x_true_roi pixels")
println("  y = $y_true_roi pixels")
println("  z = $z_true nm")
println("  N = $n_photons photons")

# Generate 100 ROIs with same true position
Random.seed!(42)
n_rois = 100
data = zeros(Float32, box_size, box_size, n_rois)
params_true = Float32[x_true_roi, y_true_roi, z_true, n_photons, bg]

for k in 1:n_rois
    for j in 1:box_size, i in 1:box_size
        μ = GaussMLE._evaluate_psf_pixel(psf, i, j, params_true)
        data[i, j, k] = Float32(rand(Poisson(μ)))
    end
end

# Fit with many iterations for convergence
fitter = GaussMLEFitter(psf_model=psf, iterations=50)
results = fit(fitter, data)

# Extract fitted values (convert from microns to ROI pixels)
# fit(Array) uses corners: x_corners[i] = 1 + (i-1)*box_size
fitted_x_roi = Float32[]
fitted_y_roi = Float32[]
fitted_z = Float32[]

for (i, e) in enumerate(results.emitters)
    # Reverse transform
    x_corner = 1 + (i-1) * box_size
    y_corner = 1

    x_cam = e.x / pixel_size + 1  # microns → camera pixels
    y_cam = e.y / pixel_size + 1

    x_roi = x_cam - x_corner + 1  # camera → ROI
    y_roi = y_cam - y_corner + 1

    push!(fitted_x_roi, x_roi)
    push!(fitted_y_roi, y_roi)
    push!(fitted_z, e.z / pixel_size)
end

println("\nFitted (ROI coordinates):")
println("  x = $(round(mean(fitted_x_roi), digits=3)) ± $(round(std(fitted_x_roi), digits=3))")
println("  y = $(round(mean(fitted_y_roi), digits=3)) ± $(round(std(fitted_y_roi), digits=3))")
println("  z = $(round(mean(fitted_z), digits=2)) ± $(round(std(fitted_z), digits=2))")

println("\nBiases:")
bias_x = mean(fitted_x_roi) - x_true_roi
bias_y = mean(fitted_y_roi) - y_true_roi
bias_z = mean(fitted_z) - z_true

println("  Δx = $(round(bias_x, digits=3)) pixels")
println("  Δy = $(round(bias_y, digits=3)) pixels")
println("  Δz = $(round(bias_z, digits=2)) nm")

println("\n" * "="^70)
println("ANALYSIS")
println("="^70)

println("\nIf x and y were swapped:")
println("  We'd see: Δx ≈ $(round(y_true_roi - x_true_roi, digits=1)) and Δy ≈ $(round(x_true_roi - y_true_roi, digits=1))")
println("  We got:   Δx = $(round(bias_x, digits=2)) and Δy = $(round(bias_y, digits=2))")

if abs(bias_x - (y_true_roi - x_true_roi)) < 0.5 && abs(bias_y - (x_true_roi - y_true_roi)) < 0.5
    println("  ⚠ SWAP DETECTED! X and Y are swapped somewhere!")
elseif abs(bias_x) < 0.2 && abs(bias_y) < 0.2
    println("  ✓ NO SWAP: Biases are small and symmetric")
else
    println("  ? SYSTEMATIC BIAS: Not a simple swap, but biases present")
end

println("\n" * "="^70)
