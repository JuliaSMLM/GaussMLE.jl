"""
Example usage of the refactored GaussMLE API
"""

using GaussMLE
using Random
using Statistics

# Set seed for reproducibility
Random.seed!(42)

# Generate synthetic data
println("Generating synthetic data...")
box_size = 11
n_spots = 100

# Create data array
data = zeros(Float32, box_size, box_size, n_spots)

# Generate random spots
for i in 1:n_spots
    # True parameters
    x = 5.5f0 + randn(Float32) * 0.5f0
    y = 5.5f0 + randn(Float32) * 0.5f0
    N = 1000.0f0 + randn(Float32) * 200.0f0
    bg = 10.0f0 + randn(Float32) * 2.0f0
    σ = 1.3f0
    
    # Generate Gaussian spot with Poisson noise
    for jj in 1:box_size, ii in 1:box_size
        intensity = N * exp(-((ii - x)^2 + (jj - y)^2) / (2 * σ^2)) / (2π * σ^2)
        expected = intensity + bg
        # Add Poisson noise
        data[ii, jj, i] = expected + randn(Float32) * sqrt(max(0, expected))
    end
end

println("\n=== Example 1: Simple usage with CPU ===")
# Use CPU for now as GPU kernel needs more work
fitter = GaussMLEFitter(device=CPU())
results = fit(fitter, data)

println(results)
println("\nMean localization precision:")
println("  X: $(mean(results.x_error)) pixels")
println("  Y: $(mean(results.y_error)) pixels")

println("\n=== Example 2: Explicit CPU usage ===")
cpu_fitter = GaussMLEFitter(device=CPU())
cpu_results = fit(cpu_fitter, data)
println("CPU fitting completed: $(cpu_results.n_fits) spots fitted")

println("\n=== Example 3: Variable sigma model ===")
var_fitter = GaussMLEFitter(
    psf_model = GaussianXYNBS(),
    device = CPU(),
    iterations = 30
)
var_results = fit(var_fitter, data)
println("Variable sigma model results:")
println("  Mean sigma: $(mean(var_results.sigma)) ± $(mean(var_results.sigma_error)) pixels")

println("\n=== Example 4: sCMOS camera with variance map ===")
# Create a variance map (simulating sCMOS readout noise)
variance_map = ones(Float32, box_size, box_size) * 25.0f0  # 5 e- readout noise

scmos_fitter = GaussMLEFitter(
    camera_model = SCMOSCamera(variance_map),
    psf_model = GaussianXYNBS(),
    device = CPU()
)
scmos_results = fit(scmos_fitter, data)
println("sCMOS fitting completed")
println("  Mean photons: $(mean(scmos_results.photons)) ± $(mean(scmos_results.photons_error))")

println("\n=== Example 5: Custom constraints ===")
# Tight constraints for challenging data
tight_constraints = ParameterConstraints{4}(
    GaussMLE.Params{4}(Float32[-1.0, -1.0, 10.0, 0.1]),        # lower bounds
    GaussMLE.Params{4}(Float32[12.0, 12.0, 1e5, 100.0]),       # upper bounds
    GaussMLE.Params{4}(Float32[0.5, 0.5, 1e4, 50.0])           # max step
)

custom_fitter = GaussMLEFitter(
    psf_model = GaussianXYNB(1.5f0),
    constraints = tight_constraints,
    device = CPU(),
    iterations = 30
)
custom_results = fit(custom_fitter, data)
println("Custom constraints fitting completed")

println("\n=== Example 6: Single ROI fitting ===")
single_roi = data[:, :, 1]
single_fitter = GaussMLEFitter(device=CPU())
params, errors = fit(single_fitter, single_roi)
println("Single ROI parameters:")
println("  x = $(params[1]) ± $(errors[1])")
println("  y = $(params[2]) ± $(errors[2])")
println("  photons = $(params[3]) ± $(errors[3])")
println("  background = $(params[4]) ± $(errors[4])")

println("\n=== Example 7: Legacy compatibility ===")
# Old API still works
θ_legacy, Σ_legacy = fitstack(data, :xynb; σ_PSF=1.3, backend=:cpu)
println("Legacy fitstack: fitted $(size(θ_legacy, 2)) spots")
println("  Mean x position: $(mean(θ_legacy[1, :]))")
println("  Mean y position: $(mean(θ_legacy[2, :]))")

println("\n✅ All examples completed successfully!")