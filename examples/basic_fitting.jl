"""
Basic GaussMLE fitting example
Demonstrates the simplest workflow: fit Gaussian blobs with default settings
"""

using Pkg
Pkg.activate(@__DIR__)  # Activate examples environment

using GaussMLE
using SMLMData
using Statistics

println("=== Basic Fitting Example ===\n")

# Generate sample data (11×11 pixel ROIs, 100 samples)
println("Generating synthetic data...")
data = rand(Float32, 11, 11, 100)

# Create fitter with defaults
println("Creating fitter with default settings...")
fitter = GaussMLEFitter()

# Fit
println("Fitting $(size(data, 3)) ROIs...")
smld = fit(fitter, data)

# Display results
println("\n=== Results ===")
println("Type: $(typeof(smld).name.name)")
println("Fitted: $(length(smld.emitters)) localizations")

# Extract statistics
x_positions = [e.x for e in smld.emitters]
y_positions = [e.y for e in smld.emitters]
photons = [e.photons for e in smld.emitters]
backgrounds = [e.bg for e in smld.emitters]
precisions_x = [e.σ_x for e in smld.emitters]

println("Mean position: ($(round(mean(x_positions), digits=2)), $(round(mean(y_positions), digits=2))) μm")
println("Mean photons: $(round(mean(photons), digits=1)) ± $(round(mean([e.σ_photons for e in smld.emitters]), digits=1))")
println("Mean background: $(round(mean(backgrounds), digits=1)) ± $(round(mean([e.σ_bg for e in smld.emitters]), digits=1))")
println("Mean precision: $(round(mean(precisions_x)*1000, digits=1)) nm")

println("\n✓ Basic fitting complete!")
println("\nOutput is BasicSMLD - ready for SMLMData ecosystem:")
println("  • Drift correction")
println("  • Rendering")
println("  • Analysis")
