"""
3D Astigmatic Localization Example
Demonstrates 3D fitting with astigmatic PSF model
"""

using Pkg
Pkg.activate(@__DIR__)

using GaussMLE
using SMLMData
using Statistics

println("=== 3D Astigmatic Localization Example ===\n")

# Create astigmatic PSF model
println("Creating astigmatic PSF model...")
psf_3d = AstigmaticXYZNB{Float32}(
    0.13f0, 0.13f0,  # σx₀, σy₀ (microns) at focus
    0.05f0, 0.05f0,  # Ax, Ay
    0.3f0, 0.3f0,    # Bx, By
    0.05f0,          # γ (focal offset, microns)
    0.10f0           # d (z-range, microns)
)
println("  PSF: Astigmatic 3D")

# Generate data
println("\nGenerating synthetic 3D data...")
data = rand(Float32, 11, 11, 50)

# Create fitter
println("\nCreating 3D fitter...")
fitter = GaussMLEFitter(psf_model=psf_3d, iterations=30)

# Fit
println("Fitting $(size(data, 3)) ROIs...")
smld = fit(fitter, data)

# Extract results
println("\n=== Results ===")
println("Fitted: $(length(smld.emitters)) 3D localizations")

x_pos = [e.x for e in smld.emitters]
y_pos = [e.y for e in smld.emitters]
photons = [e.photons for e in smld.emitters]
σ_x = [e.σ_x for e in smld.emitters]

println("\nPositions:")
println("  Mean x: $(round(mean(x_pos), digits=2)) ± $(round(mean(σ_x)*1000, digits=1)) nm")
println("  Mean y: $(round(mean(y_pos), digits=2)) ± $(round(mean(σ_x)*1000, digits=1)) nm")

println("\nPhotometry:")
println("  Mean photons: $(round(mean(photons), digits=1))")
println("  Mean precision: $(round(mean(σ_x)*1000, digits=1)) nm")

println("\n✓ 3D astigmatic fitting complete!")
println("Output: BasicSMLD ready for ecosystem")
