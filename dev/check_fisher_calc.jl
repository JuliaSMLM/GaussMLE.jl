"""
Check Fisher Information calculation for astigmatic model
"""

using Pkg
Pkg.activate(".")

using GaussMLE
using Random
using Distributions
using SMLMData
using Printf

println("="^70)
println("Fisher Information Diagnostic")
println("="^70)

# Model
psf_model = GaussMLE.AstigmaticXYZNB{Float32}(
    1.3f0, 1.3f0, 0.2f0, -0.2f0, 0.02f0, -0.02f0, 0.0f0, 500.0f0
)

# Generate single test ROI
Random.seed!(42)
z_true = 100.0f0
params_true = Float32[6.0, 6.0, z_true, 1000.0, 5.0]

# Generate data
camera = SMLMData.IdealCamera(128, 128, 1.0f0)
data = zeros(Float32, 11, 11, 1)
for j in 1:11, i in 1:11
    αx = GaussMLE.GaussLib.compute_alpha(z_true - psf_model.γ, psf_model.Ax, psf_model.Bx, psf_model.d)
    αy = GaussMLE.GaussLib.compute_alpha(z_true + psf_model.γ, psf_model.Ay, psf_model.By, psf_model.d)
    σx = psf_model.σx₀ * sqrt(αx)
    σy = psf_model.σy₀ * sqrt(αy)

    psf_x = GaussMLE.GaussLib.integral_gaussian_1d(i, params_true[1], σx)
    psf_y = GaussMLE.GaussLib.integral_gaussian_1d(j, params_true[2], σy)
    mu = params_true[5] + params_true[4] * psf_x * psf_y
    data[i, j, 1] = Float32(rand(Poisson(mu)))
end

println("\nTrue parameters: x=$(params_true[1]), y=$(params_true[2]), z=$(params_true[3]), N=$(params_true[4]), bg=$(params_true[5])")

# Fit
fitter = GaussMLE.GaussMLEFitter(psf_model=psf_model, iterations=20)
results = GaussMLE.fit(fitter, data)

println("\nFitted parameters:")
@printf("  x: %.3f ± %.3f\n", results.x[1], results.x_error[1])
@printf("  y: %.3f ± %.3f\n", results.y[1], results.y_error[1])
@printf("  z: %.1f ± %.1f\n", results.z[1], results.z_error[1])
@printf("  N: %.1f ± %.1f\n", results.photons[1], results.photons_error[1])
@printf("  bg: %.2f ± %.2f\n", results.background[1], results.background_error[1])

if isinf(results.x_error[1])
    println("\n✗ ERROR: Uncertainties are Inf!")
    println("  This suggests Fisher Information matrix is singular or has NaN/Inf values")
    println("  Likely causes:")
    println("    - Alpha goes negative during derivative calculation")
    println("    - Division by zero in derivative computation")
    println("    - Numerical overflow in second derivatives")
else
    println("\n✓ Uncertainties are finite")
end

println("="^70)
