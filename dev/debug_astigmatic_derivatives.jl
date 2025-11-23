"""
Debug astigmatic derivative calculations with numerical differentiation check
"""

using Pkg
Pkg.activate(@__DIR__)

using SpecialFunctions: erf
include("../src/gausslib/baselibrary.jl")

println("="^70)
println("Astigmatic Derivative Validation")
println("="^70)

# Test parameters
σx₀, σy₀ = 1.3f0, 1.3f0
Ax, Ay = 0.05f0, -0.05f0
Bx, By = 0.01f0, -0.01f0
γ, d = 200.0f0, 500.0f0

# Test position and parameters
i, j = 8, 8
x, y, z, N, bg = 7.8f0, 7.7f0, 100.0f0, 2000.0f0, 1.0f0

println("\nTest Configuration:")
println("  Pixel: i=$i (row/y), j=$j (column/x)")
println("  Position: x=$x, y=$y, z=$z")
println("  N=$N, bg=$bg")

# Compute αx and αy at this z
αx = compute_alpha(z - γ, Ax, Bx, d)
αy = compute_alpha(z + γ, Ay, By, d)
σx = σx₀ * sqrt(αx)
σy = σy₀ * sqrt(αy)

println("\nComputed Widths:")
println("  αx = $αx (from z-γ=$(z-γ))")
println("  αy = $αy (from z+γ=$(z+γ))")
println("  σx = $σx")
println("  σy = $σy")

# Evaluate PSF
psf_x = integral_gaussian_1d(j, x, σx)  # CURRENT convention: j for x
psf_y = integral_gaussian_1d(i, y, σy)  # CURRENT convention: i for y
model = bg + N * psf_x * psf_y

println("\nPSF Evaluation:")
println("  psf_x (at j=$j, x=$x, σx=$σx) = $psf_x")
println("  psf_y (at i=$i, y=$y, σy=$σy) = $psf_y")
println("  model = $model")

# Compute analytical z-derivative
dαx_dz = (2*(z-γ)/d^2 + 3*Ax*(z-γ)^2/d^3 + 4*Bx*(z-γ)^3/d^4)
dαy_dz = (2*(z+γ)/d^2 + 3*Ay*(z+γ)^2/d^3 + 4*By*(z+γ)^3/d^4)
dσx_dz = (σx₀ / (2*sqrt(αx))) * dαx_dz
dσy_dz = (σy₀ / (2*sqrt(αy))) * dαy_dz

println("\nAnalytical Derivatives:")
println("  dαx/dz = $dαx_dz")
println("  dαy/dz = $dαy_dz")
println("  dσx/dz = $dσx_dz")
println("  dσy/dz = $dσy_dz")

# Numerical check of dσx/dz
ε = 1.0f0
αx_plus = compute_alpha((z+ε) - γ, Ax, Bx, d)
σx_plus = σx₀ * sqrt(αx_plus)
dσx_dz_num = (σx_plus - σx) / ε

println("\nNumerical Validation (dσx/dz):")
println("  Analytical: $dσx_dz")
println("  Numerical:  $dσx_dz_num")
println("  Match: $(isapprox(dσx_dz, dσx_dz_num, rtol=0.01))")

println("\n" * "="^70)
