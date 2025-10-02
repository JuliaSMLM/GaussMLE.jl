"""
Test if parameters are too small for z-sensitivity
"""

using Pkg
Pkg.activate(".")

using GaussMLE
using Printf

println("="^70)
println("Parameter Sensitivity Analysis")
println("="^70)

# Current parameters
Ax_curr = 0.2f0
Ay_curr = -0.2f0
Bx_curr = 0.02f0
By_curr = 0.02f0
d = 500.0f0
γ = 0.0f0

println("\nCurrent parameters: Ax=$Ax_curr, Ay=$Ay_curr, Bx=$Bx_curr, By=$By_curr")

# Test derivative of alpha with respect to z at z=100nm
z_test = 100.0f0

# α(z) = 1 + (z/d)² + A·(z/d)³ + B·(z/d)⁴
# dα/dz = (2z/d² + 3A·z²/d³ + 4B·z³/d⁴)

function test_sensitivity(Ax, Bx)
    z_norm = z_test / d

    α = 1.0f0 + z_norm^2 + Ax * z_norm^3 + Bx * z_norm^4

    # Derivative
    dα_dz = (2 * z_test / d^2 + 3 * Ax * z_test^2 / d^3 + 4 * Bx * z_test^3 / d^4)

    return (α, dα_dz)
end

# Test current
(α_curr, dαdz_curr) = test_sensitivity(Ax_curr, Bx_curr)
println("\n  At z=$z_test nm:")
println("    α = $α_curr")
println("    dα/dz = $dαdz_curr")

# Test with original larger coefficients
Ax_orig = 0.5f0
Bx_orig = 0.1f0
(α_orig, dαdz_orig) = test_sensitivity(Ax_orig, Bx_orig)
println("\nOriginal parameters (Ax=0.5, Bx=0.1):")
println("  At z=$z_test nm:")
println("    α = $α_orig")
println("    dα/dz = $dαdz_orig")

println("\nSensitivity ratio: $(dαdz_orig / dαdz_curr)×")

if dαdz_curr < 0.001
    println("\n⚠ WARNING: Current parameters have very low sensitivity!")
    println("  dα/dz is too small - Fisher Information for z will be weak")
    println("  This explains Inf uncertainties")
    println("\nRECOMMENDATION: Use larger coefficients")
    println("  Try: Ax=0.3, Ay=-0.3, Bx=0.05, By=-0.05")
else
    println("\n✓ Sensitivity is reasonable")
end

println("="^70)
