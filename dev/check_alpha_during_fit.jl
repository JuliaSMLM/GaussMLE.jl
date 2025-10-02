"""
Check if alpha goes negative during fitting with current parameters
"""

using Pkg
Pkg.activate(".")

using GaussMLE
using Printf

println("="^70)
println("Check Alpha Values During Fitting")
println("="^70)

# Test parameters
σx₀ = 1.3f0
σy₀ = 1.3f0
Ax = 0.2f0
Ay = -0.2f0
Bx = 0.02f0
By = -0.02f0
γ = 0.0f0
d = 500.0f0

println("\nModel Parameters:")
println("  Ax=$Ax, Ay=$Ay, Bx=$Bx, By=$By, γ=$γ, d=$d")

# Test z range that might occur during fitting
# Start at z_init=0, true values ~200*randn() ≈ ±600nm
# But during fitting z could wander outside this range
z_test_range = -800.0f0:50.0f0:800.0f0

println("\n" * "="^70)
println("Alpha values over extended range (fitting might explore):")
println("="^70)
println("    Z (nm) |    αx    |    αy    | Status")
println("-"^70)

min_αx_val = Inf
min_αy_val = Inf
has_negative = false

for z in z_test_range
    αx = GaussMLE.GaussLib.compute_alpha(z - γ, Ax, Bx, d)
    αy = GaussMLE.GaussLib.compute_alpha(z + γ, Ay, By, d)

    global min_αx_val = min(min_αx_val, αx)
    global min_αy_val = min(min_αy_val, αy)

    if αx < 0 || αy < 0
        @printf("%10.0f | %8.3f | %8.3f | ✗ NEGATIVE!\n", z, αx, αy)
        global has_negative = true
    elseif αx < 0.1 || αy < 0.1
        @printf("%10.0f | %8.3f | %8.3f | ⚠ Low (<0.1)\n", z, αx, αy)
    end
end

println("-"^70)
println("Min αx: $min_αx_val")
println("Min αy: $min_αy_val")

if has_negative
    println("\n✗ PROBLEM: Alpha goes negative in ±800nm range!")
    println("  This will cause NaN during fitting (sqrt of negative)")
    println("  Need to either:")
    println("    1. Reduce higher-order coefficients further")
    println("    2. Add back alpha clamping (but that hides the issue)")
    println("    3. Constrain z range during fitting")
else
    println("\n✓ Alpha stays positive in ±800nm range")
    println("  Fitting should not encounter domain errors")
end

println("\n" * "="^70)
println("Recommendation:")
println("="^70)
if has_negative
    println("Alpha goes negative - reduce Bx, By further:")
    println("  Try: Ax=0.15, Ay=-0.15, Bx=0.01, By=-0.01")
else
    println("Alpha stays positive. Issue must be elsewhere:")
    println("  - Check Fisher Information calculation")
    println("  - Check derivative computation for z-parameter")
    println("  - Verify convergence criteria")
end
println("="^70)
