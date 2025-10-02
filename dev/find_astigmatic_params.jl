"""
Find astigmatic PSF parameters that keep α > 0 for ±600nm range
"""

using Pkg
Pkg.activate("dev")

using GaussMLE
using Printf

println("="^70)
println("Finding Astigmatic PSF Parameters")
println("="^70)

# Test range: ±600nm (3σ of z_true = 200*randn())
z_range = -600.0f0:50.0f0:600.0f0

function test_params(Ax, Ay, Bx, By, γ, d, name)
    println("\n" * "="^70)
    println("Testing: $name")
    println("  Ax=$Ax, Ay=$Ay, Bx=$Bx, By=$By, γ=$γ, d=$d")
    println("-"^70)

    min_alpha_x = Inf
    min_alpha_y = Inf
    max_alpha_x = -Inf
    max_alpha_y = -Inf

    for z in z_range
        αx = GaussMLE.GaussLib.compute_alpha(z - γ, Ax, Bx, d)
        αy = GaussMLE.GaussLib.compute_alpha(z + γ, Ay, By, d)

        min_alpha_x = min(min_alpha_x, αx)
        min_alpha_y = min(min_alpha_y, αy)
        max_alpha_x = max(max_alpha_x, αx)
        max_alpha_y = max(max_alpha_y, αy)
    end

    all_positive = min_alpha_x > 0 && min_alpha_y > 0

    println("  αx range: $(min_alpha_x) to $(max_alpha_x)")
    println("  αy range: $(min_alpha_y) to $(max_alpha_y)")
    println("  Result: $(all_positive ? "✓ PASS - all α > 0" : "✗ FAIL - negative α")")

    return all_positive
end

# Test 1: Current params (from validation_utils.jl)
test_params(0.5f0, -0.5f0, 0.1f0, -0.1f0, 0.0f0, 500.0f0,
            "Current (hardcoded in validation_utils)")

# Test 2: Reduced higher-order terms
test_params(0.3f0, -0.3f0, 0.05f0, -0.05f0, 0.0f0, 500.0f0,
            "Reduced coefficients")

# Test 3: Even smaller higher-order terms
test_params(0.2f0, -0.2f0, 0.02f0, -0.02f0, 0.0f0, 500.0f0,
            "Smaller coefficients")

# Test 4: Larger depth scale (spreads out effect)
test_params(0.5f0, -0.5f0, 0.1f0, -0.1f0, 0.0f0, 700.0f0,
            "Larger depth scale d=700")

# Test 5: Very conservative (minimal aberrations)
test_params(0.1f0, -0.1f0, 0.01f0, -0.01f0, 0.0f0, 500.0f0,
            "Conservative aberrations")

# Test 6: Pure quadratic (historical working config)
test_params(0.0f0, 0.0f0, 0.0f0, 0.0f0, 250.0f0, 400.0f0,
            "Pure quadratic (historical)")

println("\n" * "="^70)
println("RECOMMENDATION:")
println("="^70)
println("For ±600nm range with realistic aberrations, use:")
println("  Ax=0.2, Ay=-0.2, Bx=0.02, By=-0.02, γ=0, d=500")
println("\nThis provides:")
println("  • Realistic higher-order aberrations (per Huang et al.)")
println("  • All α > 0 over ±600nm range")
println("  • No alpha clamping needed")
println("="^70)
