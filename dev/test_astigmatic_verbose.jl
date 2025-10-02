"""
Test astigmatic model with verbose output
"""

using Pkg
Pkg.activate(".")

using GaussMLE
using Test
using Random
using Statistics
using Distributions
using LinearAlgebra
using SMLMData

# Include validation utilities
include("../test/validation_utils.jl")

println("="^70)
println("Astigmatic 3D Model Test (Verbose)")
println("="^70)

# Realistic astigmatic calibration
psf_model = GaussMLE.AstigmaticXYZNB{Float32}(
    1.3f0, 1.3f0,     # σx₀, σy₀
    0.2f0, -0.2f0,    # Ax, Ay
    0.02f0, -0.02f0,  # Bx, By
    0.0f0,            # γ
    500.0f0           # d
)

# Run validation with verbose output
println("\nRunning validation with 1000 blobs...")
passed, results = run_model_validation(
    :xynbz, psf_model, 1000;
    box_size = 11,
    device = GaussMLE.CPU(),
    sigma = 1.3f0,
    verbose = true
)

println("\n" * "="^70)
println("DETAILED RESULTS:")
println("="^70)
for (param, result) in results
    println("\n$param:")
    println("  Bias: $(result.bias)")
    println("  Empirical STD: $(result.empirical_std)")
    println("  Reported STD: $(result.mean_reported_std)")
    println("  STD ratio: $(result.std_ratio)")
    println("  Bias pass: $(result.bias_pass)")
    println("  STD pass: $(result.std_pass)")
end

println("\n" * "="^70)
println("Overall: $(passed ? "PASS ✓" : "FAIL ✗")")
println("="^70)
