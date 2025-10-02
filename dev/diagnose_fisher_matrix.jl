"""
Diagnose Fisher Information Matrix for astigmatic model
"""

using Pkg
Pkg.activate(".")

using GaussMLE
using Random
using Distributions
using SMLMData
using Printf
using LinearAlgebra
using StaticArrays

println("="^70)
println("Fisher Information Matrix Diagnostic")
println("="^70)

# Model
psf_model = GaussMLE.AstigmaticXYZNB{Float32}(
    1.3f0, 1.3f0, 0.3f0, -0.3f0, 0.05f0, -0.05f0, 0.0f0, 500.0f0
)

# Generate test data
Random.seed!(42)
box_size = 11
z_true = 100.0f0
x_true = 6.0f0
y_true = 6.0f0
N_true = 1000.0f0
bg_true = 5.0f0

data = zeros(Float32, box_size, box_size)
for j in 1:box_size, i in 1:box_size
    αx = GaussMLE.GaussLib.compute_alpha(z_true - psf_model.γ, psf_model.Ax, psf_model.Bx, psf_model.d)
    αy = GaussMLE.GaussLib.compute_alpha(z_true + psf_model.γ, psf_model.Ay, psf_model.By, psf_model.d)
    σx = psf_model.σx₀ * sqrt(αx)
    σy = psf_model.σy₀ * sqrt(αy)

    psf_x = GaussMLE.GaussLib.integral_gaussian_1d(i, x_true, σx)
    psf_y = GaussMLE.GaussLib.integral_gaussian_1d(j, y_true, σy)
    mu = bg_true + N_true * psf_x * psf_y
    data[i, j] = Float32(rand(Poisson(mu)))
end

println("\nTrue parameters:")
println("  x=$x_true, y=$y_true, z=$z_true, N=$N_true, bg=$bg_true")

# Compute Fisher matrix manually
θ = [x_true, y_true, z_true, N_true, bg_true]
F = zeros(Float32, 5, 5)

for j in 1:box_size, i in 1:box_size
    θ_sv = SVector{5,Float32}(θ)
    model, dudt, _ = GaussMLE.compute_pixel_derivatives(i, j, θ_sv, psf_model)

    if model > 0
        for k in 1:5, l in 1:5
            F[k, l] += dudt[k] * dudt[l] / model
        end
    end
end

println("\n" * "="^70)
println("Fisher Information Matrix:")
println("="^70)
println("       x         y         z         N         bg")
for i in 1:5
    @printf("%s: ", ["x", "y", "z", "N", "bg"][i])
    for j in 1:5
        @printf("%9.2e ", F[i, j])
    end
    println()
end

println("\n" * "="^70)
println("Diagonal elements:")
println("="^70)
param_names = ["x", "y", "z", "N", "bg"]
for i in 1:5
    @printf("  F[%s,%s] = %.4e\n", param_names[i], param_names[i], F[i,i])
end

println("\n" * "="^70)
println("Matrix properties:")
println("="^70)
println("  Determinant: $(det(F))")
println("  Condition number: $(cond(F))")
println("  Rank: $(rank(F))")
println("  Eigenvalues: $(eigvals(F))")

if det(F) < eps(Float32)
    println("\n✗ Matrix is numerically singular!")
    println("  det(F) = $(det(F)) < eps(Float32) = $(eps(Float32))")
else
    println("\n✓ Matrix is non-singular")

    # Try inversion
    try
        F_inv = inv(F)
        println("\nCRLB uncertainties:")
        for i in 1:5
            σ = sqrt(F_inv[i,i])
            @printf("  σ_%s = %.4f\n", param_names[i], σ)
        end
    catch e
        println("\n✗ Matrix inversion failed: $e")
    end
end

println("="^70)
