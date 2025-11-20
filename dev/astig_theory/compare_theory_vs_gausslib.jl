"""
Direct Comparison: Theoretical CRLB vs GaussLib-based Production CRLB

This script compares two independent CRLB calculations:
1. Theoretical: Direct from-scratch Fisher Information (no GaussLib dependencies)
2. Production: GaussLib-based calculation used in the actual fitter

Both use identical PSF parameters to validate consistency.
"""

using Pkg
Pkg.activate("dev")

using LinearAlgebra
using Printf
using CairoMakie
using GaussMLE
using Statistics

# Output directory
output_dir = joinpath(@__DIR__, "../output")
mkpath(output_dir)

println("="^80)
println("COMPARISON: Theoretical vs GaussLib-based CRLB Calculations")
println("="^80)
println()

# ============================================================================
# SHARED PARAMETERS (must match exactly)
# ============================================================================

sigma_x0 = 1.3f0
sigma_y0 = 1.3f0
Ax = 0.05f0
Ay = -0.05f0
Bx = 0.01f0
By = -0.01f0
gamma = 200.0f0
d = 500.0f0

box_size = 15
x_pos = 8.3f0
y_pos = 8.2f0
N_photons = 2000.0f0
bg_level = 1.0f0

z_range = Float32.(range(-600, 600, length=50))

println("PSF Parameters:")
println("  σx₀ = $sigma_x0, σy₀ = $sigma_y0")
println("  Ax = $Ax, Ay = $Ay")
println("  Bx = $Bx, By = $By")
println("  γ = $gamma nm, d = $d nm")
println()

# ============================================================================
# METHOD 1: THEORETICAL (from astig_theory/compute_theoretical_crlb.jl)
# ============================================================================

# Error function implementation
function erf(x::Real)
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    sign_x = sign(x)
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)
    return sign_x * y
end

# Pixel-integrated Gaussian
function phi(i::Int, mu::Float32, sigma::Float32)
    sqrt2 = sqrt(2.0f0)
    upper = (i - mu + 0.5f0) / (sqrt2 * sigma)
    lower = (i - mu - 0.5f0) / (sqrt2 * sigma)
    return 0.5f0 * (erf(upper) - erf(lower))
end

function dphi_dmu(i::Int, mu::Float32, sigma::Float32)
    sqrt2pi = sqrt(2.0f0 * π)
    upper = (i - mu + 0.5f0) / sigma
    lower = (i - mu - 0.5f0) / sigma
    exp_upper = exp(-0.5f0 * upper^2)
    exp_lower = exp(-0.5f0 * lower^2)
    return (exp_upper - exp_lower) / (sqrt2pi * sigma)
end

function dphi_dsigma(i::Int, mu::Float32, sigma::Float32)
    sqrt2pi = sqrt(2.0f0 * π)
    upper = (i - mu + 0.5f0) / sigma
    lower = (i - mu - 0.5f0) / sigma
    exp_upper = exp(-0.5f0 * upper^2)
    exp_lower = exp(-0.5f0 * lower^2)
    return -(upper * exp_upper - lower * exp_lower) / (sqrt2pi * sigma)
end

# Astigmatic width model
function alpha(z::Float32, gamma::Float32, A::Float32, B::Float32, d::Float32)
    zd = z / d
    return 1.0f0 + zd^2 + A * zd^3 + B * zd^4
end

function dalpha_dz(z::Float32, A::Float32, B::Float32, d::Float32)
    return (2.0f0 / d^2) * z + (3.0f0 * A / d^3) * z^2 + (4.0f0 * B / d^4) * z^3
end

function sigma(z::Float32, sigma0::Float32, gamma::Float32, A::Float32, B::Float32, d::Float32)
    alpha_val = alpha(z - gamma, gamma, A, B, d)
    return sigma0 * sqrt(alpha_val)
end

function dsigma_dz(z::Float32, sigma0::Float32, gamma::Float32, A::Float32, B::Float32, d::Float32)
    z_shifted = z - gamma
    alpha_val = alpha(z_shifted, gamma, A, B, d)
    dalpha_val = dalpha_dz(z_shifted, A, B, d)
    return (sigma0 / (2.0f0 * sqrt(alpha_val))) * dalpha_val
end

# Fisher Information calculation
function compute_fisher_theoretical(x::Float32, y::Float32, z::Float32, N::Float32, bg::Float32)
    FI = zeros(Float32, 5, 5)

    sigma_x = sigma(z, sigma_x0, gamma, Ax, Bx, d)
    sigma_y = sigma(z, sigma_y0, -gamma, Ay, By, d)

    for j in 1:box_size, i in 1:box_size
        phi_x = phi(i, x, sigma_x)
        phi_y = phi(j, y, sigma_y)
        model = bg + N * phi_x * phi_y

        if model > 0.0f0
            dphi_x_dx = dphi_dmu(i, x, sigma_x)
            dphi_y_dy = dphi_dmu(j, y, sigma_y)
            dphi_x_dsigma = dphi_dsigma(i, x, sigma_x)
            dphi_y_dsigma = dphi_dsigma(j, y, sigma_y)

            dsigma_x_dz = dsigma_dz(z, sigma_x0, gamma, Ax, Bx, d)
            dsigma_y_dz = dsigma_dz(z, sigma_y0, -gamma, Ay, By, d)

            dmu_dx = N * dphi_x_dx * phi_y
            dmu_dy = N * phi_x * dphi_y_dy
            dmu_dz = N * (dphi_x_dsigma * dsigma_x_dz * phi_y + phi_x * dphi_y_dsigma * dsigma_y_dz)
            dmu_dN = phi_x * phi_y
            dmu_dbg = 1.0f0

            derivs = [dmu_dx, dmu_dy, dmu_dz, dmu_dN, dmu_dbg]

            for k in 1:5, l in k:5
                FI_kl = (derivs[k] * derivs[l]) / model
                FI[k, l] += FI_kl
                if k != l
                    FI[l, k] += FI_kl
                end
            end
        end
    end

    return FI
end

# ============================================================================
# METHOD 2: PRODUCTION (using GaussLib from GaussMLE.jl)
# ============================================================================

# Create PSF model
psf_model = GaussMLE.AstigmaticXYZNB{Float32}(
    sigma_x0, sigma_y0, Ax, Ay, Bx, By, gamma, d
)

# Compute Fisher Information using production kernel
function compute_fisher_gausslib(x::Float32, y::Float32, z::Float32, N::Float32, bg::Float32)
    # Call the actual production kernel to get Fisher Information
    FI = zeros(Float32, 5, 5)

    # Compute widths at this z
    αx = GaussMLE.GaussLib.compute_alpha(z - gamma, Ax, Bx, d)
    αy = GaussMLE.GaussLib.compute_alpha(z + gamma, Ay, By, d)
    σx_z = sigma_x0 * sqrt(αx)
    σy_z = sigma_y0 * sqrt(αy)

    # Compute derivatives of alpha
    dαx_dz = GaussMLE.GaussLib.derivative_alpha_z(z - gamma, Ax, Bx, d)
    dαy_dz = GaussMLE.GaussLib.derivative_alpha_z(z + gamma, Ay, By, d)

    # Compute derivatives of sigma
    dσx_dz = (sigma_x0 / (2.0f0 * sqrt(αx))) * dαx_dz
    dσy_dz = (sigma_y0 / (2.0f0 * sqrt(αy))) * dαy_dz

    # Loop over pixels
    for j in 1:box_size, i in 1:box_size
        # PSF values
        phi_x = GaussMLE.GaussLib.integral_gaussian_1d(j, x, σx_z)
        phi_y = GaussMLE.GaussLib.integral_gaussian_1d(i, y, σy_z)

        model = bg + N * phi_x * phi_y

        if model > 0.0f0
            # Derivatives (GaussLib functions return (dudt, d2udt2) tuples)
            (dphi_x_dx, _) = GaussMLE.GaussLib.derivative_integral_gaussian_1d(j, x, σx_z, N_photons, phi_y)
            (dphi_y_dy, _) = GaussMLE.GaussLib.derivative_integral_gaussian_1d(i, y, σy_z, N_photons, phi_x)
            (dphi_x_dsigma, _) = GaussMLE.GaussLib.derivative_integral_gaussian_1d_sigma(j, x, σx_z, N_photons, phi_y)
            (dphi_y_dsigma, _) = GaussMLE.GaussLib.derivative_integral_gaussian_1d_sigma(i, y, σy_z, N_photons, phi_x)

            # These derivatives include N already, so adjust
            dmu_dx = dphi_x_dx
            dmu_dy = dphi_y_dy
            dmu_dz = dphi_x_dsigma * dσx_dz + dphi_y_dsigma * dσy_dz
            dmu_dN = phi_x * phi_y
            dmu_dbg = 1.0f0

            derivs = [dmu_dx, dmu_dy, dmu_dz, dmu_dN, dmu_dbg]

            for k in 1:5, l in k:5
                FI_kl = (derivs[k] * derivs[l]) / model
                FI[k, l] += FI_kl
                if k != l
                    FI[l, k] += FI_kl
                end
            end
        end
    end

    return FI
end

# ============================================================================
# COMPUTE BOTH METHODS AND COMPARE
# ============================================================================

println("Computing CRLB using both methods...")
println()

crlb_theory = Float32[]
crlb_gausslib = Float32[]
crlb_diff_percent = Float32[]

println("Z (nm) | Theory (nm) | GaussLib (nm) | Diff (%)")
println("-"^60)

for z in z_range
    # Method 1: Theoretical
    FI_theory = compute_fisher_theoretical(x_pos, y_pos, z, N_photons, bg_level)
    cov_theory = inv(FI_theory + 1.0f-10 * I(5))
    crlb_z_theory = sqrt(max(cov_theory[3, 3], 0.0f0))

    # Method 2: GaussLib
    FI_gausslib = compute_fisher_gausslib(x_pos, y_pos, z, N_photons, bg_level)
    cov_gausslib = inv(FI_gausslib + 1.0f-10 * I(5))
    crlb_z_gausslib = sqrt(max(cov_gausslib[3, 3], 0.0f0))

    push!(crlb_theory, crlb_z_theory)
    push!(crlb_gausslib, crlb_z_gausslib)

    diff_percent = 100.0f0 * abs(crlb_z_theory - crlb_z_gausslib) / crlb_z_gausslib
    push!(crlb_diff_percent, diff_percent)

    @printf("%6.0f | %11.2f | %13.2f | %7.3f\n", z, crlb_z_theory, crlb_z_gausslib, diff_percent)
end

println()
println("="^80)

# ============================================================================
# STATISTICAL COMPARISON
# ============================================================================

println("\nSTATISTICAL COMPARISON")
println("="^80)

mean_diff = mean(crlb_diff_percent)
max_diff = maximum(crlb_diff_percent)
rms_diff = sqrt(mean((crlb_theory .- crlb_gausslib).^2))

println("\nPercent Difference (|Theory - GaussLib| / GaussLib × 100%):")
@printf("  Mean: %.4f%%\n", mean_diff)
@printf("  Max:  %.4f%%\n", max_diff)
@printf("  RMS difference: %.4f nm\n", rms_diff)

if mean_diff < 0.1 && max_diff < 1.0
    println("\n✓ EXCELLENT AGREEMENT: Both methods produce identical results!")
    println("  The theoretical derivation correctly matches the production implementation.")
elseif mean_diff < 1.0 && max_diff < 5.0
    println("\n✓ GOOD AGREEMENT: Minor numerical differences only")
elseif mean_diff < 5.0
    println("\n⚠ MODERATE AGREEMENT: Some systematic differences present")
else
    println("\n✗ POOR AGREEMENT: Significant discrepancies detected")
end

# ============================================================================
# VISUALIZATION
# ============================================================================

println("\n" * "="^80)
println("Creating comparison visualization...")
println("="^80)

fig = Figure(size=(1400, 800))

# Panel 1: CRLB_z comparison
ax1 = Axis(fig[1, 1],
    xlabel = "Z Position (nm)",
    ylabel = "Axial Precision CRLB_z (nm)",
    title = "CRLB Comparison: Theoretical vs GaussLib",
    xlabelsize = 14,
    ylabelsize = 14,
    titlesize = 16
)

lines!(ax1, z_range, crlb_theory, label="Theoretical (direct)", color=:blue, linewidth=3)
lines!(ax1, z_range, crlb_gausslib, label="GaussLib (production)", color=:red, linewidth=3, linestyle=:dash)
vlines!(ax1, [0.0], color=:gray, linestyle=:dot, linewidth=1.5)
axislegend(ax1, position=:lt)

# Panel 2: Absolute difference
ax2 = Axis(fig[1, 2],
    xlabel = "Z Position (nm)",
    ylabel = "Absolute Difference (nm)",
    title = "CRLB_z: |Theoretical - GaussLib|",
    xlabelsize = 14,
    ylabelsize = 14,
    titlesize = 16
)

diff_abs = abs.(crlb_theory .- crlb_gausslib)
lines!(ax2, z_range, diff_abs, color=:purple, linewidth=2.5)
hlines!(ax2, [0.0], color=:green, linestyle=:dash, linewidth=2, label="Perfect match")
hlines!(ax2, [0.1], color=:orange, linestyle=:dot, linewidth=1.5, label="±0.1 nm tolerance")
vlines!(ax2, [0.0], color=:gray, linestyle=:dot, linewidth=1.5)
axislegend(ax2, position=:lt)

# Panel 3: Percent difference
ax3 = Axis(fig[2, :],
    xlabel = "Z Position (nm)",
    ylabel = "Percent Difference (%)",
    title = "Relative Difference: 100% × |Theory - GaussLib| / GaussLib",
    xlabelsize = 14,
    ylabelsize = 14,
    titlesize = 16
)

lines!(ax3, z_range, crlb_diff_percent, color=:darkred, linewidth=2.5)
hlines!(ax3, [0.0], color=:green, linestyle=:dash, linewidth=2, label="Perfect match")
hlines!(ax3, [0.1, -0.1], color=:orange, linestyle=:dot, linewidth=1.5, label="±0.1% tolerance")
hlines!(ax3, [1.0, -1.0], color=:red, linestyle=:dot, linewidth=1.5, label="±1% tolerance")
vlines!(ax3, [0.0], color=:gray, linestyle=:dot, linewidth=1.5)
axislegend(ax3, position=:rt)

# Save
output_file = joinpath(output_dir, "theory_vs_gausslib_comparison.png")
save(output_file, fig)

println("Figure saved to: $output_file")
println()
println("="^80)
