"""
Direct calculation of Fisher Information Matrix and CRLB for astigmatic PSF

This implements the mathematical derivation from astigmatic_fisher_derivation.md
WITHOUT using any GaussLib functions - everything derived from first principles.
"""

using Pkg
Pkg.activate("dev")

using LinearAlgebra
using Printf
using CairoMakie

# Error function - accurate approximation (Abramowitz and Stegun 7.1.26)
function erf(x::Real)
    # Constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # Save the sign of x
    sign_x = sign(x)
    x = abs(x)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)

    return sign_x * y
end

# Output directory
output_dir = joinpath(@__DIR__, "../output")
mkpath(output_dir)

println("="^80)
println("Theoretical Fisher Information and CRLB Calculation")
println("="^80)
println()

# ============================================================================
# 1. PIXEL-INTEGRATED GAUSSIAN FUNCTIONS (from first principles)
# ============================================================================

"""
Pixel-integrated 1D Gaussian over pixel i (from i-0.5 to i+0.5)
"""
function phi(i::Int, mu::Float32, sigma::Float32)
    sqrt2 = sqrt(2.0f0)
    upper = (i - mu + 0.5f0) / (sqrt2 * sigma)
    lower = (i - mu - 0.5f0) / (sqrt2 * sigma)
    return 0.5f0 * (erf(upper) - erf(lower))
end

"""
Derivative of phi with respect to position (mu)
"""
function dphi_dmu(i::Int, mu::Float32, sigma::Float32)
    sqrt2pi = sqrt(2.0f0 * π)
    upper = (i - mu + 0.5f0) / sigma
    lower = (i - mu - 0.5f0) / sigma
    exp_upper = exp(-0.5f0 * upper^2)
    exp_lower = exp(-0.5f0 * lower^2)
    return (exp_upper - exp_lower) / (sqrt2pi * sigma)
end

"""
Derivative of phi with respect to width (sigma)
"""
function dphi_dsigma(i::Int, mu::Float32, sigma::Float32)
    sqrt2pi = sqrt(2.0f0 * π)
    upper = (i - mu + 0.5f0) / sigma
    lower = (i - mu - 0.5f0) / sigma
    exp_upper = exp(-0.5f0 * upper^2)
    exp_lower = exp(-0.5f0 * lower^2)
    return -(upper * exp_upper - lower * exp_lower) / (sqrt2pi * sigma)
end

# ============================================================================
# 2. ASTIGMATIC WIDTH MODEL
# ============================================================================

"""
Alpha scaling factor for astigmatic PSF width
α(z) = 1 + ((z±γ)/d)² + A((z±γ)/d)³ + B((z±γ)/d)⁴
"""
function alpha(z::Float32, gamma::Float32, A::Float32, B::Float32, d::Float32)
    zd = z / d
    return 1.0f0 + zd^2 + A * zd^3 + B * zd^4
end

"""
Derivative of alpha with respect to z
dα/dz = (2/d²)z + (3A/d³)z² + (4B/d⁴)z³
"""
function dalpha_dz(z::Float32, A::Float32, B::Float32, d::Float32)
    return (2.0f0 / d^2) * z + (3.0f0 * A / d^3) * z^2 + (4.0f0 * B / d^4) * z^3
end

"""
Width as function of z: σ(z) = σ₀ √α(z)
"""
function sigma(z::Float32, sigma0::Float32, gamma::Float32, A::Float32, B::Float32, d::Float32)
    alpha_val = alpha(z - gamma, gamma, A, B, d)
    return sigma0 * sqrt(alpha_val)
end

"""
Derivative of width with respect to z: dσ/dz = (σ₀/(2√α)) dα/dz
"""
function dsigma_dz(z::Float32, sigma0::Float32, gamma::Float32, A::Float32, B::Float32, d::Float32)
    z_shifted = z - gamma
    alpha_val = alpha(z_shifted, gamma, A, B, d)
    dalpha_val = dalpha_dz(z_shifted, A, B, d)
    return (sigma0 / (2.0f0 * sqrt(alpha_val))) * dalpha_val
end

# ============================================================================
# 3. PSF MODEL AND DERIVATIVES
# ============================================================================

"""
PSF model: μ(i,j) = bg + N·Φ(i|x,σx)·Φ(j|y,σy)
"""
function psf_model(i::Int, j::Int, x::Float32, y::Float32, z::Float32, N::Float32, bg::Float32,
                   sigma_x0::Float32, sigma_y0::Float32,
                   Ax::Float32, Ay::Float32, Bx::Float32, By::Float32,
                   gamma::Float32, d::Float32)
    # Compute widths at this z
    sigma_x = sigma(z, sigma_x0, gamma, Ax, Bx, d)
    sigma_y = sigma(z, sigma_y0, -gamma, Ay, By, d)  # Note: -gamma for y

    # Pixel-integrated PSF
    phi_x = phi(i, x, sigma_x)
    phi_y = phi(j, y, sigma_y)

    return bg + N * phi_x * phi_y
end

"""
Compute all 5 derivatives of PSF model
Returns: (model_value, [∂μ/∂x, ∂μ/∂y, ∂μ/∂z, ∂μ/∂N, ∂μ/∂bg])
"""
function psf_derivatives(i::Int, j::Int, x::Float32, y::Float32, z::Float32, N::Float32, bg::Float32,
                        sigma_x0::Float32, sigma_y0::Float32,
                        Ax::Float32, Ay::Float32, Bx::Float32, By::Float32,
                        gamma::Float32, d::Float32)
    # Compute widths at this z
    sigma_x = sigma(z, sigma_x0, gamma, Ax, Bx, d)
    sigma_y = sigma(z, sigma_y0, -gamma, Ay, By, d)

    # PSF values
    phi_x = phi(i, x, sigma_x)
    phi_y = phi(j, y, sigma_y)

    # Model value
    model = bg + N * phi_x * phi_y

    # Position derivatives
    dphi_x_dx = dphi_dmu(i, x, sigma_x)
    dphi_y_dy = dphi_dmu(j, y, sigma_y)

    dmu_dx = N * dphi_x_dx * phi_y
    dmu_dy = N * phi_x * dphi_y_dy

    # Z derivative (chain rule through sigma)
    dphi_x_dsigma = dphi_dsigma(i, x, sigma_x)
    dphi_y_dsigma = dphi_dsigma(j, y, sigma_y)

    dsigma_x_dz = dsigma_dz(z, sigma_x0, gamma, Ax, Bx, d)
    dsigma_y_dz = dsigma_dz(z, sigma_y0, -gamma, Ay, By, d)

    dmu_dz = N * (dphi_x_dsigma * dsigma_x_dz * phi_y +
                  phi_x * dphi_y_dsigma * dsigma_y_dz)

    # Photon derivative
    dmu_dN = phi_x * phi_y

    # Background derivative
    dmu_dbg = 1.0f0

    derivatives = [dmu_dx, dmu_dy, dmu_dz, dmu_dN, dmu_dbg]

    return (model, derivatives)
end

# ============================================================================
# 4. FISHER INFORMATION MATRIX CALCULATION
# ============================================================================

"""
Compute the 5×5 Fisher Information Matrix for given parameters
FI[k,l] = Σᵢⱼ (1/μ)·(∂μ/∂θk)·(∂μ/∂θl)
"""
function compute_fisher_information(x::Float32, y::Float32, z::Float32, N::Float32, bg::Float32,
                                   box_size::Int,
                                   sigma_x0::Float32, sigma_y0::Float32,
                                   Ax::Float32, Ay::Float32, Bx::Float32, By::Float32,
                                   gamma::Float32, d::Float32)
    # Initialize 5×5 Fisher matrix
    FI = zeros(Float32, 5, 5)

    # Sum over all pixels
    for j in 1:box_size, i in 1:box_size
        model, derivs = psf_derivatives(i, j, x, y, z, N, bg,
                                       sigma_x0, sigma_y0, Ax, Ay, Bx, By, gamma, d)

        # Only include pixels with positive model value
        if model > 0.0f0
            # Accumulate Fisher Information
            # FI is symmetric, so only compute upper triangle
            for k in 1:5, l in k:5
                FI_kl = (derivs[k] * derivs[l]) / model
                FI[k, l] += FI_kl
                if k != l
                    FI[l, k] += FI_kl  # Symmetry
                end
            end
        end
    end

    return FI
end

"""
Compute CRLB from Fisher Information Matrix
CRLB = diagonal elements of FI⁻¹
"""
function compute_crlb(FI::Matrix{Float32})
    # Add small regularization for numerical stability
    reg = 1.0f-10 * maximum(abs, FI)
    FI_reg = FI + reg * I(5)

    # Invert to get covariance matrix
    try
        cov_matrix = inv(FI_reg)
        # Extract diagonal (variances)
        crlb = sqrt.(max.(diag(cov_matrix), 0.0f0))
        return crlb
    catch
        # If inversion fails, return infinite uncertainties
        return fill(Inf32, 5)
    end
end

# ============================================================================
# 5. COMPUTE CRLB VS Z
# ============================================================================

println("Computing theoretical CRLB vs z...")
println()

# PSF parameters (matching the test)
sigma_x0 = 1.3f0
sigma_y0 = 1.3f0
Ax = 0.05f0
Ay = -0.05f0
Bx = 0.01f0
By = -0.01f0
gamma = 200.0f0
d = 500.0f0

# Test conditions (matching the analysis)
box_size = 15
x_pos = 8.3f0
y_pos = 8.2f0
N_photons = 2000.0f0
bg_level = 1.0f0

# Z range
z_range = Float32.(range(-600, 600, length=50))

# Storage for results
crlb_x = Float32[]
crlb_y = Float32[]
crlb_z = Float32[]
crlb_N = Float32[]
crlb_bg = Float32[]

# Storage for gradient magnitudes
dsigma_x_dz_vals = Float32[]
dsigma_y_dz_vals = Float32[]

println("Z (nm) | CRLB_x (px) | CRLB_y (px) | CRLB_z (nm) | |dσx/dz| | |dσy/dz|")
println("-"^80)

for z in z_range
    # Compute Fisher Information
    FI = compute_fisher_information(x_pos, y_pos, z, N_photons, bg_level,
                                   box_size, sigma_x0, sigma_y0,
                                   Ax, Ay, Bx, By, gamma, d)

    # Compute CRLB
    crlb = compute_crlb(FI)

    push!(crlb_x, crlb[1])
    push!(crlb_y, crlb[2])
    push!(crlb_z, crlb[3])
    push!(crlb_N, crlb[4])
    push!(crlb_bg, crlb[5])

    # Also compute gradient magnitudes for comparison
    dsx = abs(dsigma_dz(z, sigma_x0, gamma, Ax, Bx, d))
    dsy = abs(dsigma_dz(z, sigma_y0, -gamma, Ay, By, d))
    push!(dsigma_x_dz_vals, dsx)
    push!(dsigma_y_dz_vals, dsy)

    @printf("%6.0f | %11.4f | %11.4f | %11.2f | %9.6f | %9.6f\n",
            z, crlb[1], crlb[2], crlb[3], dsx, dsy)
end

println()
println("="^80)

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

println("\nCreating visualization...")

fig = Figure(size=(1400, 1000))

# Panel 1: PSF widths vs z
ax1 = Axis(fig[1, 1],
    xlabel = "Z Position (nm)",
    ylabel = "PSF Width σ (pixels)",
    title = "Astigmatic PSF Width vs Depth",
    xlabelsize = 14,
    ylabelsize = 14,
    titlesize = 16
)

sigma_x_vals = [sigma(z, sigma_x0, gamma, Ax, Bx, d) for z in z_range]
sigma_y_vals = [sigma(z, sigma_y0, -gamma, Ay, By, d) for z in z_range]

lines!(ax1, z_range, sigma_x_vals, label="σx(z)", color=:blue, linewidth=3)
lines!(ax1, z_range, sigma_y_vals, label="σy(z)", color=:red, linewidth=3)
vlines!(ax1, [0.0], color=:gray, linestyle=:dash)
axislegend(ax1, position=:lt)

# Panel 2: Width derivatives (z-information)
ax2 = Axis(fig[1, 2],
    xlabel = "Z Position (nm)",
    ylabel = "|dσ/dz| (pixels/nm)",
    title = "Z-Information: Width Gradient Magnitude",
    xlabelsize = 14,
    ylabelsize = 14,
    titlesize = 16
)

lines!(ax2, z_range, dsigma_x_dz_vals, label="|dσx/dz|", color=:blue, linewidth=2.5)
lines!(ax2, z_range, dsigma_y_dz_vals, label="|dσy/dz|", color=:red, linewidth=2.5)
vlines!(ax2, [0.0], color=:gray, linestyle=:dash)
axislegend(ax2, position=:lt)

# Panel 3: Lateral CRLB
ax3 = Axis(fig[2, 1],
    xlabel = "Z Position (nm)",
    ylabel = "Localization Precision (pixels)",
    title = "CRLB: Lateral Precision vs Depth",
    xlabelsize = 14,
    ylabelsize = 14,
    titlesize = 16
)

lines!(ax3, z_range, crlb_x, label="CRLB x", color=:blue, linewidth=2.5)
lines!(ax3, z_range, crlb_y, label="CRLB y", color=:red, linewidth=2.5)
vlines!(ax3, [0.0], color=:gray, linestyle=:dash)
axislegend(ax3, position=:lt)

# Panel 4: Axial CRLB
ax4 = Axis(fig[2, 2],
    xlabel = "Z Position (nm)",
    ylabel = "Axial Precision (nm)",
    title = "CRLB: Z Precision vs Depth",
    xlabelsize = 14,
    ylabelsize = 14,
    titlesize = 16
)

lines!(ax4, z_range, crlb_z, label="CRLB z", color=:purple, linewidth=3)
vlines!(ax4, [0.0], color=:gray, linestyle=:dash)
axislegend(ax4, position=:lt)

# Save figure
output_file = joinpath(output_dir, "theoretical_crlb_vs_z.png")
save(output_file, fig)

println("Figure saved to: $output_file")
println()

# ============================================================================
# 7. SUMMARY
# ============================================================================

println("="^80)
println("SUMMARY")
println("="^80)
println()

println("PSF Parameters:")
println("  σx₀ = $sigma_x0, σy₀ = $sigma_y0")
println("  Ax = $Ax, Ay = $Ay")
println("  Bx = $Bx, By = $By")
println("  γ = $gamma nm, d = $d nm")
println()

println("Test Conditions:")
println("  Box size: $(box_size)×$(box_size) pixels")
println("  Position: x=$x_pos, y=$y_pos")
println("  Photons: $N_photons")
println("  Background: $bg_level photons/pixel")
println()

println("CRLB Results:")
@printf("  Lateral precision (x): %.3f - %.3f pixels\n", minimum(crlb_x), maximum(crlb_x))
@printf("  Lateral precision (y): %.3f - %.3f pixels\n", minimum(crlb_y), maximum(crlb_y))
@printf("  Axial precision (z): %.1f - %.1f nm\n", minimum(crlb_z), maximum(crlb_z))
println()

# Find z positions with best and worst z-precision
idx_best = argmin(crlb_z)
idx_worst = argmax(crlb_z)

println("Z-Precision Analysis:")
@printf("  Best at z = %.0f nm: CRLB_z = %.1f nm\n", z_range[idx_best], crlb_z[idx_best])
@printf("  Worst at z = %.0f nm: CRLB_z = %.1f nm\n", z_range[idx_worst], crlb_z[idx_worst])
@printf("  Ratio (worst/best) = %.2f×\n", crlb_z[idx_worst] / crlb_z[idx_best])
println()

println("="^80)
