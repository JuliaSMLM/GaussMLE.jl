"""
Comprehensive Analysis of Astigmatic PSF: Widths and CRLB vs Z

This script validates the astigmatic PSF model by:
1. Plotting σx(z) and σy(z) across the z-range to show astigmatic behavior
2. Computing CRLB for x, y, z positions by fitting synthetic data with:
   - FIXED true position at each z (proper CRLB validation)
   - Multiple Poisson noise realizations (n=100)
   - Measuring std(fitted - true) vs predicted CRLB
3. Visualizing localization precision vs z
4. Validating that empirical errors match Fisher Information predictions
"""

using Pkg
Pkg.activate("dev")

using GaussMLE
using CairoMakie
using Statistics
using Random
using Distributions

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("="^70)
println("Astigmatic PSF: Width and CRLB Analysis")
println("="^70)

# Use the EXACT parameters from the failing test
println("\nUsing test PSF parameters:")
psf_model = GaussMLE.AstigmaticXYZNB{Float32}(
    1.3f0, 1.3f0,     # σx₀, σy₀ - diffraction-limited base width
    0.05f0, -0.05f0,  # Ax, Ay - cubic aberrations (opposite signs for astigmatism)
    0.01f0, -0.01f0,  # Bx, By - quartic aberrations (opposite signs)
    200.0f0,          # γ = 200nm (offset focal planes - realistic astigmatic system)
    500.0f0           # d = 500nm (typical depth scale for ±600nm range)
)

println("  σx₀ = $(psf_model.σx₀), σy₀ = $(psf_model.σy₀)")
println("  Ax = $(psf_model.Ax), Ay = $(psf_model.Ay)")
println("  Bx = $(psf_model.Bx), By = $(psf_model.By)")
println("  γ = $(psf_model.γ) nm, d = $(psf_model.d) nm")

# Test parameters
box_size = 15
n_photons = 2000.0f0
background = 1.0f0
n_samples_per_z = 1000  # Number of fits per z position

println("\nTest conditions:")
println("  Box size: $(box_size)×$(box_size) pixels")
println("  Photons: $(n_photons)")
println("  Background: $(background) photons/pixel")
println("  Samples per z: $(n_samples_per_z)")

# Z range to test
z_range = Float32.(range(-600, 600, length=25))  # 25 z positions
println("\nTesting z range: $(minimum(z_range)) to $(maximum(z_range)) nm")

# =============================================================================
# Part 1: Compute PSF widths across z-range
# =============================================================================
println("\n" * "="^70)
println("Part 1: Computing PSF Widths vs Z")
println("="^70)

σx_at_z = Float32[]
σy_at_z = Float32[]

for z in z_range
    αx = GaussMLE.GaussLib.compute_alpha(z - psf_model.γ, psf_model.Ax, psf_model.Bx, psf_model.d)
    αy = GaussMLE.GaussLib.compute_alpha(z + psf_model.γ, psf_model.Ay, psf_model.By, psf_model.d)

    σx = psf_model.σx₀ * sqrt(αx)
    σy = psf_model.σy₀ * sqrt(αy)

    push!(σx_at_z, σx)
    push!(σy_at_z, σy)
end

println("σx range: $(minimum(σx_at_z)) to $(maximum(σx_at_z)) pixels")
println("σy range: $(minimum(σy_at_z)) to $(maximum(σy_at_z)) pixels")

# =============================================================================
# Part 2: Compute CRLB by fitting synthetic data at each z
# =============================================================================
println("\n" * "="^70)
println("Part 2: Computing CRLB via Fitting at Each Z Position")
println("="^70)

# Create fitter
fitter = GaussMLE.GaussMLEFitter(
    psf_model = psf_model,
    device = GaussMLE.CPU(),
    iterations = 100
)

# Storage for CRLB results
crlb_x = Float32[]
crlb_y = Float32[]
crlb_z = Float32[]
empirical_std_x = Float32[]
empirical_std_y = Float32[]
empirical_std_z = Float32[]

Random.seed!(42)

println("\nProcessing z positions:")
for (idx, z_true) in enumerate(z_range)
    print("  z = $(round(z_true, digits=1)) nm... ")

    # FIXED true position for this z (slightly off-center for non-zero derivatives)
    x_true = Float32(box_size/2 + 0.3)
    y_true = Float32(box_size/2 + 0.2)

    # Compute widths at this z
    αx = GaussMLE.GaussLib.compute_alpha(z_true - psf_model.γ, psf_model.Ax, psf_model.Bx, psf_model.d)
    αy = GaussMLE.GaussLib.compute_alpha(z_true + psf_model.γ, psf_model.Ay, psf_model.By, psf_model.d)
    σx_z = psf_model.σx₀ * sqrt(αx)
    σy_z = psf_model.σy₀ * sqrt(αy)

    # Generate synthetic data at this z position with FIXED position, varying noise
    data = zeros(Float32, box_size, box_size, n_samples_per_z)

    for k in 1:n_samples_per_z
        # Generate pixels with different Poisson noise realizations
        for j in 1:box_size, i in 1:box_size
            psf_x = GaussMLE.GaussLib.integral_gaussian_1d(i, x_true, σx_z)
            psf_y = GaussMLE.GaussLib.integral_gaussian_1d(j, y_true, σy_z)
            μ = background + n_photons * psf_x * psf_y
            data[i, j, k] = Float32(rand(Poisson(μ)))
        end
    end

    # Fit the data
    results = GaussMLE.fit(fitter, data)

    # Extract mean CRLB (reported uncertainties)
    mean_x_error = mean(results.x_error)
    mean_y_error = mean(results.y_error)
    mean_z_error = mean(results.z_error)

    # Compute empirical standard deviations of ERROR (fitted - true)
    emp_x = std(results.x .- x_true)
    emp_y = std(results.y .- y_true)
    emp_z = std(results.z .- z_true)

    push!(crlb_x, mean_x_error)
    push!(crlb_y, mean_y_error)
    push!(crlb_z, mean_z_error)
    push!(empirical_std_x, emp_x)
    push!(empirical_std_y, emp_y)
    push!(empirical_std_z, emp_z)

    println("CRLB z=$(round(mean_z_error, digits=1)) nm, empirical=$(round(emp_z, digits=1)) nm")
end

# =============================================================================
# Part 3: Analyze Results
# =============================================================================
println("\n" * "="^70)
println("Part 3: Analysis of CRLB vs Empirical")
println("="^70)

# Compute ratio of empirical to CRLB
ratio_x = empirical_std_x ./ crlb_x
ratio_y = empirical_std_y ./ crlb_y
ratio_z = empirical_std_z ./ crlb_z

println("\nCRLB vs Empirical STD Ratios:")
println("  X: mean=$(round(mean(ratio_x), digits=3)), range=$(round(minimum(ratio_x), digits=2)) to $(round(maximum(ratio_x), digits=2))")
println("  Y: mean=$(round(mean(ratio_y), digits=3)), range=$(round(minimum(ratio_y), digits=2)) to $(round(maximum(ratio_y), digits=2))")
println("  Z: mean=$(round(mean(ratio_z), digits=3)), range=$(round(minimum(ratio_z), digits=2)) to $(round(maximum(ratio_z), digits=2))")

# Identify problematic z positions
bad_z_idx = findall(ratio_z .> 1.2 .|| ratio_z .< 0.8)
if !isempty(bad_z_idx)
    println("\n⚠ Z positions with poor CRLB match (ratio outside 0.8-1.2):")
    for idx in bad_z_idx
        println("  z=$(round(z_range[idx], digits=1)) nm: ratio=$(round(ratio_z[idx], digits=3))")
    end
end

# =============================================================================
# Part 4: Visualization
# =============================================================================
println("\n" * "="^70)
println("Part 4: Creating Visualizations")
println("="^70)

fig = Figure(size=(1400, 1000))

# Panel 1: PSF Widths vs Z
ax1 = Axis(fig[1, 1],
    xlabel = "Z Position (nm)",
    ylabel = "PSF Width σ (pixels)",
    title = "Astigmatic PSF Width vs Depth",
    xlabelsize = 14,
    ylabelsize = 14,
    titlesize = 16
)

lines!(ax1, z_range, σx_at_z, label="σx(z)", color=:blue, linewidth=3)
lines!(ax1, z_range, σy_at_z, label="σy(z)", color=:red, linewidth=3)
vlines!(ax1, [0.0], color=:gray, linestyle=:dash, linewidth=1.5)
axislegend(ax1, position=:lt)

# Panel 2: Localization Precision (CRLB) vs Z
ax2 = Axis(fig[1, 2],
    xlabel = "Z Position (nm)",
    ylabel = "Localization Precision (pixels)",
    title = "CRLB: Lateral Precision vs Depth",
    xlabelsize = 14,
    ylabelsize = 14,
    titlesize = 16
)

lines!(ax2, z_range, crlb_x, label="CRLB x", color=:blue, linewidth=2.5)
lines!(ax2, z_range, crlb_y, label="CRLB y", color=:red, linewidth=2.5)
scatter!(ax2, z_range, empirical_std_x, label="Empirical x", color=:blue, marker=:circle, markersize=8, alpha=0.6)
scatter!(ax2, z_range, empirical_std_y, label="Empirical y", color=:red, marker=:circle, markersize=8, alpha=0.6)
vlines!(ax2, [0.0], color=:gray, linestyle=:dash, linewidth=1.5)
axislegend(ax2, position=:lt)

# Panel 3: Axial Precision vs Z
ax3 = Axis(fig[2, 1],
    xlabel = "Z Position (nm)",
    ylabel = "Axial Precision (nm)",
    title = "CRLB: Z Precision vs Depth",
    xlabelsize = 14,
    ylabelsize = 14,
    titlesize = 16
)

lines!(ax3, z_range, crlb_z, label="CRLB z (reported)", color=:purple, linewidth=2.5)
scatter!(ax3, z_range, empirical_std_z, label="Empirical z", color=:purple, marker=:circle, markersize=8, alpha=0.6)
vlines!(ax3, [0.0], color=:gray, linestyle=:dash, linewidth=1.5)
axislegend(ax3, position=:lt)

# Panel 4: CRLB / Empirical Ratio (Should be ~1.0)
ax4 = Axis(fig[2, 2],
    xlabel = "Z Position (nm)",
    ylabel = "Empirical STD / CRLB",
    title = "CRLB Accuracy Check (should be ≈1.0)",
    xlabelsize = 14,
    ylabelsize = 14,
    titlesize = 16
)

lines!(ax4, z_range, ratio_x, label="X ratio", color=:blue, linewidth=2.5)
lines!(ax4, z_range, ratio_y, label="Y ratio", color=:red, linewidth=2.5)
lines!(ax4, z_range, ratio_z, label="Z ratio", color=:purple, linewidth=2.5, linestyle=:dash)
hlines!(ax4, [1.0], color=:green, linestyle=:dash, linewidth=2, label="Perfect match")
hlines!(ax4, [0.8, 1.2], color=:orange, linestyle=:dot, linewidth=1.5, label="±20% tolerance")
vlines!(ax4, [0.0], color=:gray, linestyle=:dash, linewidth=1.5)
axislegend(ax4, position=:rb)

# Save figure
output_file = joinpath(output_dir, "astigmatic_crlb_vs_z.png")
save(output_file, fig)

println("Figure saved to: $output_file")

# =============================================================================
# Summary
# =============================================================================
println("\n" * "="^70)
println("SUMMARY")
println("="^70)

println("\nPSF Width Behavior:")
println("  • σx and σy vary from $(round(minimum([σx_at_z; σy_at_z]), digits=2)) to $(round(maximum([σx_at_z; σy_at_z]), digits=2)) pixels")
println("  • Crossing point: z ≈ 0 (as expected for γ=$(psf_model.γ)nm)")

println("\nCRLB Accuracy (Empirical std(error)/CRLB should be ≈1.0):")
println("  • X: $(round(mean(ratio_x), digits=3)) ± $(round(std(ratio_x), digits=3)) → $(all(0.9 .< ratio_x .< 1.1) ? "✓ EXCELLENT" : all(0.8 .< ratio_x .< 1.2) ? "✓ GOOD" : "⚠ NEEDS REVIEW")")
println("  • Y: $(round(mean(ratio_y), digits=3)) ± $(round(std(ratio_y), digits=3)) → $(all(0.9 .< ratio_y .< 1.1) ? "✓ EXCELLENT" : all(0.8 .< ratio_y .< 1.2) ? "✓ GOOD" : "⚠ NEEDS REVIEW")")
println("  • Z: $(round(mean(ratio_z), digits=3)) ± $(round(std(ratio_z), digits=3)) → $(all(0.9 .< ratio_z .< 1.1) ? "✓ EXCELLENT" : all(0.8 .< ratio_z .< 1.2) ? "✓ GOOD" : "⚠ NEEDS REVIEW")")

# Overall assessment
x_ok = all(0.8 .< ratio_x .< 1.2)
y_ok = all(0.8 .< ratio_y .< 1.2)
z_ok = all(0.8 .< ratio_z .< 1.2)

if x_ok && y_ok && z_ok
    println("\n✓ All parameters achieve CRLB within 20% across full z-range!")
    println("  MLE fitting and Fisher Information calculations are working correctly.")
elseif x_ok && y_ok
    println("\n✓ X,Y parameters achieve CRLB correctly")
    println("⚠ Z parameter shows position-dependent accuracy - this may be expected")
    println("  Near focal crossings (dσ/dz ≈ 0), z-localization is fundamentally harder")
else
    println("\n⚠ Some parameters show systematic CRLB deviation")
    println("  Review convergence, initialization, or derivative calculations")
end

println("\n" * "="^70)
