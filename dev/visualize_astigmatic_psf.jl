"""
Visual validation of Astigmatic PSF orientation and behavior

Generates PSF images at different z positions and plots σx(z), σy(z)
to verify:
1. Correct PSF width evolution with z
2. Proper image orientation (origin at top-left, y down)
3. Astigmatic behavior (elongation rotates with z)
"""

using Pkg
Pkg.activate("dev")

using GaussMLE
using CairoMakie
using Statistics

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("="^70)
println("Astigmatic PSF Visual Validation")
println("="^70)

# Use EXACT test parameters
psf_model = GaussMLE.AstigmaticXYZNB{Float32}(
    1.3f0, 1.3f0,     # σx₀, σy₀
    0.05f0, -0.05f0,  # Ax, Ay (opposite signs)
    0.01f0, -0.01f0,  # Bx, By (opposite signs)
    200.0f0,          # γ = 200nm
    500.0f0           # d = 500nm
)

println("\nPSF Model Parameters:")
println("  σx₀ = $(psf_model.σx₀), σy₀ = $(psf_model.σy₀)")
println("  Ax = $(psf_model.Ax), Ay = $(psf_model.Ay)")
println("  γ = $(psf_model.γ) nm, d = $(psf_model.d) nm")

# Parameters for visualization
box_size = 15
n_photons = 2000.0f0
background = 1.0f0
x_pos = Float32(box_size/2 + 0.3)  # Slightly off-center
y_pos = Float32(box_size/2 + 0.2)

# Z positions to visualize (including extremes and z=0)
z_positions = Float32[-600, -300, 0, 300, 600]
n_z = length(z_positions)

println("\nGenerating PSFs at z positions: $(z_positions) nm")
println("  Position: x=$(x_pos), y=$(y_pos)")
println("  Photons: $(n_photons), Background: $(background)")

# Generate PSF images at each z
psf_images = zeros(Float32, box_size, box_size, n_z)
σx_values = Float32[]
σy_values = Float32[]

for (idx, z) in enumerate(z_positions)
    # Compute widths at this z
    αx = GaussMLE.GaussLib.compute_alpha(z - psf_model.γ, psf_model.Ax, psf_model.Bx, psf_model.d)
    αy = GaussMLE.GaussLib.compute_alpha(z + psf_model.γ, psf_model.Ay, psf_model.By, psf_model.d)
    σx = psf_model.σx₀ * sqrt(αx)
    σy = psf_model.σy₀ * sqrt(αy)

    push!(σx_values, σx)
    push!(σy_values, σy)

    println("  z=$(z) nm: σx=$(round(σx, digits=3)), σy=$(round(σy, digits=3))")

    # Generate PSF image (no noise for clean visualization)
    for j in 1:box_size, i in 1:box_size
        psf_x = GaussMLE.GaussLib.integral_gaussian_1d(j, x_pos, σx)
        psf_y = GaussMLE.GaussLib.integral_gaussian_1d(i, y_pos, σy)
        μ = background + n_photons * psf_x * psf_y
        psf_images[i, j, idx] = μ
    end
end

# Compute σ vs z for full range
z_range = Float32.(range(-600, 600, length=100))
σx_curve = similar(z_range)
σy_curve = similar(z_range)

for (idx, z) in enumerate(z_range)
    αx = GaussMLE.GaussLib.compute_alpha(z - psf_model.γ, psf_model.Ax, psf_model.Bx, psf_model.d)
    αy = GaussMLE.GaussLib.compute_alpha(z + psf_model.γ, psf_model.Ay, psf_model.By, psf_model.d)
    σx_curve[idx] = psf_model.σx₀ * sqrt(αx)
    σy_curve[idx] = psf_model.σy₀ * sqrt(αy)
end

println("\n" * "="^70)
println("Creating Visualization")
println("="^70)

# Create figure with PSF widths and images
fig = Figure(size=(1600, 900))

# Top panel: σx and σy vs z
ax_widths = Axis(fig[1, 1:n_z],
    xlabel = "Z Position (nm)",
    ylabel = "PSF Width σ (pixels)",
    title = "Astigmatic PSF Width Evolution",
    xlabelsize = 16,
    ylabelsize = 16,
    titlesize = 18
)

lines!(ax_widths, z_range, σx_curve, label="σx(z)", color=:blue, linewidth=3)
lines!(ax_widths, z_range, σy_curve, label="σy(z)", color=:red, linewidth=3)
scatter!(ax_widths, z_positions, σx_values, color=:blue, markersize=15)
scatter!(ax_widths, z_positions, σy_values, color=:red, markersize=15)
vlines!(ax_widths, [0.0], color=:gray, linestyle=:dash, linewidth=2)
axislegend(ax_widths, position=:lt, labelsize=14)

# Bottom panels: PSF images at each z position
# CRITICAL: Images must display with origin at (1,1) top-left, y going down
# For CairoMakie: transpose data and set yreversed=true
for (idx, z) in enumerate(z_positions)
    ax_img = Axis(fig[2, idx],
        title = "z = $(Int(z)) nm\nσx=$(round(σx_values[idx], digits=2)), σy=$(round(σy_values[idx], digits=2))",
        aspect = DataAspect(),
        titlesize = 12,
        yreversed = true  # Origin at top-left, y down
    )

    # Transpose image: data[i,j] → display[j,i] so rows become columns
    # Combined with yreversed, this gives proper orientation:
    # data[1,1] → top-left, data[1,end] → top-right, data[end,1] → bottom-left
    img_data = psf_images[:, :, idx]'  # Transpose

    heatmap!(ax_img, img_data, colormap=:hot)

    # Add crosshair at PSF center
    vlines!(ax_img, [x_pos], color=:cyan, linewidth=1, linestyle=:dash)
    hlines!(ax_img, [y_pos], color=:cyan, linewidth=1, linestyle=:dash)

    # Show coordinate convention
    if idx == 1
        text!(ax_img, 1, 1, text="(1,1)\ntop-left",
              align=(:left, :top), color=:white, fontsize=10)
    end
end

# Add annotation about coordinate convention
Label(fig[3, 1:n_z],
      "Image Convention: Origin at (1,1) top-left, x→right (columns), y↓down (rows). Data transposed + yreversed for display.",
      fontsize = 11, color = :gray50)

# Save figure
output_file = joinpath(output_dir, "astigmatic_psf_visual_validation.png")
save(output_file, fig)

println("\n✓ Figure saved to: $output_file")

# Analysis
println("\n" * "="^70)
println("ANALYSIS")
println("="^70)

println("\nPSF Width Behavior:")
println("  σx range: $(round(minimum(σx_values), digits=3)) to $(round(maximum(σx_values), digits=3)) pixels")
println("  σy range: $(round(minimum(σy_values), digits=3)) to $(round(maximum(σy_values), digits=3)) pixels")

# Check if widths cross (they should around z=0)
crossing_idx = argmin(abs.(σx_curve .- σy_curve))
crossing_z = z_range[crossing_idx]
println("  Crossing point: z ≈ $(round(crossing_z, digits=1)) nm (expected: 0 nm)")

# Check which dimension elongates with z
if σx_values[end] > σx_values[1]
    println("\n✓ σx INCREASES with positive z")
    println("  → Astigmatism formula: σx ∝ sqrt(1 + ((z-γ)/d)²) ✓")
else
    println("\n⚠ σx DECREASES with positive z (unexpected!)")
end

if σy_values[end] > σy_values[1]
    println("✓ σy INCREASES with positive z")
    println("  → Astigmatism formula: σy ∝ sqrt(1 + ((z+γ)/d)²) ✓")
else
    println("⚠ σy DECREASES with positive z (unexpected!)")
end

# Check orientation in images
println("\nExpected Visual Behavior:")
println("  At z = -600 nm: PSF should be NARROWER in one direction")
println("  At z = 0 nm:    PSF should be most CIRCULAR (σx ≈ σy)")
println("  At z = +600 nm: PSF should be NARROWER in opposite direction")
println("\nWith current convention (j=x=horizontal, i=y=vertical):")
println("  σx controls HORIZONTAL width (x-direction, left-right)")
println("  σy controls VERTICAL width (y-direction, up-down)")
println("  → At positive z: σx > σy means HORIZONTAL elongation")
println("  → At negative z: σy > σx means VERTICAL elongation")

println("\n" * "="^70)
println("Review the saved image to verify PSF orientation matches theory!")
println("="^70)
