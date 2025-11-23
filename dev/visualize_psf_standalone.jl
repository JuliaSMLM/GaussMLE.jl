"""
Standalone visual validation of Astigmatic PSF
(No package dependencies - uses source code directly)
"""

using Pkg
Pkg.activate("dev")

using CairoMakie
using SpecialFunctions: erf  # Needed for GaussLib

# Include only what we need from source
include("../src/gausslib/baselibrary.jl")

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("="^70)
println("Astigmatic PSF Visual Validation (Standalone)")
println("="^70)

# Test parameters (EXACT from test suite)
σx₀ = 1.3f0
σy₀ = 1.3f0
Ax = 0.05f0
Ay = -0.05f0
Bx = 0.01f0
By = -0.01f0
γ = 200.0f0  # nm
d = 500.0f0  # nm

println("\nPSF Parameters:")
println("  σx₀ = $σx₀, σy₀ = $σy₀")
println("  Ax = $Ax, Ay = $Ay")
println("  γ = $γ nm, d = $d nm")

# Visualization parameters
box_size = 15
n_photons = 2000.0f0
background = 1.0f0
x_pos = Float32(box_size/2 + 0.3)
y_pos = Float32(box_size/2 + 0.2)

# Z positions to visualize
z_positions = Float32[-600, -300, 0, 300, 600]
n_z = length(z_positions)

println("\nGenerating PSFs at z = $(z_positions) nm")
println("  Center: x=$x_pos, y=$y_pos")

# Generate PSF images
psf_images = zeros(Float32, box_size, box_size, n_z)
σx_values = Float32[]
σy_values = Float32[]

for (idx, z) in enumerate(z_positions)
    # Compute widths using astigmatic formula
    αx = compute_alpha(z - γ, Ax, Bx, d)
    αy = compute_alpha(z + γ, Ay, By, d)
    σx = σx₀ * sqrt(αx)
    σy = σy₀ * sqrt(αy)

    push!(σx_values, σx)
    push!(σy_values, σy)

    println("  z=$z nm: σx=$(round(σx, digits=3)), σy=$(round(σy, digits=3))")

    # Generate PSF image
    for j in 1:box_size, i in 1:box_size
        psf_x = integral_gaussian_1d(j, x_pos, σx)
        psf_y = integral_gaussian_1d(i, y_pos, σy)
        μ = background + n_photons * psf_x * psf_y
        psf_images[i, j, idx] = μ
    end
end

# Compute continuous curves
z_range = Float32.(range(-600, 600, length=100))
σx_curve = [σx₀ * sqrt(compute_alpha(z - γ, Ax, Bx, d)) for z in z_range]
σy_curve = [σy₀ * sqrt(compute_alpha(z + γ, Ay, By, d)) for z in z_range]

println("\n" * "="^70)
println("Creating Visualization")
println("="^70)

# Create figure
fig = Figure(size=(1600, 900))

# Top: Width curves
ax_widths = Axis(fig[1, 1:n_z],
    xlabel = "Z Position (nm)",
    ylabel = "PSF Width σ (pixels)",
    title = "Astigmatic PSF: σx and σy vs Z",
    xlabelsize = 16,
    ylabelsize = 16,
    titlesize = 18
)

lines!(ax_widths, z_range, σx_curve, label="σx(z) [HORIZONTAL width]", color=:blue, linewidth=3)
lines!(ax_widths, z_range, σy_curve, label="σy(z) [VERTICAL width]", color=:red, linewidth=3)
scatter!(ax_widths, z_positions, σx_values, color=:blue, markersize=15)
scatter!(ax_widths, z_positions, σy_values, color=:red, markersize=15)
vlines!(ax_widths, [0.0], color=:gray, linestyle=:dash, linewidth=2, label="z=0")
axislegend(ax_widths, position=:lt, labelsize=14)

# Bottom: PSF images
# CRITICAL: Origin at (1,1) top-left, y goes down
# Method: Transpose data + yreversed=true
for (idx, z) in enumerate(z_positions)
    ax_img = Axis(fig[2, idx],
        title = "z = $(Int(z)) nm\nσx=$(round(σx_values[idx], digits=2)) σy=$(round(σy_values[idx], digits=2))",
        aspect = DataAspect(),
        titlesize = 12,
        yreversed = true  # Y axis goes DOWN (origin at top)
    )

    # Transpose: data[i,j] (i=row=y, j=col=x) → display[j,i] (j=x, i=y)
    # With yreversed: data[1,1] appears at top-left ✓
    img_data = psf_images[:, :, idx]'  # Transpose

    heatmap!(ax_img, img_data, colormap=:hot)

    # Crosshair at center
    vlines!(ax_img, [x_pos], color=:cyan, linewidth=1, linestyle=:dash)
    hlines!(ax_img, [y_pos], color=:cyan, linewidth=1, linestyle=:dash)

    # Mark origin on first image
    if idx == 1
        text!(ax_img, 1.5, 1.5, text="(1,1)\norigin",
              align=(:left, :top), color=:white, fontsize=10)
    end

    # Mark elongation direction
    if σx_values[idx] > σy_values[idx] + 0.2
        text!(ax_img, box_size/2, box_size-1,
              text="HORIZONTAL\nelongation\n(σx > σy)",
              align=(:center, :bottom), color=:yellow, fontsize=9)
    elseif σy_values[idx] > σx_values[idx] + 0.2
        text!(ax_img, box_size/2, box_size-1,
              text="VERTICAL\nelongation\n(σy > σx)",
              align=(:center, :bottom), color=:yellow, fontsize=9)
    end
end

# Convention note
Label(fig[3, 1:n_z],
      "Convention: data[i,j] with i=row (y-axis), j=column (x-axis). Display: transposed + yreversed → origin (1,1) at top-left.",
      fontsize = 11, color = :gray50)

# Save
output_file = joinpath(output_dir, "astigmatic_psf_standalone.png")
save(output_file, fig)
println("\n✓ Saved: $output_file")

# Analysis
println("\n" * "="^70)
println("ANALYSIS")
println("="^70)

crossing_idx = argmin(abs.(σx_curve .- σy_curve))
crossing_z = z_range[crossing_idx]

println("\nWidth Behavior:")
println("  σx: $(round(minimum(σx_values), digits=2)) → $(round(maximum(σx_values), digits=2)) pixels")
println("  σy: $(round(minimum(σy_values), digits=2)) → $(round(maximum(σy_values), digits=2)) pixels")
println("  Crossing: z ≈ $(round(crossing_z, digits=1)) nm (expected 0)")

println("\nCurrent Convention (j=x horizontal, i=y vertical):")
println("  σx = HORIZONTAL width (x-direction, left-right)")
println("  σy = VERTICAL width (y-direction, up-down)")

println("\nExpected Behavior:")
at_neg = findfirst(z_positions .== -600)
at_pos = findfirst(z_positions .== 600)
if σx_values[at_pos] > σy_values[at_pos]
    println("  ✓ At z=+600: σx > σy → HORIZONTAL elongation")
else
    println("  ⚠ At z=+600: σy > σx → VERTICAL elongation (unexpected!)")
end
if σy_values[at_neg] > σx_values[at_neg]
    println("  ✓ At z=-600: σy > σx → VERTICAL elongation")
else
    println("  ⚠ At z=-600: σx > σy → HORIZONTAL elongation (unexpected!)")
end

println("\n" * "="^70)
println("Review image to verify PSF shapes match σ values!")
println("="^70)
