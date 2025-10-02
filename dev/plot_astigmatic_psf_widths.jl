"""
Visualization of Astigmatic PSF Width vs Z Position

This script plots σx(z) and σy(z) for the astigmatic PSF model
over the range -1 to +1 micron to understand the physical behavior
and verify our calibration parameters match realistic astigmatism.
"""

using Pkg
Pkg.activate("dev")

using GaussMLE
using CairoMakie

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("="^70)
println("Astigmatic PSF Width Visualization")
println("="^70)

# NEW calibration parameters (realistic, with aberrations)
σx₀ = 1.3f0  # pixels
σy₀ = 1.3f0  # pixels
Ax = 0.5f0   # cubic term x
Ay = -0.5f0  # cubic term y (opposite sign!)
Bx = 0.1f0   # quartic term x
By = -0.1f0  # quartic term y (opposite sign!)
γ = 0.0f0    # focal plane offset (nm)
d = 500.0f0  # depth scale (nm)

println("\nCalibration Parameters:")
println("  σx₀ = $σx₀ pixels")
println("  σy₀ = $σy₀ pixels")
println("  Ax = $Ax, Ay = $Ay (cubic aberrations)")
println("  Bx = $Bx, By = $By (quartic aberrations)")
println("  γ = $γ nm (focal plane offset)")
println("  d = $d nm (depth scale)")

# Z range: -1000 to +1000 nm
z_values = range(-1000.0f0, 1000.0f0, length=200)

# Compute σx and σy using actual GaussLib functions
σx_values = Float32[]
σy_values = Float32[]

for z in z_values
    # Use the actual compute_alpha function from GaussLib
    αx = GaussMLE.GaussLib.compute_alpha(z - γ, Ax, Bx, d)
    αy = GaussMLE.GaussLib.compute_alpha(z + γ, Ay, By, d)

    # Compute widths
    σx = σx₀ * sqrt(αx)
    σy = σy₀ * sqrt(αy)

    push!(σx_values, σx)
    push!(σy_values, σy)
end

println("\nPSF Width Statistics:")
println("  σx range: $(minimum(σx_values)) to $(maximum(σx_values)) pixels")
println("  σy range: $(minimum(σy_values)) to $(maximum(σy_values)) pixels")
println("  Max width / min width: $(maximum(σx_values)/minimum(σx_values))×")

# Find crossing point (where σx ≈ σy)
crossing_idx = argmin(abs.(σx_values .- σy_values))
crossing_z = z_values[crossing_idx]
println("  Crossing point: z ≈ $(crossing_z) nm (should be ~0)")

# Create visualization
fig = Figure(size=(1000, 700))

# Main plot
ax = Axis(fig[1, 1],
    xlabel = "Z Position (nm)",
    ylabel = "PSF Width σ (pixels)",
    title = "Astigmatic PSF Width vs Depth\n(Ax=$Ax, Ay=$Ay, Bx=$Bx, By=$By, d=$d nm)",
    xlabelsize = 16,
    ylabelsize = 16,
    titlesize = 18
)

# Plot σx and σy
lines!(ax, z_values, σx_values, label="σx(z)", color=:blue, linewidth=3)
lines!(ax, z_values, σy_values, label="σy(z)", color=:red, linewidth=3)

# Mark focal plane
vlines!(ax, [0.0], color=:gray, linestyle=:dash, linewidth=2, label="Focal plane (z=0)")

# Mark diffraction limit
hlines!(ax, [σx₀], color=:gray, linestyle=:dot, linewidth=2, label="Base width σ₀")

# Add legend
axislegend(ax, position=:lt, labelsize=14)

# Add grid
ax.xgridvisible = true
ax.ygridvisible = true

# Add annotations
text!(ax, 200, maximum(σx_values)*0.95,
    text="σx grows with +z\n(focus in y)",
    color=:blue, fontsize=12)
text!(ax, -800, maximum(σy_values)*0.95,
    text="σy grows with -z\n(focus in x)",
    color=:red, fontsize=12)

# Physical interpretation panel
ax2 = Axis(fig[2, 1],
    xlabel = "Z Position (nm)",
    ylabel = "α(z) = σ²/σ₀²",
    title = "Alpha Functions (Width Scaling)",
    xlabelsize = 14,
    ylabelsize = 14
)

# Compute and plot alpha values
αx_values = [(σx/σx₀)^2 for σx in σx_values]
αy_values = [(σy/σy₀)^2 for σy in σy_values]

lines!(ax2, z_values, αx_values, label="αx(z)", color=:blue, linewidth=2)
lines!(ax2, z_values, αy_values, label="αy(z)", color=:red, linewidth=2)
hlines!(ax2, [1.0], color=:gray, linestyle=:dash, linewidth=1)
vlines!(ax2, [0.0], color=:gray, linestyle=:dash, linewidth=1)

axislegend(ax2, position=:lt, labelsize=12)

# Save figure
output_file = joinpath(output_dir, "astigmatic_psf_widths.png")
save(output_file, fig)

println("\n" * "="^70)
println("Figure saved to: $output_file")
println("="^70)

# Analyze the behavior
println("\nPhysical Interpretation:")
println("  • At z=0 (focal plane): σx ≈ σy ≈ $(σx_values[crossing_idx]) pixels")
println("  • As z increases (+): σx grows, σy shrinks → elongated in x")
println("  • As z decreases (-): σy grows, σx shrinks → elongated in y")
println("  • This creates the characteristic 'X' pattern of astigmatic PSFs")

# Check if behavior is realistic
max_width = max(maximum(σx_values), maximum(σy_values))
min_width = min(minimum(σx_values), minimum(σy_values))
width_ratio = max_width / min_width

println("\nRealism Check:")
if width_ratio < 2.5
    println("  ✓ Width variation ($width_ratio×) is reasonable for ±1μm range")
elseif width_ratio < 4.0
    println("  ⚠ Width variation ($width_ratio×) is large but possible")
else
    println("  ✗ Width variation ($width_ratio×) seems excessive!")
end

if abs(crossing_z) < 50
    println("  ✓ Crossing point at z=$crossing_z nm is near focal plane")
else
    println("  ⚠ Crossing point at z=$crossing_z nm is offset from focus")
end

# Check symmetry
σx_at_plus500 = σx_values[findfirst(z -> z ≈ 500, z_values)]
σy_at_minus500 = σy_values[findfirst(z -> z ≈ -500, z_values)]
asymmetry = abs(σx_at_plus500 - σy_at_minus500)

if asymmetry < 0.05
    println("  ✓ Symmetric behavior: σx(+500) ≈ σy(-500)")
else
    println("  ⚠ Asymmetric: σx(+500) = $σx_at_plus500, σy(-500) = $σy_at_minus500")
end

println()
