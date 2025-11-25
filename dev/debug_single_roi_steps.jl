"""
Debug astigmatic fitting step-by-step on a single ROI

Shows:
1. Ground truth ROI image
2. Initial parameters (0 iterations)
3. Parameters after 1 iteration
4. Parameters after 10 iterations
5. Model PSF at each step

This reveals WHERE fitting goes wrong: initialization, gradient direction, or convergence.
"""

using Pkg
Pkg.activate("dev")

using GaussMLE
using SMLMData
using CairoMakie
using Statistics
using Random
using Distributions: Poisson

# Output
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("="^70)
println("Single ROI Step-by-Step Fitting Debug")
println("="^70)

# Test PSF at z=0 (circular PSF, simplest case)
psf_model = AstigmaticXYZNB{Float32}(
    1.3f0, 1.3f0,     # σx₀, σy₀
    0.05f0, -0.05f0,  # Ax, Ay
    0.01f0, -0.01f0,  # Bx, By
    200.0f0,          # γ
    500.0f0           # d
)

# Ground truth parameters
box_size = 15
x_true = Float32(box_size/2 + 0.3)  # 7.8
y_true = Float32(box_size/2 + 0.2)  # 7.7
z_true = 0.0f0  # At focus - σx ≈ σy (circular PSF)
n_true = 2000.0f0
bg_true = 1.0f0

println("\nGround Truth:")
println("  Position: x=$x_true, y=$y_true, z=$z_true")
println("  Photons: $n_true, Background: $bg_true")

# Compute expected widths at z=0
αx = GaussMLE.GaussLib.compute_alpha(z_true - psf_model.γ, psf_model.Ax, psf_model.Bx, psf_model.d)
αy = GaussMLE.GaussLib.compute_alpha(z_true + psf_model.γ, psf_model.Ay, psf_model.By, psf_model.d)
σx_true = psf_model.σx₀ * sqrt(αx)
σy_true = psf_model.σy₀ * sqrt(αy)
println("  At z=0: σx=$σx_true, σy=$σy_true")

# Generate single ROI using simulator
Random.seed!(42)
roi_data = zeros(Float32, box_size, box_size, 1)
params_vec = Float32[x_true, y_true, z_true, n_true, bg_true]

for j in 1:box_size, i in 1:box_size
    μ = GaussMLE._evaluate_psf_pixel(psf_model, i, j, params_vec)
    roi_data[i, j, 1] = Float32(rand(Poisson(μ)))
end

println("\nROI Statistics:")
println("  Sum: $(sum(roi_data))")
println("  Max: $(maximum(roi_data))")
println("  Min: $(minimum(roi_data))")

# Create fitters with different iteration counts
println("\n" * "="^70)
println("Fitting with Different Iteration Counts")
println("="^70)

iteration_counts = [0, 1, 10]
results_at_iters = []

pixel_size = 0.1f0  # Emitters are in microns with 100nm pixels

for n_iter in iteration_counts
    fitter = GaussMLEFitter(psf_model=psf_model, iterations=n_iter)
    result = fit(fitter, roi_data)

    # Extract fitted parameters and convert to ROI pixels for comparison
    e = result.emitters[1]

    # Convert emitter microns to ROI pixels (corner assumed at 1,1 for single ROI)
    x_fitted_roi = e.x / pixel_size + 1 - 1 + 1  # microns→camera→ROI
    y_fitted_roi = e.y / pixel_size + 1 - 1 + 1
    z_fitted = e.z / pixel_size

    println("\n[$n_iter iterations]")
    println("  x: $(round(x_fitted_roi, digits=3)) (Δ = $(round(x_fitted_roi - x_true, digits=3)), σ=$(round(e.σ_x/pixel_size, digits=3)))")
    println("  y: $(round(y_fitted_roi, digits=3)) (Δ = $(round(y_fitted_roi - y_true, digits=3)), σ=$(round(e.σ_y/pixel_size, digits=3)))")
    println("  z: $(round(z_fitted, digits=3)) (Δ = $(round(z_fitted - z_true, digits=3)), σ=$(round(e.σ_z/pixel_size, digits=3)))")
    println("  N: $(round(e.photons, digits=1)) (Δ = $(round(e.photons - n_true, digits=1)))")
    println("  bg: $(round(e.bg, digits=2)) (Δ = $(round(e.bg - bg_true, digits=2)))")

    push!(results_at_iters, (n_iter=n_iter, emitter=e, x_roi=x_fitted_roi, y_roi=y_fitted_roi))
end

# Generate model PSFs at each iteration (convert emitter values to ROI pixels)
model_psfs = []
for r in results_at_iters
    model_roi = zeros(Float32, box_size, box_size)
    # Use ROI pixel coordinates, not microns!
    fitted_params = Float32[r.x_roi, r.y_roi, r.emitter.z/pixel_size, r.emitter.photons, r.emitter.bg]

    for j in 1:box_size, i in 1:box_size
        model_roi[i, j] = GaussMLE._evaluate_psf_pixel(psf_model, i, j, fitted_params)
    end

    push!(model_psfs, model_roi)
end

# Visualize
println("\n" * "="^70)
println("Creating Visualization")
println("="^70)

fig = Figure(size=(1800, 1200))

# Row 1: Data and models
titles = ["Ground Truth\nData", "0 iter\n(Initialization)", "1 iter\n(First Step)", "10 iter\n(Converged?)"]
for (col, (title, img)) in enumerate(zip(titles, [roi_data[:,:,1], model_psfs...]))
    ax = Axis(fig[1, col],
        title = title,
        aspect = DataAspect(),
        yreversed = true,
        titlesize = 14
    )

    # Transpose for display: data[i,j] → display[j,i], origin top-left
    heatmap!(ax, img', colormap=:hot)

    # Mark true center
    vlines!(ax, [x_true], color=:cyan, linewidth=1, linestyle=:dash)
    hlines!(ax, [y_true], color=:cyan, linewidth=1, linestyle=:dash)

    # Mark fitted center if not ground truth (use ROI coordinates)
    if col > 1
        r = results_at_iters[col-1]
        scatter!(ax, [r.x_roi], [r.y_roi], color=:yellow, markersize=12, marker=:xcross)
    end
end

# Row 2: Residuals (data - model)
for col in 1:3
    ax = Axis(fig[2, col+1],
        title = "Residual\n(Data - Model)",
        aspect = DataAspect(),
        yreversed = true,
        titlesize = 12
    )

    residual = roi_data[:,:,1] - model_psfs[col]
    heatmap!(ax, residual', colormap=:RdBu, colorrange=(-20, 20))
end

# Row 3: Parameter evolution table
Label(fig[3, :],
      "Parameter Evolution:\n" *
      "Iteration | x (Δ) | y (Δ) | z (Δ) | N (Δ) | bg (Δ)\n" *
      join([let e = r.emitter, x_r = r.x_roi, y_r = r.y_roi
             "$(r.n_iter) iter | " *
             "$(round(x_r,digits=2)) ($(round(x_r-x_true,digits=2))) | " *
             "$(round(y_r,digits=2)) ($(round(y_r-y_true,digits=2))) | " *
             "$(round(e.z/pixel_size,digits=1)) ($(round(e.z/pixel_size-z_true,digits=1))) | " *
             "$(round(e.photons,digits=0)) ($(round(e.photons-n_true,digits=0))) | " *
             "$(round(e.bg,digits=2)) ($(round(e.bg-bg_true,digits=2)))"
           end for r in results_at_iters], "\n"),
      fontsize = 11,
      halign = :left
)

output_file = joinpath(output_dir, "debug_single_roi_steps.png")
save(output_file, fig)

println("\n✓ Saved: $output_file")

# Analysis
println("\n" * "="^70)
println("ANALYSIS")
println("="^70)

println("\nKey Questions:")

# Check initialization
r0 = results_at_iters[1]
if abs(r0.x_roi - x_true) < 1.0 && abs(r0.y_roi - y_true) < 1.0
    println("  ✓ Initialization: x,y within 1 pixel of truth")
else
    println("  ✗ Initialization: x,y OFF by $(round(r0.x_roi-x_true,digits=2)), $(round(r0.y_roi-y_true,digits=2)) pixels")
end

# Check first iteration
r1 = results_at_iters[2]
if abs(r1.x_roi - x_true) < abs(r0.x_roi - x_true)
    println("  ✓ First iteration: x improved")
else
    println("  ✗ First iteration: x got WORSE (Δ: $(round(r0.x_roi-x_true,digits=3)) → $(round(r1.x_roi-x_true,digits=3)))")
end

if abs(r1.y_roi - y_true) < abs(r0.y_roi - y_true)
    println("  ✓ First iteration: y improved")
else
    println("  ✗ First iteration: y got WORSE (Δ: $(round(r0.y_roi-y_true,digits=3)) → $(round(r1.y_roi-y_true,digits=3)))")
end

# Check convergence
r10 = results_at_iters[3]
if abs(r10.x_roi - x_true) < 0.1 && abs(r10.y_roi - y_true) < 0.1
    println("  ✓ After 10 iter: x,y converged to truth")
else
    println("  ✗ After 10 iter: x,y still OFF by $(round(r10.x_roi-x_true,digits=2)), $(round(r10.y_roi-y_true,digits=2))")
end

# Check residuals
for (idx, model_psf) in enumerate(model_psfs)
    residual = roi_data[:,:,1] - model_psf
    rms = sqrt(mean(residual.^2))
    println("  Residual RMS at $(iteration_counts[idx]) iter: $(round(rms, digits=2))")
end

println("\n" * "="^70)
