"""
Test what simple_initialize_common actually computes for our ROI
"""

using Pkg
Pkg.activate("dev")

using GaussMLE
using Statistics
using Random
using Distributions: Poisson

println("="^70)
println("Testing Initialization Logic")
println("="^70)

function test_init()
# Create same ROI as debug script
psf_model = AstigmaticXYZNB{Float32}(1.3f0, 1.3f0, 0.05f0, -0.05f0, 0.01f0, -0.01f0, 200.0f0, 500.0f0)
box_size = 15
x_true = 7.8f0
y_true = 7.7f0
z_true = 0.0f0
n_true = 2000.0f0
bg_true = 1.0f0

# Generate ROI
Random.seed!(42)
roi = zeros(Float32, box_size, box_size)
params_true = Float32[x_true, y_true, z_true, n_true, bg_true]

for j in 1:box_size, i in 1:box_size
    μ = GaussMLE._evaluate_psf_pixel(psf_model, i, j, params_true)
    roi[i, j] = Float32(rand(Poisson(μ)))
end

println("\nROI Stats:")
println("  Sum: $(sum(roi))")
println("  Max: $(maximum(roi)) at $(argmax(roi))")
println("  Min: $(minimum(roi))")

# Manually compute initialization following simple_initialize_common logic
edge_sum = 0.0f0
edge_count = 0
for j in 1:box_size, i in 1:box_size
    if i == 1 || i == box_size || j == 1 || j == box_size
        edge_sum += roi[i,j]
        edge_count += 1
    end
end
bg_init = edge_sum / edge_count

println("\nBackground Estimation:")
println("  Edge sum: $edge_sum, count: $edge_count")
println("  bg_init = $bg_init (true bg = $bg_true)")

# Center of mass
total_signal = 0.0f0
x_weighted = 0.0f0
y_weighted = 0.0f0

for j in 1:box_size, i in 1:box_size
    signal = max(0.0f0, roi[i,j] - bg_init)
    total_signal += signal
    x_weighted += signal * Float32(j)  # j for x (column)
    y_weighted += signal * Float32(i)  # i for y (row)
end

println("\nCenter of Mass:")
println("  Total signal: $total_signal")
println("  x_weighted: $x_weighted")
println("  y_weighted: $y_weighted")

x_init = total_signal > 10 ? x_weighted / total_signal : (box_size + 1) / 2
y_init = total_signal > 10 ? y_weighted / total_signal : (box_size + 1) / 2

println("  x_init (before clamp): $x_init (true: $x_true)")
println("  y_init (before clamp): $y_init (true: $y_true)")

x_init = clamp(x_init, 1.5f0, box_size - 0.5f0)
y_init = clamp(y_init, 1.5f0, box_size - 0.5f0)

println("  x_init (after clamp): $x_init")
println("  y_init (after clamp): $y_init")

println("\nErrors:")
println("  Δx = $(x_init - x_true)")
println("  Δy = $(y_init - y_true)")

println("\n" * "="^70)
end

test_init()
