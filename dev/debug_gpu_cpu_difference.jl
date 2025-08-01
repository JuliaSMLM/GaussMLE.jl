#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using GaussMLE
using CUDA
using Statistics

println("Debugging GPU vs CPU Differences")
println("=" ^ 50)

# Create identical synthetic data that both algorithms should handle well
n_rois = 10
roi_size = 7
data = zeros(Float32, roi_size, roi_size, n_rois)

# Generate well-conditioned Gaussian spots
for k in 1:n_rois
    x_true = 3.0f0 + rand(Float32) * 1.0f0  # 3-4 range (well inside ROI)
    y_true = 3.0f0 + rand(Float32) * 1.0f0  # 3-4 range  
    intensity = 1000f0 + rand(Float32) * 500f0  # 1000-1500 photons
    bg = 10f0 + rand(Float32) * 5f0  # 10-15 background
    
    println("ROI $k: true x=$(x_true), y=$(y_true), n=$(intensity), bg=$(bg)")
    
    for i in 1:roi_size
        for j in 1:roi_size
            dx = Float32(j) - x_true
            dy = Float32(i) - y_true
            gauss = intensity * exp(-(dx^2 + dy^2) / (2*1.3f0^2)) / (2π*1.3f0^2)
            data[i, j, k] = bg + gauss
        end
    end
end

# Test CPU fitting
println("\nCPU Results:")
θ_cpu, Σ_cpu = fitstack(data, :xynb, backend=:cpu)
for i in 1:n_rois
    println("  ROI $i: x=$(θ_cpu[i].x), y=$(θ_cpu[i].y), n=$(θ_cpu[i].n), bg=$(θ_cpu[i].bg)")
end

# Test GPU fitting  
println("\nGPU Results:")
θ_gpu, Σ_gpu = fitstack(data, :xynb, backend=:gpu)
for i in 1:n_rois
    println("  ROI $i: x=$(θ_gpu[i].x), y=$(θ_gpu[i].y), n=$(θ_gpu[i].n), bg=$(θ_gpu[i].bg)")
end

# Analyze differences
println("\nDifferences (GPU - CPU):")
x_diffs = [θ_gpu[i].x - θ_cpu[i].x for i in 1:n_rois]
y_diffs = [θ_gpu[i].y - θ_cpu[i].y for i in 1:n_rois]
n_diffs = [θ_gpu[i].n - θ_cpu[i].n for i in 1:n_rois]
bg_diffs = [θ_gpu[i].bg - θ_cpu[i].bg for i in 1:n_rois]

println("  x differences: $(x_diffs)")
println("  y differences: $(y_diffs)")
println("  Max |x| diff: $(maximum(abs.(x_diffs)))")
println("  Max |y| diff: $(maximum(abs.(y_diffs)))")
println("  Max |n| diff: $(maximum(abs.(n_diffs)))")
println("  Max |bg| diff: $(maximum(abs.(bg_diffs)))")

# Check if results are reasonable
println("\nSanity checks:")
println("  CPU x range: [$(minimum(θ_cpu[i].x for i in 1:n_rois)), $(maximum(θ_cpu[i].x for i in 1:n_rois))]")
println("  GPU x range: [$(minimum(θ_gpu[i].x for i in 1:n_rois)), $(maximum(θ_gpu[i].x for i in 1:n_rois))]")
println("  CPU y range: [$(minimum(θ_cpu[i].y for i in 1:n_rois)), $(maximum(θ_cpu[i].y for i in 1:n_rois))]")
println("  GPU y range: [$(minimum(θ_gpu[i].y for i in 1:n_rois)), $(maximum(θ_gpu[i].y for i in 1:n_rois))]")

# Test with a single, perfect Gaussian
println("\n" * "=" ^ 50)
println("Testing with single perfect Gaussian:")

perfect_data = zeros(Float32, 7, 7, 1)
x_true, y_true = 4.0f0, 4.0f0
n_true, bg_true = 1000f0, 10f0

for i in 1:7
    for j in 1:7
        dx = Float32(j) - x_true
        dy = Float32(i) - y_true
        gauss = n_true * exp(-(dx^2 + dy^2) / (2*1.3f0^2)) / (2π*1.3f0^2)
        perfect_data[i, j, 1] = bg_true + gauss
    end
end

println("Perfect data center: $(perfect_data[4,4,1])")

θ_cpu_perfect, _ = fitstack(perfect_data, :xynb, backend=:cpu)
θ_gpu_perfect, _ = fitstack(perfect_data, :xynb, backend=:gpu)

println("True:      x=$x_true, y=$y_true, n=$n_true, bg=$bg_true")
println("CPU fit:   x=$(θ_cpu_perfect[1].x), y=$(θ_cpu_perfect[1].y), n=$(θ_cpu_perfect[1].n), bg=$(θ_cpu_perfect[1].bg)")
println("GPU fit:   x=$(θ_gpu_perfect[1].x), y=$(θ_gpu_perfect[1].y), n=$(θ_gpu_perfect[1].n), bg=$(θ_gpu_perfect[1].bg)")
println("CPU error: x=$(abs(θ_cpu_perfect[1].x - x_true)), y=$(abs(θ_cpu_perfect[1].y - y_true))")
println("GPU error: x=$(abs(θ_gpu_perfect[1].x - x_true)), y=$(abs(θ_gpu_perfect[1].y - y_true))")
println("Difference: x=$(abs(θ_gpu_perfect[1].x - θ_cpu_perfect[1].x)), y=$(abs(θ_gpu_perfect[1].y - θ_cpu_perfect[1].y))")