#!/usr/bin/env julia
#
# Test the fixed CUDA kernel implementation

using GaussMLE
using GaussMLE.GaussGPU: launch_cuda_fit_kernel!
using CUDA
using Random
using Printf
using Statistics

println("Testing Fixed CUDA Kernel")
println("=" ^ 60)

# Test parameters
roi_size = 7
n_tests = 5
Random.seed!(42)

# Generate test cases with known parameters
test_params = [
    (x=3.5f0, y=3.5f0, n=1000f0, bg=10f0),    # Center
    (x=2.0f0, y=2.0f0, n=800f0, bg=15f0),     # Off-center
    (x=5.0f0, y=5.0f0, n=1200f0, bg=8f0),     # Near edge
    (x=1.5f0, y=4.5f0, n=600f0, bg=20f0),     # Asymmetric
    (x=4.0f0, y=3.0f0, n=1500f0, bg=5f0)      # Bright spot
]

sigma = 1.5f0
errors_gpu = []
errors_cpu = []

for (i, params) in enumerate(test_params)
    println("\nTest $i: True params: x=$(params.x), y=$(params.y), n=$(params.n), bg=$(params.bg)")
    
    # Generate synthetic ROI
    roi = zeros(Float32, roi_size, roi_size)
    for i in 1:roi_size
        for j in 1:roi_size
            # Note: model uses 0-based coordinates internally
            dx = Float32(i - 1) - params.x
            dy = Float32(j - 1) - params.y
            gauss = params.n * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
            roi[i, j] = params.bg + gauss
        end
    end
    
    # Add realistic Poisson noise
    using Distributions
    for idx in 1:roi_size*roi_size
        roi[idx] = Float32(rand(Poisson(roi[idx])))
    end
    
    # Show ROI center region
    println("  ROI center (3x3):")
    for i in 3:5
        print("    ")
        for j in 3:5
            @printf("%6.1f ", roi[i,j])
        end
        println()
    end
    
    # Test GPU kernel
    d_data = CuArray(reshape(roi, roi_size, roi_size, 1))
    d_params = CUDA.zeros(Float32, 4, 1)
    d_crlb = CUDA.zeros(Float32, 4, 1)
    
    launch_cuda_fit_kernel!(d_data, d_params, d_crlb, Int32(roi_size))
    CUDA.synchronize()
    
    params_gpu = Array(d_params)
    
    # Test CPU version
    args = GaussMLE.GaussModel.Args_xynb(sigma)
    θ_cpu, Σ_cpu = GaussMLE.GaussFit.fitstack(reshape(roi, roi_size, roi_size, 1), :xynb, args)
    
    # Calculate errors
    error_x_gpu = abs(params_gpu[1,1] - params.x)
    error_y_gpu = abs(params_gpu[2,1] - params.y)
    error_n_gpu = abs(params_gpu[3,1] - params.n) / params.n * 100
    error_bg_gpu = abs(params_gpu[4,1] - params.bg)
    
    error_x_cpu = abs(θ_cpu[1].x - params.x)
    error_y_cpu = abs(θ_cpu[1].y - params.y)
    error_n_cpu = abs(θ_cpu[1].n - params.n) / params.n * 100
    error_bg_cpu = abs(θ_cpu[1].bg - params.bg)
    
    push!(errors_gpu, (x=error_x_gpu, y=error_y_gpu, n=error_n_gpu, bg=error_bg_gpu))
    push!(errors_cpu, (x=error_x_cpu, y=error_y_cpu, n=error_n_cpu, bg=error_bg_cpu))
    
    println("  GPU fit: x=$(params_gpu[1,1]), y=$(params_gpu[2,1]), n=$(params_gpu[3,1]), bg=$(params_gpu[4,1])")
    println("  CPU fit: x=$(θ_cpu[1].x), y=$(θ_cpu[1].y), n=$(θ_cpu[1].n), bg=$(θ_cpu[1].bg)")
    println("  GPU errors: x=$(@sprintf("%.3f", error_x_gpu)), y=$(@sprintf("%.3f", error_y_gpu)), n=$(@sprintf("%.1f", error_n_gpu))%, bg=$(@sprintf("%.2f", error_bg_gpu))")
    println("  CPU errors: x=$(@sprintf("%.3f", error_x_cpu)), y=$(@sprintf("%.3f", error_y_cpu)), n=$(@sprintf("%.1f", error_n_cpu))%, bg=$(@sprintf("%.2f", error_bg_cpu))")
end

# Summary statistics
println("\n" * ("=" ^ 60))
println("Summary Statistics:")
println("=" * 60)

avg_error_x_gpu = mean(e.x for e in errors_gpu)
avg_error_y_gpu = mean(e.y for e in errors_gpu)
avg_error_n_gpu = mean(e.n for e in errors_gpu)
avg_error_bg_gpu = mean(e.bg for e in errors_gpu)

avg_error_x_cpu = mean(e.x for e in errors_cpu)
avg_error_y_cpu = mean(e.y for e in errors_cpu)
avg_error_n_cpu = mean(e.n for e in errors_cpu)
avg_error_bg_cpu = mean(e.bg for e in errors_cpu)

println("\nAverage Errors:")
println("  GPU: x=$(@sprintf("%.3f", avg_error_x_gpu)), y=$(@sprintf("%.3f", avg_error_y_gpu)), n=$(@sprintf("%.1f", avg_error_n_gpu))%, bg=$(@sprintf("%.2f", avg_error_bg_gpu))")
println("  CPU: x=$(@sprintf("%.3f", avg_error_x_cpu)), y=$(@sprintf("%.3f", avg_error_y_cpu)), n=$(@sprintf("%.1f", avg_error_n_cpu))%, bg=$(@sprintf("%.2f", avg_error_bg_cpu))")

# Test with larger dataset for performance
println("\n" * ("=" ^ 60))
println("Performance Test (1000 ROIs):")
println("=" * 60)

n_rois = 1000
data_batch = zeros(Float32, roi_size, roi_size, n_rois)

# Generate batch of ROIs
for k in 1:n_rois
    x_true = rand() * (roi_size - 2) + 1
    y_true = rand() * (roi_size - 2) + 1
    n_true = 500 + rand() * 1000
    bg_true = 5 + rand() * 15
    
    for i in 1:roi_size
        for j in 1:roi_size
            dx = Float32(i - 1) - x_true
            dy = Float32(j - 1) - y_true
            gauss = n_true * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
            data_batch[i, j, k] = Float32(rand(Poisson(bg_true + gauss)))
        end
    end
end

# Time GPU
d_data = CuArray(data_batch)
d_params = CUDA.zeros(Float32, 4, n_rois)
d_crlb = CUDA.zeros(Float32, 4, n_rois)

CUDA.@sync begin
    t_gpu = @elapsed begin
        launch_cuda_fit_kernel!(d_data, d_params, d_crlb, Int32(roi_size))
        CUDA.synchronize()
    end
end

# Time CPU
args = GaussMLE.GaussModel.Args_xynb(sigma)
t_cpu = @elapsed begin
    θ_cpu, Σ_cpu = GaussMLE.GaussFit.fitstack(data_batch, :xynb, args)
end

println("GPU time: $(@sprintf("%.3f", t_gpu * 1000)) ms")
println("CPU time: $(@sprintf("%.3f", t_cpu * 1000)) ms")
println("Speedup: $(@sprintf("%.1f", t_cpu / t_gpu))x")

# Check that GPU results are reasonable
params_gpu = Array(d_params)
n_valid = sum(0 < params_gpu[1,i] < roi_size && 
              0 < params_gpu[2,i] < roi_size && 
              params_gpu[3,i] > 0 && 
              params_gpu[4,i] > 0 
              for i in 1:n_rois)

println("\nValid fits: $n_valid / $n_rois ($(@sprintf("%.1f", n_valid/n_rois*100))%)")

# Check convergence patterns
x_positions = params_gpu[1,:]
y_positions = params_gpu[2,:]
println("X position range: [$(@sprintf("%.2f", minimum(x_positions))), $(@sprintf("%.2f", maximum(x_positions)))]")
println("Y position range: [$(@sprintf("%.2f", minimum(y_positions))), $(@sprintf("%.2f", maximum(y_positions)))]")