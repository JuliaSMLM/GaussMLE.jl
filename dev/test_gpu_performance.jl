#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using GaussMLE
using GaussMLE.GaussGPU
using CUDA
using Statistics
using Printf

println("=== GPU vs CPU Performance Test ===")
println()

# Test parameters
roi_size = 7
sigma = 1.5f0

function create_test_data(n_rois)
    # Generate realistic synthetic data
    data = zeros(Float32, roi_size, roi_size, n_rois)
    
    for k in 1:n_rois
        # Random parameters within ROI bounds
        x_true = 1.5f0 + rand(Float32) * (roi_size - 3)
        y_true = 1.5f0 + rand(Float32) * (roi_size - 3)
        intensity = 500f0 + rand(Float32) * 1000f0
        bg = 5f0 + rand(Float32) * 15f0
        
        # Generate Gaussian spot
        for j in 1:roi_size
            for i in 1:roi_size
                dx = Float32(j) - x_true
                dy = Float32(i) - y_true
                gauss = intensity * exp(-(dx^2 + dy^2) / (2*sigma^2)) / (2π*sigma^2)
                data[i, j, k] = bg + gauss
            end
        end
    end
    
    # Add Poisson noise (simple approximation)
    data .= max.(data .+ 0.1f0 .* randn(Float32, size(data)), 0.1f0)
    
    return data
end

# Test different dataset sizes
test_sizes = [1000, 10000, 100000]

for n_rois in test_sizes
    println("Testing with $n_rois ROIs:")
    println("-" ^ 40)
    
    # Create test data
    data = create_test_data(n_rois)
    println("  Data size: $(size(data))")
    println("  Memory: $(round(sizeof(data) / 1e6, digits=1)) MB")
    
    # GPU test
    backend = select_backend()
    println("\nGPU Test ($(backend_name(backend))):")
    
    # First call includes compilation
    t_gpu_cold = @elapsed begin
        θ_gpu, Σ_gpu = fitstack_gpu(data, :xynb, backend)
    end
    println("  Cold time: $(round(t_gpu_cold, digits=3))s (includes compilation)")
    
    # Second call is warmed up
    t_gpu_warm = @elapsed begin
        θ_gpu2, Σ_gpu2 = fitstack_gpu(data, :xynb, backend)
    end
    println("  Warm time: $(round(t_gpu_warm, digits=3))s")
    println("  GPU rate: $(round(n_rois / t_gpu_warm, digits=0)) ROIs/sec")
    
    # CPU test
    println("\nCPU Test ($(Threads.nthreads()) threads):")
    t_cpu = @elapsed begin
        θ_cpu, Σ_cpu = fitstack(data, :xynb)
    end
    println("  CPU time: $(round(t_cpu, digits=3))s")
    println("  CPU rate: $(round(n_rois / t_cpu, digits=0)) ROIs/sec")
    
    # Compare results
    max_x_diff = maximum(abs(θ_gpu2[i].x - θ_cpu[i].x) for i in 1:n_rois)
    max_y_diff = maximum(abs(θ_gpu2[i].y - θ_cpu[i].y) for i in 1:n_rois)
    max_n_diff = maximum(abs(θ_gpu2[i].n - θ_cpu[i].n) / θ_cpu[i].n for i in 1:n_rois)
    
    println("\nAccuracy Check:")
    println("  Max x difference: $(round(max_x_diff, digits=6))")
    println("  Max y difference: $(round(max_y_diff, digits=6))")
    println("  Max intensity rel diff: $(round(max_n_diff*100, digits=3))%")
    
    # Performance summary
    speedup = t_cpu / t_gpu_warm
    println("\nPerformance Summary:")
    println("  Speedup: $(round(speedup, digits=2))x")
    println("  Throughput improvement: $(round((n_rois/t_gpu_warm)/(n_rois/t_cpu), digits=2))x")
    
    println("\n" * "=" ^ 60 * "\n")
end

# GPU memory usage
if CUDA.functional()
    total_mem = CUDA.available_memory()
    println("GPU Memory:")
    println("  Total: $(round(total_mem / 1e9, digits=1)) GB")
end