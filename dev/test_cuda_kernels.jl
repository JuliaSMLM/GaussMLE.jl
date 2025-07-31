#!/usr/bin/env julia
#
# Test script for CUDA kernel implementation

using GaussMLE
using GaussMLE: fitstack_gpu, CUDABackend, CPUBackend
using CUDA
using Random
using Statistics
using Printf

println("CUDA Kernel Test")
println("=" ^ 50)

# Check CUDA availability
if !CUDA.functional()
    println("ERROR: CUDA not available")
    exit(1)
end

println("GPU: ", CUDA.name(CUDA.device()))
println()

# Test parameters
const test_sizes = [100, 1000, 10000]
const roi_size = 7
const model_symbol = :xynb

for n_rois in test_sizes
    println("\nTesting with $n_rois ROIs")
    println("-" ^ 30)
    
    # Generate synthetic data
    Random.seed!(42)
    data, θ_true, args = GaussMLE.GaussSim.genstack(roi_size, n_rois, model_symbol; 
                                                    T=Float32, poissonnoise=true)
    
    # CPU reference (single-threaded for consistency)
    println("Running CPU reference...")
    cpu_time = @elapsed begin
        θ_cpu, Σ_cpu = GaussMLE.GaussFit.fitstack(data, model_symbol, args)
    end
    println("  CPU time: $(round(cpu_time * 1000, digits=1)) ms")
    println("  Speed: $(round(n_rois / cpu_time, digits=0)) ROIs/sec")
    
    # GPU with new kernels
    println("\nRunning GPU kernels...")
    gpu_time = @elapsed begin
        θ_gpu, Σ_gpu = fitstack_gpu(data, model_symbol, CUDABackend(); verbose=false)
    end
    println("  GPU time: $(round(gpu_time * 1000, digits=1)) ms")
    println("  Speed: $(round(n_rois / gpu_time, digits=0)) ROIs/sec")
    println("  Speedup: $(round(cpu_time / gpu_time, digits=1))x")
    
    # Compare results
    println("\nValidating results...")
    
    # Extract parameters for comparison
    x_cpu = [θ.x for θ in θ_cpu]
    y_cpu = [θ.y for θ in θ_cpu]
    n_cpu = [θ.n for θ in θ_cpu]
    bg_cpu = [θ.bg for θ in θ_cpu]
    
    x_gpu = [θ.x for θ in θ_gpu]
    y_gpu = [θ.y for θ in θ_gpu]
    n_gpu = [θ.n for θ in θ_gpu]
    bg_gpu = [θ.bg for θ in θ_gpu]
    
    # Compute differences
    diff_x = mean(abs.(x_cpu - x_gpu))
    diff_y = mean(abs.(y_cpu - y_gpu))
    diff_n = mean(abs.(n_cpu - n_gpu))
    diff_bg = mean(abs.(bg_cpu - bg_gpu))
    
    max_diff_x = maximum(abs.(x_cpu - x_gpu))
    max_diff_y = maximum(abs.(y_cpu - y_gpu))
    max_diff_n = maximum(abs.(n_cpu - n_gpu))
    max_diff_bg = maximum(abs.(bg_cpu - bg_gpu))
    
    println("  Mean absolute differences:")
    println("    X: $(round(diff_x, sigdigits=3)) (max: $(round(max_diff_x, sigdigits=3)))")
    println("    Y: $(round(diff_y, sigdigits=3)) (max: $(round(max_diff_y, sigdigits=3)))")
    println("    N: $(round(diff_n, sigdigits=3)) (max: $(round(max_diff_n, sigdigits=3)))")
    println("    BG: $(round(diff_bg, sigdigits=3)) (max: $(round(max_diff_bg, sigdigits=3)))")
    
    # Check if results match within tolerance
    tolerance = 0.01  # 1% of typical parameter values
    match_x = max_diff_x < tolerance * roi_size
    match_y = max_diff_y < tolerance * roi_size
    match_n = max_diff_n < tolerance * 1000  # typical intensity
    match_bg = max_diff_bg < tolerance * 10   # typical background
    
    all_match = match_x && match_y && match_n && match_bg
    
    println("\n  Result: ", all_match ? "✓ PASS" : "✗ FAIL")
    
    if !all_match
        println("\n  Failed checks:")
        !match_x && println("    - X position")
        !match_y && println("    - Y position")
        !match_n && println("    - Intensity")
        !match_bg && println("    - Background")
    end
end

# Memory usage test
println("\n\nMemory Usage Test")
println("=" ^ 50)

mem_before = CUDA.available_memory() / 1024^3
println("Available GPU memory: $(round(mem_before, digits=2)) GB")

# Test with larger dataset
n_large = 100_000
println("\nAllocating data for $n_large ROIs...")
data_large, _, args_large = GaussMLE.GaussSim.genstack(roi_size, n_large, model_symbol; 
                                                       T=Float32, poissonnoise=true)

# Run GPU fitting
println("Running GPU fitting...")
gpu_time_large = @elapsed begin
    θ_gpu_large, Σ_gpu_large = fitstack_gpu(data_large, model_symbol, CUDABackend(); verbose=false)
end

mem_after = CUDA.available_memory() / 1024^3
mem_used = mem_before - mem_after

println("\nResults:")
println("  Time: $(round(gpu_time_large, digits=2)) seconds")
println("  Speed: $(round(n_large / gpu_time_large, digits=0)) ROIs/sec")
println("  Memory used: $(round(mem_used * 1024, digits=0)) MB")
println("  Memory per ROI: $(round(mem_used * 1024^2 / n_large, digits=1)) KB")

println("\n" * ("=" ^ 50))
println("CUDA Kernel Test Complete")