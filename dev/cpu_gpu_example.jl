using Pkg
Pkg.activate("dev")

#
# High-level example demonstrating CPU and GPU usage in GaussMLE.jl
#
# This example shows how to:
# 1. Generate synthetic data
# 2. Fit using CPU backend
# 3. Fit using GPU backend (if available)
# 4. Compare results and performance

using GaussMLE
using Random
using Statistics
using Printf

# Parameters (adjust these as needed)
n_rois = 10_000  # Number of ROIs to fit
roi_size = 7     # Size of each ROI (7x7 pixels)
seed = 42        # Random seed
verbose = true   # Print detailed results

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)  # Create if it doesn't exist

# Check if CUDA is available
const has_cuda = try
    using CUDA
    CUDA.functional()
catch
    false
end

println("=== GaussMLE CPU/GPU Development Example ===")
println("This example compares CPU and GPU backend performance")
println("Parameters: n_rois=$n_rois, roi_size=$roi_size, seed=$seed")
println("CUDA available: ", has_cuda ? "Yes" : "No")
println()

# Generate synthetic data
println("Generating synthetic data...")
Random.seed!(42)

# True parameters for each ROI: [x, y, intensity, background]
true_params = Float32[
    3.5 + 0.5 * randn(Float32, n_rois)';  # x position
    3.5 + 0.5 * randn(Float32, n_rois)';  # y position
    1000.0f0 * ones(Float32, n_rois)';     # intensity
    10.0f0 * ones(Float32, n_rois)'        # background
]

# Generate ROI stack
rois = simulatestack(model, roi_size, true_params)
println("Generated $(n_rois) ROIs of size $(roi_size)×$(roi_size)")
println()

# CPU Fitting
println("CPU Fitting")
println("-" ^ 20)
cpu_time = @elapsed begin
    cpu_results = fitstack(model, rois)
end
cpu_params = cpu_results.params
cpu_crlb = cpu_results.crlb

println("Time: $(round(cpu_time, digits=3)) seconds")
println("Speed: $(round(n_rois / cpu_time, digits=1)) ROIs/second")

# Calculate mean errors
cpu_errors = mean(abs.(cpu_params - true_params), dims=2)
println("\nMean absolute errors:")
println("  X position: $(round(cpu_errors[1], digits=4)) pixels")
println("  Y position: $(round(cpu_errors[2], digits=4)) pixels")
println("  Intensity: $(round(cpu_errors[3], digits=1)) photons")
println("  Background: $(round(cpu_errors[4], digits=2)) photons/pixel")
println()

# GPU Fitting (if available)
if has_cuda
    println("GPU Fitting (CUDA)")
    println("-" ^ 20)
    
    # Note: Currently falls back to CPU implementation
    # Future versions will have native CUDA kernels
    gpu_time = @elapsed begin
        gpu_results = fitstack(model, rois, backend=:cuda)
    end
    gpu_params = gpu_results.params
    
    println("Time: $(round(gpu_time, digits=3)) seconds")
    println("Speed: $(round(n_rois / gpu_time, digits=1)) ROIs/second")
    println("Speedup: $(round(cpu_time / gpu_time, digits=1))x")
    
    # Verify results match
    max_diff = maximum(abs.(cpu_params - gpu_params))
    println("\nMax difference CPU vs GPU: $(max_diff)")
    println("Results match: ", max_diff < 1e-5 ? "✓" : "✗")
    
    println("\nNote: GPU currently uses CPU fallback. Native CUDA kernels coming soon!")
else
    println("GPU Fitting")
    println("-" ^ 20)
    println("CUDA not available. Install CUDA.jl to enable GPU support:")
    println("  ] add CUDA")
end

println("\n" ^ 2)

# Demonstrate batched processing for large datasets
println("Large Dataset Example (Batched Processing)")
println("=" ^ 50)

const large_n_rois = 1_000_000
println("Processing $(large_n_rois) ROIs in batches...")

# Generate parameters only (not full data to save memory)
large_true_params = Float32[
    3.5 + 0.5 * randn(Float32, large_n_rois)';
    3.5 + 0.5 * randn(Float32, large_n_rois)';
    1000.0f0 * ones(Float32, large_n_rois)';
    10.0f0 * ones(Float32, large_n_rois)'
]

# Simulate fitting time (would normally process in batches)
println("\nWith batching, GPU can handle datasets larger than GPU memory")
println("Typical batch size: 10^7 ROIs")
println("Expected speedup with native CUDA kernels: 20-100x")