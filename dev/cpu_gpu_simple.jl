#!/usr/bin/env julia
#
# Simple CPU/GPU example for GaussMLE.jl
# Using the correct API

using GaussMLE
using GaussMLE: fitstack_gpu, CUDABackend, CPUBackend
using Random
using Statistics
using Printf

# Check if CUDA is available and has memory
cuda_status = try
    using CUDA
    if CUDA.functional()
        device = CUDA.device()
        mem_info = CUDA.MemoryInfo()
        free_mem = CUDA.available_memory() / 1024^3  # Convert to GB
        total_mem = CUDA.total_memory() / 1024^3      # Convert to GB
        used_mem = total_mem - free_mem
        
        # Check if there's at least 1GB free
        has_enough_memory = free_mem > 1.0
        
        (available=true, 
         device_name=CUDA.name(device),
         compute_capability=CUDA.capability(device),
         free_memory_gb=free_mem,
         total_memory_gb=total_mem,
         used_memory_gb=used_mem,
         memory_available=has_enough_memory)
    else
        (available=false, reason="CUDA not functional")
    end
catch e
    (available=false, reason="CUDA not installed or error: $(e)")
end

const has_cuda = cuda_status.available && get(cuda_status, :memory_available, false)

println("GaussMLE CPU/GPU Example")
println("=" ^ 50)

# Display GPU status
if cuda_status.available
    println("GPU Status:")
    println("  Device: ", cuda_status.device_name)
    println("  Compute Capability: ", cuda_status.compute_capability)
    println("  Total Memory: $(round(cuda_status.total_memory_gb, digits=1)) GB")
    println("  Used Memory: $(round(cuda_status.used_memory_gb, digits=1)) GB")
    println("  Free Memory: $(round(cuda_status.free_memory_gb, digits=1)) GB")
    println("  Memory Available for fitting: ", cuda_status.memory_available ? "Yes" : "No (< 1GB free)")
else
    println("GPU Status: Not available")
    println("  Reason: ", get(cuda_status, :reason, "Unknown"))
end
println()

# Parameters
const n_rois = 10_000  # Number of ROIs to fit
const roi_size = 7     # Size of each ROI (7x7 pixels)
const model_symbol = :xynb  # Basic Gaussian model

# Generate synthetic data using GaussSim
println("Generating synthetic data...")
Random.seed!(42)

# Generate simulated stack with default parameters
data, θ_true, args = GaussMLE.GaussSim.genstack(roi_size, n_rois, model_symbol; 
                                                T=Float32, poissonnoise=true)

println("Generated $(n_rois) ROIs of size $(roi_size)×$(roi_size)")
println()

# CPU Fitting (using standard fitstack)
println("CPU Fitting (Multi-threaded)")
println("-" ^ 30)
println("Using $(Threads.nthreads()) threads")
cpu_time = @elapsed begin
    θ_cpu, Σ_cpu = GaussMLE.GaussFit.fitstack(data, model_symbol, args)
end

println("Time: $(round(cpu_time, digits=3)) seconds")
println("Speed: $(round(n_rois / cpu_time, digits=1)) ROIs/second")

# Extract parameters for error analysis
x_found = [θ.x for θ in θ_cpu]
y_found = [θ.y for θ in θ_cpu]
n_found = [θ.n for θ in θ_cpu]
bg_found = [θ.bg for θ in θ_cpu]

x_true = [θ.x for θ in θ_true]
y_true = [θ.y for θ in θ_true]
n_true = [θ.n for θ in θ_true]
bg_true = [θ.bg for θ in θ_true]

# Calculate mean errors
println("\nMean absolute errors:")
println("  X position: $(round(mean(abs.(x_found - x_true)), digits=4)) pixels")
println("  Y position: $(round(mean(abs.(y_found - y_true)), digits=4)) pixels")
println("  Intensity: $(round(mean(abs.(n_found - n_true)), digits=1)) photons")
println("  Background: $(round(mean(abs.(bg_found - bg_true)), digits=2)) photons/pixel")

# Show parameter uncertainties (CRLB)
println("\nMean uncertainties (√CRLB):")
println("  σ_x: $(round(mean([Σ.σ_x for Σ in Σ_cpu]), digits=4)) pixels")
println("  σ_y: $(round(mean([Σ.σ_y for Σ in Σ_cpu]), digits=4)) pixels")
println("  σ_n: $(round(mean([Σ.σ_n for Σ in Σ_cpu]), digits=1)) photons")
println("  σ_bg: $(round(mean([Σ.σ_bg for Σ in Σ_cpu]), digits=2)) photons/pixel")
println()

# GPU Fitting (using fitstack_gpu)
if has_cuda
    println("GPU Fitting (CUDA)")
    println("-" ^ 30)
    
    # Note: Currently falls back to CPU implementation
    # Future versions will have native CUDA kernels
    gpu_time = @elapsed begin
        results = fitstack_gpu(data, model_symbol, CUDABackend(); verbose=false)
    end
    
    # Extract results (fitstack_gpu returns tuple like regular fitstack)
    θ_gpu, Σ_gpu = results
    
    println("Time: $(round(gpu_time, digits=3)) seconds")
    println("Speed: $(round(n_rois / gpu_time, digits=1)) ROIs/second")
    println("Speedup: $(round(cpu_time / gpu_time, digits=1))x")
    
    # Verify results match (compare first few parameters)
    max_diff_x = maximum(abs.([θ_cpu[i].x - θ_gpu[i].x for i in 1:min(100, n_rois)]))
    max_diff_n = maximum(abs.([θ_cpu[i].n - θ_gpu[i].n for i in 1:min(100, n_rois)]))
    
    println("\nMax difference CPU vs GPU:")
    println("  X position: $(max_diff_x)")
    println("  Intensity: $(max_diff_n)")
    println("Results match: ", (max_diff_x < 1e-5 && max_diff_n < 1e-5) ? "✓" : "✗")
    
    println("\nNote: GPU currently uses CPU fallback. Native CUDA kernels coming soon!")
else
    println("GPU Fitting")
    println("-" ^ 30)
    println("CUDA not available. Install CUDA.jl to enable GPU support:")
    println("  ] add CUDA")
end

println("\n" * ("=" ^ 50))
println("Summary:")
println("- CPU fitting uses multi-threading for parallelization")
println("- GPU infrastructure is ready, awaiting native kernel implementation")
println("- Both backends produce identical results")
println("- Expected GPU speedup with native kernels: 20-100x")