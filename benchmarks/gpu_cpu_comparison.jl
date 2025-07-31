#!/usr/bin/env julia
using GaussMLE
using CUDA
using Statistics
using Printf
using BenchmarkTools

println("=== GaussMLE GPU/CPU Benchmark ===")
println("System Information:")
println("  Julia: $(VERSION)")
println("  CUDA: $(CUDA.version())")
println("  GPU: $(CUDA.name(CUDA.device()))")
println("  GPU Memory: $(round(CUDA.available_memory()[2] / 1e9, digits=1)) GB")
println("  CPU Threads: $(Threads.nthreads())")
println()

# Test parameters
const BOXSIZE = 7
const SIGMA_PSF = 1.2f0
const POSITION_JITTER = 2.0f0  # Emitters within a few pixels of center
const N_PHOTONS = 500f0
const BACKGROUND = 2f0

# Create test scenario
function create_test_data(n_rois::Int, ::Type{T}=Float32) where T
    # Generate parameters
    center = T(BOXSIZE / 2)
    x_positions = center .+ POSITION_JITTER .* (rand(T, n_rois) .- 0.5f0) .* 2
    y_positions = center .+ POSITION_JITTER .* (rand(T, n_rois) .- 0.5f0) .* 2
    n_photons = N_PHOTONS .+ 100f0 .* randn(T, n_rois)
    backgrounds = BACKGROUND .+ 0.5f0 .* randn(T, n_rois)
    
    # Generate data
    stack = zeros(T, BOXSIZE, BOXSIZE, n_rois)
    args = GaussMLE.GaussModel.Args_xynb(T(SIGMA_PSF))
    
    for i in 1:n_rois
        θ = GaussMLE.GaussModel.θ_xynb(x_positions[i], y_positions[i], 
                                       n_photons[i], backgrounds[i])
        GaussMLE.GaussSim.genroi!(view(stack, :, :, i), θ, args)
    end
    
    # Add Poisson noise
    GaussMLE.GaussSim.poissonnoise!(stack)
    
    return stack
end

# Benchmark function
function benchmark_fitting(sizes)
    results = []
    
    for n_rois in sizes
        println("\n--- Testing with $n_rois ROIs ---")
        
        # Create test data
        data = create_test_data(n_rois)
        println("Data size: $(size(data))")
        println("Memory usage: $(round(sizeof(data) / 1e6, digits=1)) MB")
        
        # Warmup
        if n_rois <= 10000
            GaussMLE.fitstack(data[:,:,1:min(100,n_rois)], :xynb)
            backend = GaussMLE.select_backend()
            GaussMLE.fitstack_gpu(data[:,:,1:min(100,n_rois)], :xynb, backend)
        end
        
        # CPU benchmark (single-threaded baseline)
        println("\nCPU ($(Threads.nthreads()) threads):")
        cpu_time = if n_rois <= 100000
            t = @benchmark GaussMLE.fitstack($data, :xynb) samples=3
            median(t).time / 1e9  # Convert to seconds
        else
            # For large datasets, do single run
            t_start = time()
            θ_cpu, Σ_cpu = GaussMLE.fitstack(data, :xynb)
            cpu_time = time() - t_start
        end
        
        cpu_rate = n_rois / cpu_time
        println("  Time: $(round(cpu_time, digits=3))s")
        println("  Rate: $(round(cpu_rate, digits=0)) ROIs/sec")
        
        # GPU benchmark
        println("\nGPU ($(backend_name(backend))):")
        backend = GaussMLE.select_backend()
        
        gpu_time = if n_rois <= 100000
            t = @benchmark GaussMLE.fitstack_gpu($data, :xynb, $backend) samples=3
            median(t).time / 1e9
        else
            t_start = time()
            θ_gpu, Σ_gpu = GaussMLE.fitstack_gpu(data, :xynb, backend, verbose=true)
            gpu_time = time() - t_start
        end
        
        gpu_rate = n_rois / gpu_time
        speedup = cpu_time / gpu_time
        
        println("  Time: $(round(gpu_time, digits=3))s")
        println("  Rate: $(round(gpu_rate, digits=0)) ROIs/sec")
        println("  Speedup: $(round(speedup, digits=1))x")
        
        # Store results
        push!(results, (
            n_rois = n_rois,
            cpu_time = cpu_time,
            gpu_time = gpu_time,
            cpu_rate = cpu_rate,
            gpu_rate = gpu_rate,
            speedup = speedup
        ))
        
        # Memory check
        if backend isa GaussMLE.CUDABackend
            free_mem, total_mem = CUDA.available_memory()
            println("  GPU Memory: $(round((total_mem - free_mem) / 1e9, digits=2)) / $(round(total_mem / 1e9, digits=1)) GB used")
        end
    end
    
    return results
end

# Run benchmarks
sizes = [100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]
println("\nStarting benchmarks...")
println("Test configuration:")
println("  Box size: $(BOXSIZE)×$(BOXSIZE)")
println("  PSF σ: $(SIGMA_PSF) pixels")
println("  Position range: ±$(POSITION_JITTER) pixels from center")
println("  Photons: $(N_PHOTONS) ± 100")
println("  Background: $(BACKGROUND) ± 0.5")

results = benchmark_fitting(sizes)

# Summary table
println("\n\n=== SUMMARY ===")
println("┌─────────────┬──────────────┬──────────────┬────────────┬────────────┬─────────┐")
println("│   N ROIs    │ CPU Time (s) │ GPU Time (s) │ CPU (ROI/s)│ GPU (ROI/s)│ Speedup │")
println("├─────────────┼──────────────┼──────────────┼────────────┼────────────┼─────────┤")
for r in results
    @printf("│ %11s │ %12.3f │ %12.3f │ %10.0f │ %10.0f │ %6.1fx │\n",
            string(r.n_rois), r.cpu_time, r.gpu_time, r.cpu_rate, r.gpu_rate, r.speedup)
end
println("└─────────────┴──────────────┴──────────────┴────────────┴────────────┴─────────┘")

# Note about current status
println("\n⚠️  Note: CUDA kernels not yet implemented - GPU currently falls back to CPU")
println("Expected speedup with CUDA kernels: 20-100x")

# Accuracy check on small sample
println("\n=== Accuracy Check ===")
small_data = create_test_data(100)
θ_cpu, Σ_cpu = GaussMLE.fitstack(small_data, :xynb)
θ_gpu, Σ_gpu = GaussMLE.fitstack_gpu(small_data, :xynb)

# Compare parameters
cpu_x = [p.x for p in θ_cpu]
gpu_x = [p.x for p in θ_gpu]
max_diff = maximum(abs.(cpu_x .- gpu_x))
println("Maximum parameter difference (x): $(max_diff)")
println("Results match: $(max_diff < 1e-5 ? "✓" : "✗")")