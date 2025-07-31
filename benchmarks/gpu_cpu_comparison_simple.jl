#!/usr/bin/env julia
using GaussMLE
using CUDA
using Statistics
using Printf

println("=== GaussMLE GPU/CPU Benchmark ===")
println("System Information:")
println("  Julia: $(VERSION)")
println("  CUDA functional: $(CUDA.functional())")
if CUDA.functional()
    println("  GPU: $(CUDA.name(CUDA.device()))")
    total_mem = CUDA.total_memory()
    println("  GPU Memory: $(round(total_mem / 1e9, digits=1)) GB total")
end
println("  CPU Threads: $(Threads.nthreads())")
println()

# Test parameters
const BOXSIZE = 7
const SIGMA_PSF = 1.2f0
const POSITION_JITTER = 2.0f0  # Emitters within a few pixels of center
const N_PHOTONS = 500f0
const BACKGROUND = 2f0

# Create test scenario for benchmark
scenario = GaussMLE.GaussSim.GPUTestScenario{Float32}(
    "benchmark",
    BOXSIZE,
    0,  # Will set per test
    (N_PHOTONS - 100f0, N_PHOTONS + 100f0),
    (BACKGROUND - 0.5f0, BACKGROUND + 0.5f0),
    POSITION_JITTER,
    SIGMA_PSF,
    nothing
)

# Benchmark function
function benchmark_fitting(sizes)
    results = []
    
    for n_rois in sizes
        println("\n--- Testing with $n_rois ROIs ---")
        
        # Create test data using our scenario
        test_scenario = GaussMLE.GaussSim.GPUTestScenario{Float32}(
            scenario.name, scenario.boxsize, n_rois,
            scenario.n_photons_range, scenario.background_range,
            scenario.position_jitter, scenario.σ_PSF, scenario.variance_map
        )
        
        gen = GaussMLE.GaussSim.StreamingBatchGenerator(test_scenario, n_rois)
        batch = GaussMLE.GaussSim.next_batch!(gen)
        data = batch.data
        
        println("Data size: $(size(data))")
        println("Memory usage: $(round(sizeof(data) / 1e6, digits=1)) MB")
        
        # Get model args
        args = GaussMLE.GaussModel.genargs(GaussMLE.GaussModel.θ_xynb, T=Float32)
        
        # Warmup
        if n_rois <= 1000
            GaussMLE.fitstack(data[:,:,1:min(100,n_rois)], :xynb, args)
            backend = GaussMLE.select_backend()
            GaussMLE.fitstack_gpu(data[:,:,1:min(100,n_rois)], :xynb, backend)
        end
        
        # CPU benchmark
        println("\nCPU ($(Threads.nthreads()) threads):")
        GC.gc()
        t_start = time()
        θ_cpu, Σ_cpu = GaussMLE.fitstack(data, :xynb, args)
        cpu_time = time() - t_start
        
        cpu_rate = n_rois / cpu_time
        println("  Time: $(round(cpu_time, digits=3))s")
        println("  Rate: $(round(cpu_rate, digits=0)) ROIs/sec")
        
        # GPU benchmark
        backend = GaussMLE.select_backend()
        println("\nGPU ($(GaussMLE.backend_name(backend))):")
        
        GC.gc()
        if backend isa GaussMLE.CUDABackend
            CUDA.reclaim()
        end
        
        verbose = n_rois >= 1_000_000
        t_start = time()
        θ_gpu, Σ_gpu = GaussMLE.fitstack_gpu(data, :xynb, backend, verbose=verbose)
        gpu_time = time() - t_start
        
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
            free_mem = CUDA.available_memory()
            total_mem = CUDA.total_memory()
            used_mem = total_mem - free_mem
            println("  GPU Memory: $(round(used_mem / 1e9, digits=2)) / $(round(total_mem / 1e9, digits=1)) GB used")
        end
        
        # Quick accuracy check
        if n_rois <= 1000
            cpu_x = [p.x for p in θ_cpu[1:min(100,end)]]
            gpu_x = [p.x for p in θ_gpu[1:min(100,end)]]
            max_diff = maximum(abs.(cpu_x .- gpu_x))
            println("  Accuracy check: max diff = $(max_diff) $(max_diff < 1e-5 ? "✓" : "✗")")
        end
    end
    
    return results
end

# Run benchmarks
sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
if Threads.nthreads() > 1 || CUDA.functional()
    push!(sizes, 10_000_000)
end

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
println("\nCurrent speedup is from multi-threaded CPU execution in the GPU backend")