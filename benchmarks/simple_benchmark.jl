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
    println("  GPU Memory: $(round(CUDA.total_memory() / 1e9, digits=1)) GB")
end
println("  CPU Threads: $(Threads.nthreads())")
println()

# Function to run benchmark using existing test infrastructure
function benchmark_with_size(n_rois::Int)
    println("\n--- Testing with $n_rois ROIs ---")
    
    # Use the same approach as in tests
    T = Float32
    boxsz = 7
    out, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, n_rois, :xynb; T=T, poissonnoise=true)
    
    println("Data size: $(size(out))")
    println("Memory usage: $(round(sizeof(out) / 1e6, digits=1)) MB")
    
    # CPU benchmark
    println("\nCPU ($(Threads.nthreads()) threads):")
    GC.gc()
    t_start = time()
    θ_cpu, Σ_cpu = GaussMLE.GaussFit.fitstack(out, :xynb, args)
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
    θ_gpu, Σ_gpu = GaussMLE.fitstack_gpu(out, :xynb, backend, verbose=verbose)
    gpu_time = time() - t_start
    
    gpu_rate = n_rois / gpu_time
    speedup = cpu_time / gpu_time
    
    println("  Time: $(round(gpu_time, digits=3))s")
    println("  Rate: $(round(gpu_rate, digits=0)) ROIs/sec")
    println("  Speedup: $(round(speedup, digits=1))x")
    
    # Quick accuracy check
    if n_rois <= 1000
        cpu_x = mean([p.x for p in θ_cpu])
        gpu_x = mean([p.x for p in θ_gpu])
        println("  Mean x - CPU: $(round(cpu_x, digits=3)), GPU: $(round(gpu_x, digits=3))")
        println("  Difference: $(round(abs(cpu_x - gpu_x), digits=6))")
    end
    
    return (n_rois=n_rois, cpu_time=cpu_time, gpu_time=gpu_time, 
            cpu_rate=cpu_rate, gpu_rate=gpu_rate, speedup=speedup)
end

# Run benchmarks
sizes = [100, 1_000, 10_000]
if Threads.nthreads() > 1 || CUDA.functional()
    append!(sizes, [100_000, 1_000_000])
    if Threads.nthreads() >= 8
        push!(sizes, 10_000_000)
    end
end

println("\nStarting benchmarks...")
println("Using standard GaussMLE test data generation")

results = []
for n in sizes
    try
        push!(results, benchmark_with_size(n))
    catch e
        println("\nError with $n ROIs: $e")
        if n == sizes[1]
            rethrow(e)
        end
    end
end

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

# Performance analysis
if length(results) > 0
    avg_speedup = mean([r.speedup for r in results])
    println("\nAverage speedup: $(round(avg_speedup, digits=1))x")
    
    if CUDA.functional()
        println("\n⚠️  Note: CUDA kernels not yet implemented")
        println("Current speedup is from multi-threaded CPU execution")
        println("Expected speedup with native CUDA kernels: 20-100x")
    end
end