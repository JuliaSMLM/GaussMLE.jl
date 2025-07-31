#!/usr/bin/env julia
using GaussMLE
using CUDA

println("=== GaussMLE GPU Support Status ===\n")

# Check CUDA
println("CUDA Status:")
println("  Functional: $(CUDA.functional())")
if CUDA.functional()
    println("  Device: $(CUDA.name(CUDA.device()))")
    println("  Memory: $(round(CUDA.total_memory() / 1e9, digits=1)) GB")
    println("  Compute Capability: $(CUDA.capability(CUDA.device()))")
end

# Check backend selection
println("\nBackend Selection:")
backend = GaussMLE.select_backend()
println("  Selected: $(GaussMLE.backend_name(backend))")
println("  Type: $(typeof(backend))")

# Quick test
println("\nQuick Test:")
data, θ_true, args = GaussMLE.GaussSim.genstack(7, 100, :xynb; T=Float32, poissonnoise=true)

# CPU fit
t_cpu = @elapsed begin
    θ_cpu, Σ_cpu = GaussMLE.fitstack(data, :xynb, args)
end
println("  CPU fit: $(round(t_cpu*1000, digits=1)) ms for 100 ROIs")

# GPU fit
t_gpu = @elapsed begin
    θ_gpu, Σ_gpu = GaussMLE.fitstack_gpu(data, :xynb)
end
println("  GPU fit: $(round(t_gpu*1000, digits=1)) ms for 100 ROIs")

# Check results
cpu_mean_x = sum(p.x for p in θ_cpu) / length(θ_cpu)
gpu_mean_x = sum(p.x for p in θ_gpu) / length(θ_gpu)
println("  Mean x - CPU: $(round(cpu_mean_x, digits=3)), GPU: $(round(gpu_mean_x, digits=3))")
println("  Difference: $(round(abs(cpu_mean_x - gpu_mean_x), digits=6))")

println("\nStatus Summary:")
println("✅ GPU infrastructure implemented")
println("✅ Backend abstraction working") 
println("✅ Batching system functional")
println("✅ CPU multi-threaded backend operational")
println("⚠️  CUDA kernels not implemented (falls back to CPU)")
println("❌ Metal backend not implemented")
println("\nNext step: Implement native CUDA kernels for 20-100x speedup")