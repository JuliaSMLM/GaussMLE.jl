#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using GaussMLE
using GaussMLE.GaussGPU
using CUDA
# using BenchmarkTools

println("Testing GPU Integration")
println("=" ^ 60)

# Check CUDA availability
println("\nCUDA Status:")
println("  Functional: $(CUDA.functional())")
if CUDA.functional()
    println("  Device: $(CUDA.name(CUDA.device()))")
    mem_info = CUDA.available_memory()
    println("  Memory info: $mem_info")
end

# Test backend selection
println("\nBackend Selection:")
backend = GaussGPU.select_backend()
println("  Selected: $(typeof(backend))")
println("  Name: $(GaussGPU.backend_name(backend))")

# Create small test data
n_rois = 100
roi_size = 7
data = rand(Float32, roi_size, roi_size, n_rois) .* 100 .+ 10

# Test direct CUDA kernel call
println("\nDirect CUDA Kernel Test:")
if backend isa GaussGPU.CUDABackend
    d_data = CuArray(data)
    d_params = CUDA.zeros(Float32, 4, n_rois)
    d_crlb = CUDA.zeros(Float32, 4, n_rois)
    
    # Time the kernel directly
    CUDA.@sync begin
        t_kernel = @elapsed GaussGPU.launch_gaussian_fit!(d_data, d_params, d_crlb)
    end
    println("  Kernel time: $(t_kernel * 1000) ms")
    
    # Check results
    params = Array(d_params)
    println("  First ROI result: x=$(params[1,1]), y=$(params[2,1])")
end

# Test through fitstack_gpu API
println("\nFitstack_gpu API Test:")
t_gpu = @elapsed begin
    θ_gpu, Σ_gpu = GaussGPU.fitstack_gpu(data, :xynb, backend)
end
println("  Total time: $(t_gpu * 1000) ms")
println("  First ROI: x=$(θ_gpu[1].x), y=$(θ_gpu[1].y)")

# Compare with CPU
println("\nCPU Comparison:")
t_cpu = @elapsed begin
    θ_cpu, Σ_cpu = GaussMLE.fitstack(data, :xynb)
end
println("  CPU time: $(t_cpu * 1000) ms")
println("  Speedup: $(t_cpu / t_gpu)x")

# Check if results match
max_diff = maximum(abs(θ_gpu[i].x - θ_cpu[i].x) for i in 1:n_rois)
println("  Max difference: $max_diff")
println("  Match: $(max_diff < 1e-4 ? "✓" : "✗")")