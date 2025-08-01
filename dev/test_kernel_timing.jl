#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using GaussMLE
using GaussMLE.GaussGPU
using CUDA
using Statistics

println("Kernel Timing Test")
println("=" ^ 40)

# Create test data
roi_size = 7
n_rois = 100
data = rand(Float32, roi_size, roi_size, n_rois) .* 100 .+ 10

# Transfer to GPU
d_data = CuArray(data)
d_params = CUDA.zeros(Float32, 4, n_rois)
d_crlb = CUDA.zeros(Float32, 4, n_rois)

# First call - includes compilation
println("\nFirst call (includes compilation):")
CUDA.@sync begin
    t1 = @elapsed GaussGPU.launch_gaussian_fit!(d_data, d_params, d_crlb)
end
println("  Time: $(t1 * 1000) ms")

# Second call - should be fast
println("\nSecond call (compiled):")
CUDA.@sync begin
    t2 = @elapsed GaussGPU.launch_gaussian_fit!(d_data, d_params, d_crlb)
end
println("  Time: $(t2 * 1000) ms")

# Third call
println("\nThird call:")
CUDA.@sync begin
    t3 = @elapsed GaussGPU.launch_gaussian_fit!(d_data, d_params, d_crlb)
end
println("  Time: $(t3 * 1000) ms")

# Multiple calls
println("\nAverage of 10 calls:")
times = Float64[]
for i in 1:10
    CUDA.@sync begin
        t = @elapsed GaussGPU.launch_gaussian_fit!(d_data, d_params, d_crlb)
    end
    push!(times, t * 1000)
end
println("  Mean: $(mean(times)) ms")
println("  Min: $(minimum(times)) ms")
println("  Max: $(maximum(times)) ms")

# Check results
params = Array(d_params)
println("\nFirst few results:")
for i in 1:min(5, n_rois)
    println("  ROI $i: x=$(params[1,i]), y=$(params[2,i]), n=$(params[3,i]), bg=$(params[4,i])")
end