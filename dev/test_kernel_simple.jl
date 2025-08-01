#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using GaussMLE
using GaussMLE.GaussGPU
using CUDA

println("Simple CUDA Kernel Test")
println("=" ^ 40)

# Create minimal test case
roi_size = 7
n_rois = 1  # Just one ROI

# Create a simple Gaussian spot
data = zeros(Float32, roi_size, roi_size)
x_true, y_true = 4.0f0, 4.0f0
intensity = 1000f0
bg = 10f0
sigma = 1.5f0

for i in 1:roi_size
    for j in 1:roi_size
        dx = Float32(i) - x_true
        dy = Float32(j) - y_true
        gauss = intensity * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
        data[i, j] = bg + gauss
    end
end

println("Data center: $(data[4,4])")
println("Data corners: $(data[1,1]), $(data[7,7])")

# Reshape for batch processing
data_batch = reshape(data, roi_size, roi_size, 1)

# Test direct kernel call
d_data = CuArray(data_batch)
d_params = CUDA.zeros(Float32, 4, 1)
d_crlb = CUDA.zeros(Float32, 4, 1)

println("\nCalling kernel...")
t_start = time()
GaussGPU.launch_gaussian_fit!(d_data, d_params, d_crlb)
CUDA.synchronize()
t_kernel = time() - t_start
println("Kernel time: $(t_kernel * 1000) ms")

# Get results
params = Array(d_params)
println("\nResults:")
println("  x = $(params[1,1]) (true: $x_true)")
println("  y = $(params[2,1]) (true: $y_true)")
println("  N = $(params[3,1]) (true: $intensity)")
println("  bg = $(params[4,1]) (true: $bg)")

# Compare with CPU
θ_cpu, _ = GaussMLE.fitstack(data_batch, :xynb)
println("\nCPU Results:")
println("  x = $(θ_cpu[1].x)")
println("  y = $(θ_cpu[1].y)")
println("  N = $(θ_cpu[1].n)")
println("  bg = $(θ_cpu[1].bg)")