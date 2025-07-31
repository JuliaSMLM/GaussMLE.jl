#!/usr/bin/env julia
#
# Debug script for CUDA kernel

using GaussMLE
using GaussMLE: fitstack_gpu, CUDABackend
using GaussMLE.GaussGPU: launch_cuda_fit_kernel!
using CUDA
using Random
using Printf

println("CUDA Kernel Debug")
println("=" ^ 50)

# Simple test case
Random.seed!(42)
roi_size = 7
n_rois = 1

# Generate one simple ROI with known parameters
x_true = 3.5f0
y_true = 3.5f0
intensity_true = 1000f0
bg_true = 10f0

# Create synthetic ROI
roi = zeros(Float32, roi_size, roi_size)
sigma = 1.5f0

for i in 1:roi_size
    for j in 1:roi_size
        dx = Float32(i - 1) - x_true
        dy = Float32(j - 1) - y_true
        gauss = intensity_true * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
        roi[i, j] = bg_true + gauss
    end
end

# Add Poisson noise
using Distributions
for i in 1:roi_size*roi_size
    roi[i] = rand(Poisson(roi[i]))
end

println("Test ROI:")
println("True params: x=$x_true, y=$y_true, n=$intensity_true, bg=$bg_true")
println("\nROI data:")
for i in 1:roi_size
    for j in 1:roi_size
        @printf("%6.1f ", roi[i,j])
    end
    println()
end

# Test direct kernel call
println("\n\nTesting direct kernel call...")
d_data = CuArray(reshape(roi, roi_size, roi_size, 1))
d_params = CUDA.zeros(Float32, 4, 1)
d_crlb = CUDA.zeros(Float32, 4, 1)

launch_cuda_fit_kernel!(d_data, d_params, d_crlb, Int32(roi_size))
CUDA.synchronize()

params = Array(d_params)
crlb = Array(d_crlb)

println("\nFitted parameters:")
println("  x: $(params[1,1]) (error: $(abs(params[1,1] - x_true)))")
println("  y: $(params[2,1]) (error: $(abs(params[2,1] - y_true)))")
println("  n: $(params[3,1]) (error: $(abs(params[3,1] - intensity_true)))")
println("  bg: $(params[4,1]) (error: $(abs(params[4,1] - bg_true)))")

# Test through full API
println("\n\nTesting through fitstack_gpu API...")
data_3d = reshape(roi, roi_size, roi_size, 1)
θ_gpu, Σ_gpu = fitstack_gpu(data_3d, :xynb, CUDABackend(); verbose=false)

println("\nFitted parameters:")
println("  x: $(θ_gpu[1].x) (error: $(abs(θ_gpu[1].x - x_true)))")
println("  y: $(θ_gpu[1].y) (error: $(abs(θ_gpu[1].y - y_true)))")
println("  n: $(θ_gpu[1].n) (error: $(abs(θ_gpu[1].n - intensity_true)))")
println("  bg: $(θ_gpu[1].bg) (error: $(abs(θ_gpu[1].bg - bg_true)))")

# Compare with CPU
println("\n\nComparing with CPU...")
args = GaussMLE.GaussModel.Args_xynb(1.5f0)
θ_cpu, Σ_cpu = GaussMLE.GaussFit.fitstack(data_3d, :xynb, args)

println("\nCPU fitted parameters:")
println("  x: $(θ_cpu[1].x) (error: $(abs(θ_cpu[1].x - x_true)))")
println("  y: $(θ_cpu[1].y) (error: $(abs(θ_cpu[1].y - y_true)))")
println("  n: $(θ_cpu[1].n) (error: $(abs(θ_cpu[1].n - intensity_true)))")
println("  bg: $(θ_cpu[1].bg) (error: $(abs(θ_cpu[1].bg - bg_true)))")