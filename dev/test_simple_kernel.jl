#!/usr/bin/env julia

using GaussMLE
using CUDA
using Printf

# Simple synthetic test
roi_size = 7
# In GaussMLE continuous coordinates: pixel 4 has center at position 4.0
x_true = 4.0f0  
y_true = 4.0f0
n_true = 1000f0
bg_true = 10f0
sigma = 1.5f0

# Generate perfect Gaussian
roi = zeros(Float32, roi_size, roi_size)
for i in 1:roi_size
    for j in 1:roi_size
        # Pixel centers in continuous coordinates
        xi = Float32(i)
        yi = Float32(j)
        dx = xi - x_true
        dy = yi - y_true
        gauss = n_true * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
        roi[i, j] = bg_true + gauss
    end
end

println("Simple Kernel Test")
println("True params: x=$x_true, y=$y_true, n=$n_true, bg=$bg_true")
println("\nROI center values:")
for i in 2:6
    print("  ")
    for j in 2:6
        @printf("%6.1f ", roi[i,j])
    end
    println()
end

# Test GPU
d_data = CuArray(reshape(roi, roi_size, roi_size, 1))
d_params = CUDA.zeros(Float32, 4, 1)
d_crlb = CUDA.zeros(Float32, 4, 1)

GaussMLE.GaussGPU.launch_cuda_fit_kernel!(d_data, d_params, d_crlb, Int32(roi_size))
CUDA.synchronize()

params = Array(d_params)
println("\nGPU fit: x=$(params[1,1]), y=$(params[2,1]), n=$(params[3,1]), bg=$(params[4,1])")

# Compare with CPU
args = GaussMLE.GaussModel.Args_xynb(sigma)
θ_cpu, _ = GaussMLE.GaussFit.fitstack(reshape(roi, roi_size, roi_size, 1), :xynb, args)
println("CPU fit: x=$(θ_cpu[1].x), y=$(θ_cpu[1].y), n=$(θ_cpu[1].n), bg=$(θ_cpu[1].bg)")