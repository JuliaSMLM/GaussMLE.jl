#!/usr/bin/env julia

using GaussMLE
using CUDA
using Printf

println("Testing Fixed CUDA Kernel - Simple Cases")
println("=" ^ 50)

# Test 1: Perfect centered Gaussian
roi_size = 7
sigma = 1.5f0

test_cases = [
    (x=4.0f0, y=4.0f0, n=1000f0, bg=10f0, desc="centered"),
    (x=3.0f0, y=3.0f0, n=800f0, bg=15f0, desc="off-center low"),
    (x=5.0f0, y=5.0f0, n=1200f0, bg=8f0, desc="off-center high"),
    (x=2.5f0, y=4.5f0, n=600f0, bg=20f0, desc="fractional"),
]

for (i, test) in enumerate(test_cases)
    println("\nTest $i: $(test.desc)")
    println("True: x=$(test.x), y=$(test.y), n=$(test.n), bg=$(test.bg)")
    
    # Generate noiseless Gaussian
    roi = zeros(Float32, roi_size, roi_size)
    for i in 1:roi_size
        for j in 1:roi_size
            xi = Float32(i)
            yi = Float32(j)
            dx = xi - test.x
            dy = yi - test.y
            gauss = test.n * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
            roi[i, j] = test.bg + gauss
        end
    end
    
    # GPU fit
    d_data = CuArray(reshape(roi, roi_size, roi_size, 1))
    d_params = CUDA.zeros(Float32, 4, 1)
    d_crlb = CUDA.zeros(Float32, 4, 1)
    
    GaussMLE.GaussGPU.launch_cuda_fit_kernel!(d_data, d_params, d_crlb, Int32(roi_size),
                                              Float32(1e-6), Int32(50))
    CUDA.synchronize()
    
    params = Array(d_params)
    
    # CPU fit
    args = GaussMLE.GaussModel.Args_xynb(sigma)
    θ_cpu, _ = GaussMLE.GaussFit.fitstack(reshape(roi, roi_size, roi_size, 1), :xynb, args)
    
    println("GPU: x=$(params[1,1]), y=$(params[2,1]), n=$(params[3,1]), bg=$(params[4,1])")
    println("CPU: x=$(θ_cpu[1].x), y=$(θ_cpu[1].y), n=$(θ_cpu[1].n), bg=$(θ_cpu[1].bg)")
    
    # Errors
    err_x_gpu = abs(params[1,1] - test.x)
    err_y_gpu = abs(params[2,1] - test.y)
    err_x_cpu = abs(θ_cpu[1].x - test.x)
    err_y_cpu = abs(θ_cpu[1].y - test.y)
    
    println("GPU errors: x=$(@sprintf("%.6f", err_x_gpu)), y=$(@sprintf("%.6f", err_y_gpu))")
    println("CPU errors: x=$(@sprintf("%.6f", err_x_cpu)), y=$(@sprintf("%.6f", err_y_cpu))")
end

# Test with Poisson noise
println("\n" * ("=" ^ 50))
println("Test with Poisson noise:")

using Distributions
using Random
Random.seed!(42)

x_true = 4.2f0
y_true = 3.8f0
n_true = 1000f0
bg_true = 10f0

roi = zeros(Float32, roi_size, roi_size)
for i in 1:roi_size
    for j in 1:roi_size
        xi = Float32(i)
        yi = Float32(j)
        dx = xi - x_true
        dy = yi - y_true
        gauss = n_true * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
        expected = bg_true + gauss
        roi[i, j] = Float32(rand(Poisson(expected)))
    end
end

println("\nTrue: x=$x_true, y=$y_true, n=$n_true, bg=$bg_true")

# Show ROI
println("ROI center (3x3):")
for i in 3:5
    print("  ")
    for j in 3:5
        @printf("%4.0f ", roi[i,j])
    end
    println()
end

# GPU fit
d_data = CuArray(reshape(roi, roi_size, roi_size, 1))
d_params = CUDA.zeros(Float32, 4, 1)
d_crlb = CUDA.zeros(Float32, 4, 1)

GaussMLE.GaussGPU.launch_cuda_fit_kernel!(d_data, d_params, d_crlb, Int32(roi_size))
CUDA.synchronize()

params = Array(d_params)

# CPU fit
args = GaussMLE.GaussModel.Args_xynb(sigma)
θ_cpu, _ = GaussMLE.GaussFit.fitstack(reshape(roi, roi_size, roi_size, 1), :xynb, args)

println("\nGPU: x=$(params[1,1]), y=$(params[2,1]), n=$(params[3,1]), bg=$(params[4,1])")
println("CPU: x=$(θ_cpu[1].x), y=$(θ_cpu[1].y), n=$(θ_cpu[1].n), bg=$(θ_cpu[1].bg)")

err_x_gpu = abs(params[1,1] - x_true)
err_y_gpu = abs(params[2,1] - y_true)
err_x_cpu = abs(θ_cpu[1].x - x_true)
err_y_cpu = abs(θ_cpu[1].y - y_true)

println("\nGPU errors: x=$(@sprintf("%.3f", err_x_gpu)), y=$(@sprintf("%.3f", err_y_gpu))")
println("CPU errors: x=$(@sprintf("%.3f", err_x_cpu)), y=$(@sprintf("%.3f", err_y_cpu))")