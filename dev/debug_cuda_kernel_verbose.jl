#!/usr/bin/env julia
#
# Debug kernel with verbose output

using GaussMLE
using GaussMLE.GaussGPU
using CUDA
using Random
using Printf
using Statistics

# Create a simple debug kernel that shows initialization
function debug_init_kernel(d_data::CuDeviceArray{T,3}, d_debug::CuDeviceArray{T,2},
                          roi_size::Int32, n_rois::Int32) where T
    tid = threadIdx().x
    warp_id = (blockIdx().x - 1) * (blockDim().x ÷ 32) + (tid - 1) ÷ 32
    lane_id = (tid - 1) % 32
    
    if warp_id >= n_rois
        return nothing
    end
    
    # Only process first ROI for debugging
    if warp_id > 0
        return nothing
    end
    
    # Load ROI data
    n_pixels = roi_size * roi_size
    local sum_val = T(0)
    local min_val = T(1e10)
    
    # Each thread processes some pixels
    for idx in (lane_id+1):32:n_pixels
        i = (idx - 1) ÷ roi_size + 1
        j = (idx - 1) % roi_size + 1
        val = d_data[i, j, warp_id + 1]
        sum_val += val
        min_val = min(min_val, val)
    end
    
    # Reduce across warp
    sum_val = GaussMLE.GaussGPU.warp_reduce_sum(sum_val)
    
    # Find minimum across warp
    for offset in [16, 8, 4, 2, 1]
        min_val = min(min_val, CUDA.shfl_down_sync(0xffffffff, min_val, offset))
    end
    
    if lane_id == 0
        d_debug[1, 1] = sum_val  # Total sum
        d_debug[2, 1] = min_val  # Minimum value
        d_debug[3, 1] = sum_val / T(n_pixels)  # Average
        d_debug[4, 1] = T(n_pixels)  # Pixel count
    end
    
    return nothing
end

println("Debug CUDA Initialization")
println("=" ^ 50)

# Simple test case
roi_size = 7
roi = Float32[
    12 7 13 17 18 8 10;
    8 16 25 24 20 23 18;
    10 25 37 48 50 35 15;
    14 27 39 74 67 51 20;
    10 25 46 65 81 48 26;
    17 18 44 59 53 34 19;
    9 15 17 28 25 19 19
]

println("ROI data:")
for i in 1:roi_size
    for j in 1:roi_size
        @printf("%4.0f ", roi[i,j])
    end
    println()
end

# Test initialization
d_data = CuArray(reshape(roi, roi_size, roi_size, 1))
d_debug = CUDA.zeros(Float32, 10, 1)

@cuda threads=32 blocks=1 debug_init_kernel(d_data, d_debug, Int32(roi_size), Int32(1))
CUDA.synchronize()

debug_vals = Array(d_debug)
println("\nDebug values from kernel:")
println("  Total sum: $(debug_vals[1,1])")
println("  Min value: $(debug_vals[2,1])")
println("  Average: $(debug_vals[3,1])")
println("  Pixel count: $(debug_vals[4,1])")

# Calculate expected values
println("\nExpected values:")
println("  Total sum: $(sum(roi))")
println("  Min value: $(minimum(roi))")
println("  Average: $(mean(roi))")

# Now test actual fitting kernel
d_params = CUDA.zeros(Float32, 4, 1)
d_crlb = CUDA.zeros(Float32, 4, 1)

GaussMLE.GaussGPU.launch_cuda_fit_kernel!(d_data, d_params, d_crlb, Int32(roi_size))
CUDA.synchronize()

params = Array(d_params)
println("\nFitted parameters:")
println("  x: $(params[1,1])")
println("  y: $(params[2,1])")
println("  n: $(params[3,1])")
println("  bg: $(params[4,1])")