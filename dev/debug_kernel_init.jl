#!/usr/bin/env julia
#
# Debug kernel initialization specifically

using GaussMLE
using CUDA
using Printf

# Create custom debug kernel to print initialization values
function debug_init_kernel(d_data::CuDeviceArray{T,3}, d_debug::CuDeviceArray{T,2},
                          roi_size::Int32, n_rois::Int32) where T
    tid = threadIdx().x
    warp_id = (blockIdx().x - 1) * (blockDim().x ÷ 32) + (tid - 1) ÷ 32
    lane_id = (tid - 1) % 32
    
    if warp_id >= n_rois || warp_id > 0
        return nothing
    end
    
    # Load ROI data
    n_pixels = roi_size * roi_size
    
    if lane_id == 0
        # Find minimum
        min_val = d_data[1, 1, 1]
        for idx in 1:n_pixels
            i = (idx - 1) ÷ roi_size + 1
            j = (idx - 1) % roi_size + 1
            val = d_data[i, j, 1]
            min_val = min(min_val, val)
        end
        
        bg = min_val
        
        # Calculate center of mass
        sum_above_bg = T(0)
        sum_x_weighted = T(0)
        sum_y_weighted = T(0)
        
        for idx in 1:n_pixels
            i = (idx - 1) ÷ roi_size + 1  # 1-based row
            j = (idx - 1) % roi_size + 1  # 1-based column
            val = d_data[i, j, 1]
            val_above_bg = max(val - bg, T(0))
            
            sum_above_bg += val_above_bg
            sum_x_weighted += val_above_bg * T(j)  # 1-based
            sum_y_weighted += val_above_bg * T(i)  # 1-based
            
            # Store debug info for first few pixels
            if idx <= 5
                d_debug[idx, 1] = T(i)
                d_debug[idx, 2] = T(j)
                d_debug[idx, 3] = val
                d_debug[idx, 4] = val_above_bg
            end
        end
        
        # Store sums
        d_debug[6, 1] = sum_above_bg
        d_debug[6, 2] = sum_x_weighted
        d_debug[6, 3] = sum_y_weighted
        d_debug[6, 4] = bg
        
        # Calculate COM
        if sum_above_bg > T(0)
            com_x_1based = sum_x_weighted / sum_above_bg
            com_y_1based = sum_y_weighted / sum_above_bg
            com_x_0based = com_x_1based - T(1)
            com_y_0based = com_y_1based - T(1)
        else
            com_x_1based = T((roi_size + 1) / 2)
            com_y_1based = T((roi_size + 1) / 2)
            com_x_0based = T((roi_size - 1) / 2)
            com_y_0based = T((roi_size - 1) / 2)
        end
        
        d_debug[7, 1] = com_x_1based
        d_debug[7, 2] = com_y_1based
        d_debug[7, 3] = com_x_0based
        d_debug[7, 4] = com_y_0based
    end
    
    return nothing
end

# Test case
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

println("Debug Kernel Initialization")
println("=" ^ 50)
println("\nROI data:")
for i in 1:roi_size
    for j in 1:roi_size
        @printf("%4.0f ", roi[i,j])
    end
    println()
end

# Test debug kernel
d_data = CuArray(reshape(roi, roi_size, roi_size, 1))
d_debug = CUDA.zeros(Float32, 10, 4)

@cuda threads=32 blocks=1 debug_init_kernel(d_data, d_debug, Int32(roi_size), Int32(1))
CUDA.synchronize()

debug_vals = Array(d_debug)

println("\nDebug output:")
println("First 5 pixels:")
for i in 1:5
    println("  Pixel $i: i=$(Int(debug_vals[i,1])), j=$(Int(debug_vals[i,2])), val=$(debug_vals[i,3]), above_bg=$(debug_vals[i,4])")
end

println("\nSums and background:")
println("  sum_above_bg = $(debug_vals[6,1])")
println("  sum_x_weighted = $(debug_vals[6,2])")
println("  sum_y_weighted = $(debug_vals[6,3])")
println("  background = $(debug_vals[6,4])")

println("\nCenter of mass:")
println("  1-based: x=$(debug_vals[7,1]), y=$(debug_vals[7,2])")
println("  0-based: x=$(debug_vals[7,3]), y=$(debug_vals[7,4])")

# Manual calculation for comparison
min_val = minimum(roi)
bg_manual = min_val
sum_above_bg_manual = 0.0
sum_x_weighted_manual = 0.0
sum_y_weighted_manual = 0.0

for i in 1:roi_size
    for j in 1:roi_size
        val_above_bg = max(roi[i,j] - bg_manual, 0.0)
        sum_above_bg_manual += val_above_bg
        sum_x_weighted_manual += val_above_bg * j
        sum_y_weighted_manual += val_above_bg * i
    end
end

com_x_1based = sum_x_weighted_manual / sum_above_bg_manual
com_y_1based = sum_y_weighted_manual / sum_above_bg_manual
com_x_0based = com_x_1based - 1
com_y_0based = com_y_1based - 1

println("\nManual calculation:")
println("  background = $bg_manual")
println("  sum_above_bg = $sum_above_bg_manual")
println("  sum_x_weighted = $sum_x_weighted_manual")
println("  sum_y_weighted = $sum_y_weighted_manual")
println("  1-based COM: x=$com_x_1based, y=$com_y_1based")
println("  0-based COM: x=$com_x_0based, y=$com_y_0based")

# Now test actual kernel
println("\n" * ("=" ^ 50))
println("Testing actual kernel with this ROI:")

d_params = CUDA.zeros(Float32, 4, 1)
d_crlb = CUDA.zeros(Float32, 4, 1)

GaussMLE.GaussGPU.launch_cuda_fit_kernel!(d_data, d_params, d_crlb, Int32(roi_size))
CUDA.synchronize()

params = Array(d_params)
println("\nKernel result:")
println("  x=$(params[1,1]), y=$(params[2,1]), n=$(params[3,1]), bg=$(params[4,1])")

# Expected position based on visual inspection
println("\nExpected approximate position: x≈3.5, y≈3.5 (0-based)")