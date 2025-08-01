#!/usr/bin/env julia

using GaussMLE
using CUDA
using Printf

# Create a debug kernel that prints intermediate values
function debug_nr_kernel(d_data::CuDeviceArray{T,3}, d_debug::CuDeviceArray{T,2},
                        roi_size::Int32, max_iter::Int32) where T
    tid = threadIdx().x
    warp_id = (blockIdx().x - 1) * (blockDim().x ÷ 32) + (tid - 1) ÷ 32
    lane_id = (tid - 1) % 32
    
    if warp_id >= 1 || warp_id > 0
        return nothing
    end
    
    # Load data into shared memory
    n_pixels = roi_size * roi_size
    shared_mem = @cuDynamicSharedMem(T, n_pixels)
    
    # Collaborative load
    pixels_per_thread = (n_pixels + 32 - 1) ÷ 32
    for p in 1:pixels_per_thread
        idx = lane_id * pixels_per_thread + p
        if idx <= n_pixels
            i = (idx - 1) ÷ roi_size + 1
            j = (idx - 1) % roi_size + 1
            shared_mem[idx] = d_data[i, j, 1]
        end
    end
    
    CUDA.sync_warp()
    
    # Initialize (lane 0 only)
    local x::T = T(0)
    local y::T = T(0)
    local n::T = T(0)
    local bg::T = T(0)
    
    if lane_id == 0
        # Find background
        min_val = shared_mem[1]
        for idx in 2:n_pixels
            min_val = min(min_val, shared_mem[idx])
        end
        bg = min_val
        
        # Center of mass
        sum_above = T(0)
        sum_x = T(0)
        sum_y = T(0)
        
        for idx in 1:n_pixels
            val_above = max(shared_mem[idx] - bg, T(0))
            i = (idx - 1) ÷ roi_size + 1
            j = (idx - 1) % roi_size + 1
            sum_above += val_above
            sum_x += val_above * T(j)
            sum_y += val_above * T(i)
        end
        
        x = sum_x / sum_above
        y = sum_y / sum_above
        n = sum_above
        
        # Store initial values
        d_debug[1, 1] = x
        d_debug[1, 2] = y
        d_debug[1, 3] = n
        d_debug[1, 4] = bg
    end
    
    # Broadcast to all lanes
    x = CUDA.shfl_sync(0xffffffff, x, 1)
    y = CUDA.shfl_sync(0xffffffff, y, 1)
    n = CUDA.shfl_sync(0xffffffff, n, 1)
    bg = CUDA.shfl_sync(0xffffffff, bg, 1)
    
    # Do first iteration
    sigma = T(1.5)
    
    # Calculate gradients
    grad_x = T(0)
    grad_y = T(0)
    grad_n = T(0)
    grad_bg = T(0)
    
    for p in 1:pixels_per_thread
        idx = lane_id * pixels_per_thread + p
        if idx <= n_pixels
            i = (idx - 1) ÷ roi_size + 1
            j = (idx - 1) % roi_size + 1
            
            # Model value
            xi = T(i)
            yi = T(j)
            dx = xi - x
            dy = yi - y
            gaussian = n * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
            model_val = bg + gaussian
            
            if model_val > T(1e-10) && idx <= 10 && lane_id == 0
                # Store some model values for debugging
                d_debug[10 + idx, 1] = shared_mem[idx]  # data
                d_debug[10 + idx, 2] = model_val        # model
                d_debug[10 + idx, 3] = gaussian         # gaussian term
                d_debug[10 + idx, 4] = T(idx)           # pixel index
            end
            
            if model_val > T(1e-10)
                # Derivatives
                d_x = gaussian * dx / sigma^2
                d_y = gaussian * dy / sigma^2
                d_n = gaussian
                d_bg = T(1)
                
                residual = (shared_mem[idx] - model_val) / model_val
                
                grad_x += residual * d_x
                grad_y += residual * d_y
                grad_n += residual * d_n
                grad_bg += residual * d_bg
            end
        end
    end
    
    # Reduce gradients
    grad_x = GaussMLE.GaussGPU.warp_reduce_sum(grad_x)
    grad_y = GaussMLE.GaussGPU.warp_reduce_sum(grad_y)
    grad_n = GaussMLE.GaussGPU.warp_reduce_sum(grad_n)
    grad_bg = GaussMLE.GaussGPU.warp_reduce_sum(grad_bg)
    
    if lane_id == 0
        d_debug[2, 1] = grad_x
        d_debug[2, 2] = grad_y
        d_debug[2, 3] = grad_n
        d_debug[2, 4] = grad_bg
    end
    
    return nothing
end

# Test case
roi_size = 7
x_true = 4.0f0
y_true = 4.0f0
n_true = 1000f0
bg_true = 10f0
sigma = 1.5f0

# Generate Gaussian
roi = zeros(Float32, roi_size, roi_size)
for i in 1:roi_size
    for j in 1:roi_size
        xi = Float32(i)
        yi = Float32(j)
        dx = xi - x_true
        dy = yi - y_true
        gauss = n_true * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
        roi[i, j] = bg_true + gauss
    end
end

println("Newton-Raphson Debug")
println("True: x=$x_true, y=$y_true, n=$n_true, bg=$bg_true")

# Run debug kernel
d_data = CuArray(reshape(roi, roi_size, roi_size, 1))
d_debug = CUDA.zeros(Float32, 30, 4)

shmem_size = sizeof(Float32) * roi_size * roi_size
@cuda threads=32 blocks=1 shmem=shmem_size debug_nr_kernel(d_data, d_debug, Int32(roi_size), Int32(1))
CUDA.synchronize()

debug_vals = Array(d_debug)

println("\nInitial parameters:")
println("  x=$(debug_vals[1,1]), y=$(debug_vals[1,2]), n=$(debug_vals[1,3]), bg=$(debug_vals[1,4])")

println("\nGradients after first eval:")
println("  grad_x=$(debug_vals[2,1])")
println("  grad_y=$(debug_vals[2,2])")
println("  grad_n=$(debug_vals[2,3])")
println("  grad_bg=$(debug_vals[2,4])")

println("\nFirst 10 pixels (data, model, gaussian, idx):")
for i in 1:10
    if debug_vals[10+i, 4] > 0
        println("  Pixel $(Int(debug_vals[10+i,4])): data=$(debug_vals[10+i,1]), model=$(debug_vals[10+i,2]), gauss=$(debug_vals[10+i,3])")
    end
end