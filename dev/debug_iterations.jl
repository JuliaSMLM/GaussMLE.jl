#!/usr/bin/env julia

using GaussMLE
using CUDA
using Printf

# Custom kernel that saves iteration history
function debug_iteration_kernel(d_data::CuDeviceArray{T,3}, 
                               d_history::CuDeviceArray{T,2},
                               roi_size::Int32) where T
    tid = threadIdx().x
    warp_id = (blockIdx().x - 1) * (blockDim().x ÷ 32) + (tid - 1) ÷ 32
    lane_id = (tid - 1) % 32
    
    if warp_id >= 1
        return nothing
    end
    
    # Shared memory for data
    n_pixels = roi_size * roi_size
    shared_mem = @cuDynamicSharedMem(T, n_pixels)
    
    # Load data
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
    
    # Initialize
    local x::T = T(0)
    local y::T = T(0) 
    local intensity::T = T(0)
    local bg::T = T(0)
    
    if lane_id == 0
        # Background estimate
        min_val = shared_mem[1]
        for idx in 2:n_pixels
            min_val = min(min_val, shared_mem[idx])
        end
        bg = min_val
        
        # Center of mass
        sum_above_bg = T(0)
        sum_x_weighted = T(0)
        sum_y_weighted = T(0)
        
        for idx in 1:n_pixels
            val_above_bg = max(shared_mem[idx] - bg, T(0))
            i = (idx - 1) ÷ roi_size + 1
            j = (idx - 1) % roi_size + 1
            sum_above_bg += val_above_bg
            sum_x_weighted += val_above_bg * T(j)
            sum_y_weighted += val_above_bg * T(i)
        end
        
        if sum_above_bg > T(0)
            x = sum_x_weighted / sum_above_bg
            y = sum_y_weighted / sum_above_bg
            intensity = sum_above_bg
        else
            x = T((roi_size + 1) / 2)
            y = T((roi_size + 1) / 2)
            intensity = T(100)
        end
        
        # Save initial values
        d_history[1, 1] = x
        d_history[2, 1] = y
        d_history[3, 1] = intensity
        d_history[4, 1] = bg
    end
    
    # Broadcast
    x = CUDA.shfl_sync(0xffffffff, x, 1)
    y = CUDA.shfl_sync(0xffffffff, y, 1)
    intensity = CUDA.shfl_sync(0xffffffff, intensity, 1)
    bg = CUDA.shfl_sync(0xffffffff, bg, 1)
    
    # Do 5 iterations
    sigma = T(1.5)
    
    for iter in 1:5
        # Calculate gradients and Hessian
        grad_x = T(0)
        grad_y = T(0)
        grad_n = T(0)
        grad_bg = T(0)
        
        hess_xx = T(0)
        hess_yy = T(0)
        hess_nn = T(0)
        hess_bb = T(0)
        
        for p in 1:pixels_per_thread
            idx = lane_id * pixels_per_thread + p
            if idx <= n_pixels
                i = (idx - 1) ÷ roi_size + 1
                j = (idx - 1) % roi_size + 1
                
                # Model evaluation
                xi = T(i)
                yi = T(j)
                dx = xi - x
                dy = yi - y
                gaussian_term = intensity * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
                model_val = bg + gaussian_term
                
                if model_val > T(1e-10)
                    # Derivatives
                    d_x = gaussian_term * dx / sigma^2
                    d_y = gaussian_term * dy / sigma^2
                    d_n = gaussian_term / intensity
                    d_bg = T(1)
                    
                    residual = (shared_mem[idx] - model_val) / model_val
                    weight = T(1) / model_val
                    
                    grad_x += residual * d_x
                    grad_y += residual * d_y
                    grad_n += residual * d_n
                    grad_bg += residual * d_bg
                    
                    hess_xx += weight * d_x * d_x
                    hess_yy += weight * d_y * d_y
                    hess_nn += weight * d_n * d_n
                    hess_bb += weight * d_bg * d_bg
                end
            end
        end
        
        # Reduce
        grad_x = GaussMLE.GaussGPU.warp_reduce_sum(grad_x)
        grad_y = GaussMLE.GaussGPU.warp_reduce_sum(grad_y)
        grad_n = GaussMLE.GaussGPU.warp_reduce_sum(grad_n)
        grad_bg = GaussMLE.GaussGPU.warp_reduce_sum(grad_bg)
        
        hess_xx = GaussMLE.GaussGPU.warp_reduce_sum(hess_xx)
        hess_yy = GaussMLE.GaussGPU.warp_reduce_sum(hess_yy)
        hess_nn = GaussMLE.GaussGPU.warp_reduce_sum(hess_nn)
        hess_bb = GaussMLE.GaussGPU.warp_reduce_sum(hess_bb)
        
        # Update (lane 0)
        if lane_id == 0
            # Save gradients and Hessian
            col = (iter - 1) * 4 + 2
            d_history[1, col] = grad_x
            d_history[2, col] = grad_y
            d_history[3, col] = grad_n
            d_history[4, col] = grad_bg
            
            d_history[1, col+1] = hess_xx
            d_history[2, col+1] = hess_yy
            d_history[3, col+1] = hess_nn
            d_history[4, col+1] = hess_bb
            
            # Simple Newton update
            hess_xx_reg = hess_xx + T(1e-6)
            hess_yy_reg = hess_yy + T(1e-6)
            hess_nn_reg = hess_nn + T(1e-6)
            hess_bb_reg = hess_bb + T(1e-6)
            
            update_x = grad_x / hess_xx_reg
            update_y = grad_y / hess_yy_reg
            update_n = grad_n / hess_nn_reg
            update_bg = grad_bg / hess_bb_reg
            
            # Save updates
            d_history[1, col+2] = update_x
            d_history[2, col+2] = update_y
            d_history[3, col+2] = update_n
            d_history[4, col+2] = update_bg
            
            # Apply updates
            x = x + update_x
            y = y + update_y
            intensity = intensity + update_n
            bg = bg + update_bg
            
            # Bounds
            x = clamp(x, T(1), T(roi_size))
            y = clamp(y, T(1), T(roi_size))
            intensity = max(intensity, T(1))
            bg = max(bg, T(0.01))
            
            # Save new values
            d_history[1, col+3] = x
            d_history[2, col+3] = y
            d_history[3, col+3] = intensity
            d_history[4, col+3] = bg
        end
        
        # Broadcast new values
        x = CUDA.shfl_sync(0xffffffff, x, 1)
        y = CUDA.shfl_sync(0xffffffff, y, 1)
        intensity = CUDA.shfl_sync(0xffffffff, intensity, 1)
        bg = CUDA.shfl_sync(0xffffffff, bg, 1)
    end
    
    return nothing
end

# Test
roi_size = 7
x_true = 4.0f0
y_true = 4.0f0
n_true = 1000f0
bg_true = 10f0
sigma = 1.5f0

# Generate perfect Gaussian
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

println("Debug Iteration History")
println("True: x=$x_true, y=$y_true, n=$n_true, bg=$bg_true")
println()

# Run debug kernel
d_data = CuArray(reshape(roi, roi_size, roi_size, 1))
d_history = CUDA.zeros(Float32, 4, 22)  # params x (init + 5*(grad,hess,update,new))

shmem = sizeof(Float32) * roi_size * roi_size
@cuda threads=32 blocks=1 shmem=shmem debug_iteration_kernel(d_data, d_history, Int32(roi_size))
CUDA.synchronize()

history = Array(d_history)

# Print history
println("Iteration history:")
println("  Init: x=$(history[1,1]), y=$(history[2,1]), n=$(history[3,1]), bg=$(history[4,1])")

for iter in 1:5
    col = (iter - 1) * 4 + 2
    println("\nIter $iter:")
    println("  Gradients: x=$(history[1,col]), y=$(history[2,col]), n=$(history[3,col]), bg=$(history[4,col])")
    println("  Hessian diag: xx=$(history[1,col+1]), yy=$(history[2,col+1]), nn=$(history[3,col+1]), bb=$(history[4,col+1])")
    println("  Updates: x=$(history[1,col+2]), y=$(history[2,col+2]), n=$(history[3,col+2]), bg=$(history[4,col+2])")
    println("  New vals: x=$(history[1,col+3]), y=$(history[2,col+3]), n=$(history[3,col+3]), bg=$(history[4,col+3])")
end