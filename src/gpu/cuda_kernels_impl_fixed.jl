"""
CUDA kernel implementations for GaussMLE - FIXED VERSION
Using warp-level parallelism: one warp (32 threads) per ROI
"""

using CUDA
using StaticArrays
using LinearAlgebra

# Constants for kernel configuration
const WARP_SIZE = 32
const MAX_ITERATIONS = 50
const DEFAULT_TOLERANCE = Float32(1e-6)

# Helper functions for warp-level operations
@inline function warp_reduce_sum(val::T) where T
    """Reduce sum across warp using shuffle operations"""
    # Butterfly reduction pattern
    val += CUDA.shfl_down_sync(0xffffffff, val, 16)
    val += CUDA.shfl_down_sync(0xffffffff, val, 8)
    val += CUDA.shfl_down_sync(0xffffffff, val, 4)
    val += CUDA.shfl_down_sync(0xffffffff, val, 2)
    val += CUDA.shfl_down_sync(0xffffffff, val, 1)
    return val
end

@inline function warp_all(predicate::Bool)
    """Check if all threads in warp satisfy predicate"""
    return CUDA.all_sync(0xffffffff, predicate)
end

# Gaussian model evaluation
@inline function evaluate_gaussian(x::T, y::T, intensity::T, bg::T, 
                                 i::Int32, j::Int32, roi_size::Int32) where T
    """Evaluate 2D Gaussian at pixel (i,j) - expecting 1-based indices"""
    # Convert to 0-indexed for computation
    xi = T(i - 1)
    yi = T(j - 1)
    
    # Standard PSF width (could be parameterized)
    sigma = T(1.5)
    two_sigma_sq = T(2.0) * sigma * sigma
    
    # Gaussian evaluation
    dx = xi - x
    dy = yi - y
    exponent = -(dx*dx + dy*dy) / two_sigma_sq
    
    return bg + intensity * CUDA.exp(exponent) / (T(2π) * sigma * sigma)
end

# Main fitting kernel
function cuda_fit_kernel_fixed!(d_data::CuDeviceArray{T,3}, 
                         d_params::CuDeviceArray{T,2},
                         d_crlb::CuDeviceArray{T,2},
                         roi_size::Int32,
                         n_rois::Int32,
                         tolerance::T,
                         max_iter::Int32) where T
    
    # Thread indexing - one warp per ROI
    tid = threadIdx().x
    warp_id = (blockIdx().x - 1) * (blockDim().x ÷ WARP_SIZE) + (tid - 1) ÷ WARP_SIZE
    lane_id = (tid - 1) % WARP_SIZE
    
    # Bounds check
    if warp_id >= n_rois
        return nothing
    end
    
    # Shared memory for ROI data and working arrays
    shared_mem = @cuDynamicSharedMem(T, roi_size * roi_size * 3)
    roi_data = view(shared_mem, 1:roi_size*roi_size)
    model_vals = view(shared_mem, roi_size*roi_size+1:2*roi_size*roi_size)
    
    # Load ROI data collaboratively
    n_pixels = roi_size * roi_size
    pixels_per_thread = (n_pixels + WARP_SIZE - 1) ÷ WARP_SIZE
    
    for p in 1:pixels_per_thread
        pixel_idx = lane_id * pixels_per_thread + p
        if pixel_idx <= n_pixels
            i = (pixel_idx - 1) ÷ roi_size + 1
            j = (pixel_idx - 1) % roi_size + 1
            roi_data[pixel_idx] = d_data[i, j, warp_id + 1]
        end
    end
    
    # Wait for all threads to finish loading
    CUDA.sync_warp()
    
    # Initialize parameters using center of mass
    local x::T, y::T, intensity::T, bg::T
    
    if lane_id == 0
        # Find minimum as background estimate
        min_val = roi_data[1]
        for idx in 2:n_pixels
            min_val = min(min_val, roi_data[idx])
        end
        bg = min_val
        
        # Center of mass above background - FIXED to use 1-based indexing
        sum_above_bg = T(0)
        sum_x_weighted = T(0)
        sum_y_weighted = T(0)
        
        for idx in 1:n_pixels
            val_above_bg = max(roi_data[idx] - bg, T(0))
            # Convert to 1-based i,j for center of mass calculation
            i = (idx - 1) ÷ roi_size + 1  # 1-based row
            j = (idx - 1) % roi_size + 1  # 1-based column
            sum_above_bg += val_above_bg
            # Use 1-based indices for weighting, then convert to 0-based
            sum_x_weighted += val_above_bg * T(j - 1)  # x corresponds to column (j)
            sum_y_weighted += val_above_bg * T(i - 1)  # y corresponds to row (i)
        end
        
        if sum_above_bg > T(0)
            x = sum_x_weighted / sum_above_bg
            y = sum_y_weighted / sum_above_bg
            # Estimate intensity from total counts above background
            sigma = T(1.5)
            intensity = sum_above_bg
        else
            x = T((roi_size - 1) / 2)
            y = T((roi_size - 1) / 2)
            intensity = T(100)
        end
    else
        x = T(0)
        y = T(0)
        intensity = T(0)
        bg = T(0)
    end
    
    # Broadcast parameters to all lanes
    x = CUDA.shfl_sync(0xffffffff, x, 1)
    y = CUDA.shfl_sync(0xffffffff, y, 1)
    intensity = CUDA.shfl_sync(0xffffffff, intensity, 1)
    bg = CUDA.shfl_sync(0xffffffff, bg, 1)
    
    # Newton-Raphson iterations
    converged = false
    
    for iter in 1:max_iter
        # Evaluate model at current parameters
        for p in 1:pixels_per_thread
            pixel_idx = lane_id * pixels_per_thread + p
            if pixel_idx <= n_pixels
                i = (pixel_idx - 1) ÷ roi_size + 1
                j = (pixel_idx - 1) % roi_size + 1
                model_vals[pixel_idx] = evaluate_gaussian(x, y, intensity, bg, 
                                                         Int32(i), Int32(j), roi_size)
            end
        end
        
        CUDA.sync_warp()
        
        # Compute gradients and Hessian elements
        # Each thread accumulates its portion
        grad_x = T(0)
        grad_y = T(0)
        grad_n = T(0)
        grad_bg = T(0)
        
        # Full 4x4 Hessian matrix elements
        hess = @MMatrix zeros(T, 4, 4)
        
        sigma = T(1.5)
        sigma_sq = sigma * sigma
        
        for p in 1:pixels_per_thread
            pixel_idx = lane_id * pixels_per_thread + p
            if pixel_idx <= n_pixels
                i = (pixel_idx - 1) ÷ roi_size + 1
                j = (pixel_idx - 1) % roi_size + 1
                
                data_val = roi_data[pixel_idx]
                model_val = model_vals[pixel_idx]
                
                # Skip if model value too small
                if model_val > T(1e-10)
                    # Compute derivatives (0-based coordinates for Gaussian)
                    xi = T(i - 1)
                    yi = T(j - 1)
                    dx = xi - x
                    dy = yi - y
                    
                    # Gaussian part without background
                    gaussian_val = model_val - bg
                    
                    # First derivatives
                    d_x = gaussian_val * dx / sigma_sq
                    d_y = gaussian_val * dy / sigma_sq
                    d_n = gaussian_val / intensity  # FIXED
                    d_bg = T(1.0)
                    
                    # Gradient for MLE under Poisson noise
                    factor = (data_val / model_val - T(1.0))
                    
                    # Accumulate gradients
                    grad_x += factor * d_x
                    grad_y += factor * d_y
                    grad_n += factor * d_n
                    grad_bg += factor * d_bg
                    
                    # Hessian (Fisher information matrix)
                    weight = data_val / (model_val * model_val)
                    
                    # Build derivative vector
                    derivs = @SVector [d_x, d_y, d_n, d_bg]
                    
                    # Accumulate Hessian
                    for i in 1:4
                        for j in i:4
                            hess[i,j] += weight * derivs[i] * derivs[j]
                        end
                    end
                end
            end
        end
        
        # Reduce gradients and Hessian across warp
        grad_x = warp_reduce_sum(grad_x)
        grad_y = warp_reduce_sum(grad_y)
        grad_n = warp_reduce_sum(grad_n)
        grad_bg = warp_reduce_sum(grad_bg)
        
        # Reduce Hessian elements
        for i in 1:4
            for j in i:4
                hess[i,j] = warp_reduce_sum(hess[i,j])
                if j > i
                    hess[j,i] = hess[i,j]  # Symmetric matrix
                end
            end
        end
        
        # Lane 0 solves the 4x4 system
        local converged_iter::Bool = false
        if lane_id == 0
            # Build gradient vector
            grad = @SVector [grad_x, grad_y, grad_n, grad_bg]
            
            # Regularize Hessian diagonal to ensure positive definite
            for i in 1:4
                hess[i,i] = max(hess[i,i], T(1e-10))
            end
            
            # Solve H * delta = grad using Cholesky decomposition or direct inverse
            # For 4x4, we can use a simple LU decomposition or direct formula
            try
                # Simple iterative solver or direct inverse for 4x4
                delta = hess \ grad
                
                # Apply updates with damping for stability
                damping = T(1.0)
                x_new = x + damping * delta[1]
                y_new = y + damping * delta[2]
                n_new = intensity + damping * delta[3]
                bg_new = bg + damping * delta[4]
                
                # Bounds constraints
                x_new = clamp(x_new, T(-1), T(roi_size))
                y_new = clamp(y_new, T(-1), T(roi_size))
                n_new = max(n_new, T(0.1))
                bg_new = max(bg_new, T(0.01))
                
                # Check convergence
                update_norm = sqrt(delta[1]^2 + delta[2]^2 + 
                                 (delta[3]/intensity)^2 + (delta[4]/bg)^2)
                
                converged_iter = update_norm < tolerance
                
                # Update parameters
                x = x_new
                y = y_new
                intensity = n_new
                bg = bg_new
            catch
                # Matrix singular, stop iterations
                converged_iter = true
            end
        end
        
        # Broadcast convergence flag and new parameters
        converged = CUDA.shfl_sync(0xffffffff, converged_iter, 1)
        if converged
            break
        end
        
        x = CUDA.shfl_sync(0xffffffff, x, 1)
        y = CUDA.shfl_sync(0xffffffff, y, 1)
        intensity = CUDA.shfl_sync(0xffffffff, intensity, 1)
        bg = CUDA.shfl_sync(0xffffffff, bg, 1)
    end
    
    # Store results (lane 0 only)
    if lane_id == 0
        d_params[1, warp_id + 1] = x
        d_params[2, warp_id + 1] = y
        d_params[3, warp_id + 1] = intensity
        d_params[4, warp_id + 1] = bg
        
        # Compute CRLB from inverse Hessian diagonal
        # For now, use simple estimates
        d_crlb[1, warp_id + 1] = T(0.1)
        d_crlb[2, warp_id + 1] = T(0.1)
        d_crlb[3, warp_id + 1] = T(10.0)
        d_crlb[4, warp_id + 1] = T(0.5)
    end
    
    return nothing
end

# Wrapper function to launch kernel
function launch_cuda_fit_kernel_fixed!(d_data::CuArray{T,3}, d_params::CuArray{T,2}, 
                               d_crlb::CuArray{T,2}, roi_size::Int32,
                               tolerance::T=T(DEFAULT_TOLERANCE), 
                               max_iter::Int32=Int32(MAX_ITERATIONS)) where T
    
    n_rois = size(d_data, 3)
    
    # Configure kernel launch
    warps_per_block = 4  # 128 threads per block
    threads_per_block = warps_per_block * WARP_SIZE
    blocks = (n_rois + warps_per_block - 1) ÷ warps_per_block
    
    # Shared memory size: 3 arrays of roi_size²
    shmem_size = sizeof(T) * roi_size * roi_size * 3
    
    # Launch kernel
    @cuda threads=threads_per_block blocks=blocks shmem=shmem_size cuda_fit_kernel_fixed!(
        d_data, d_params, d_crlb, roi_size, Int32(n_rois), tolerance, max_iter
    )
    
    return nothing
end