"""
CUDA kernel implementations for GaussMLE - FIXED VERSION V2
Using warp-level parallelism: one warp (32 threads) per ROI
Fixed coordinate system and gradient computation issues
"""

using CUDA
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

# Simple 4x4 linear solver for GPU
@inline function solve_4x4!(A::CuDeviceArray{T,2}, b::CuDeviceArray{T,1}, x::CuDeviceArray{T,1}) where T
    """Solve 4x4 system Ax=b using Gaussian elimination with partial pivoting"""
    # Copy A and b to avoid modifying originals
    for i in 1:4
        for j in 1:4
            A[i,j] = A[i,j]
        end
        x[i] = b[i]
    end
    
    # Forward elimination
    for k in 1:3
        # Find pivot
        max_val = abs(A[k,k])
        max_row = k
        for i in (k+1):4
            if abs(A[i,k]) > max_val
                max_val = abs(A[i,k])
                max_row = i
            end
        end
        
        # Swap rows if needed
        if max_row != k
            for j in k:4
                temp = A[k,j]
                A[k,j] = A[max_row,j]
                A[max_row,j] = temp
            end
            temp = x[k]
            x[k] = x[max_row]
            x[max_row] = temp
        end
        
        # Eliminate
        if abs(A[k,k]) > T(1e-10)
            for i in (k+1):4
                factor = A[i,k] / A[k,k]
                for j in (k+1):4
                    A[i,j] -= factor * A[k,j]
                end
                x[i] -= factor * x[k]
            end
        end
    end
    
    # Back substitution
    for i in 4:-1:1
        if abs(A[i,i]) > T(1e-10)
            for j in (i+1):4
                x[i] -= A[i,j] * x[j]
            end
            x[i] /= A[i,i]
        else
            x[i] = T(0)
        end
    end
    
    return nothing
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
    shared_mem = @cuDynamicSharedMem(T, roi_size * roi_size * 3 + 4 * 4 + 4 + 4)
    roi_data = view(shared_mem, 1:roi_size*roi_size)
    model_vals = view(shared_mem, roi_size*roi_size+1:2*roi_size*roi_size)
    # Space for 4x4 Hessian, gradient, and solution vectors
    hess_start = 2*roi_size*roi_size + 1
    grad_start = hess_start + 16
    delta_start = grad_start + 4
    
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
        bg = max(min_val, T(0.01))
        
        # Center of mass above background
        sum_above_bg = T(0)
        sum_x_weighted = T(0)
        sum_y_weighted = T(0)
        
        for idx in 1:n_pixels
            val_above_bg = max(roi_data[idx] - bg, T(0))
            # Use 1-based indexing for pixel positions
            row = Float32((idx - 1) ÷ roi_size + 1)
            col = Float32((idx - 1) % roi_size + 1)
            sum_above_bg += val_above_bg
            # Convert to 0-based coordinates for Gaussian center
            sum_x_weighted += val_above_bg * (col - T(1))
            sum_y_weighted += val_above_bg * (row - T(1))
        end
        
        if sum_above_bg > T(10)
            x = sum_x_weighted / sum_above_bg
            y = sum_y_weighted / sum_above_bg
            intensity = sum_above_bg
        else
            # Center of ROI in 0-based coordinates
            x = T((roi_size - 1) * 0.5)
            y = T((roi_size - 1) * 0.5)
            intensity = T(100)
        end
        
        # Ensure parameters are within bounds
        x = clamp(x, T(0), T(roi_size - 1))
        y = clamp(y, T(0), T(roi_size - 1))
        intensity = max(intensity, T(1))
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
        
        # Initialize gradient and Hessian accumulators
        grad_x = T(0)
        grad_y = T(0)
        grad_n = T(0)
        grad_bg = T(0)
        
        hess_xx = T(0)
        hess_yy = T(0)
        hess_nn = T(0)
        hess_bb = T(0)
        hess_xy = T(0)
        hess_xn = T(0)
        hess_xb = T(0)
        hess_yn = T(0)
        hess_yb = T(0)
        hess_nb = T(0)
        
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
                    d_n = gaussian_val / intensity
                    d_bg = T(1.0)
                    
                    # For MLE with Poisson noise
                    factor = (data_val / model_val - T(1.0))
                    
                    # Accumulate gradients
                    grad_x += factor * d_x
                    grad_y += factor * d_y
                    grad_n += factor * d_n
                    grad_bg += factor * d_bg
                    
                    # Hessian elements (Fisher information)
                    weight = data_val / (model_val * model_val)
                    
                    hess_xx += weight * d_x * d_x
                    hess_yy += weight * d_y * d_y
                    hess_nn += weight * d_n * d_n
                    hess_bb += weight * d_bg * d_bg
                    hess_xy += weight * d_x * d_y
                    hess_xn += weight * d_x * d_n
                    hess_xb += weight * d_x * d_bg
                    hess_yn += weight * d_y * d_n
                    hess_yb += weight * d_y * d_bg
                    hess_nb += weight * d_n * d_bg
                end
            end
        end
        
        # Reduce gradients and Hessian across warp
        grad_x = warp_reduce_sum(grad_x)
        grad_y = warp_reduce_sum(grad_y)
        grad_n = warp_reduce_sum(grad_n)
        grad_bg = warp_reduce_sum(grad_bg)
        
        hess_xx = warp_reduce_sum(hess_xx)
        hess_yy = warp_reduce_sum(hess_yy)
        hess_nn = warp_reduce_sum(hess_nn)
        hess_bb = warp_reduce_sum(hess_bb)
        hess_xy = warp_reduce_sum(hess_xy)
        hess_xn = warp_reduce_sum(hess_xn)
        hess_xb = warp_reduce_sum(hess_xb)
        hess_yn = warp_reduce_sum(hess_yn)
        hess_yb = warp_reduce_sum(hess_yb)
        hess_nb = warp_reduce_sum(hess_nb)
        
        # Lane 0 solves the 4x4 system
        local converged_iter::Bool = false
        if lane_id == 0
            # Build 4x4 Hessian matrix in shared memory
            shared_mem[hess_start + 0] = hess_xx  # H[1,1]
            shared_mem[hess_start + 1] = hess_xy  # H[1,2]
            shared_mem[hess_start + 2] = hess_xn  # H[1,3]
            shared_mem[hess_start + 3] = hess_xb  # H[1,4]
            
            shared_mem[hess_start + 4] = hess_xy  # H[2,1]
            shared_mem[hess_start + 5] = hess_yy  # H[2,2]
            shared_mem[hess_start + 6] = hess_yn  # H[2,3]
            shared_mem[hess_start + 7] = hess_yb  # H[2,4]
            
            shared_mem[hess_start + 8] = hess_xn  # H[3,1]
            shared_mem[hess_start + 9] = hess_yn  # H[3,2]
            shared_mem[hess_start + 10] = hess_nn # H[3,3]
            shared_mem[hess_start + 11] = hess_nb # H[3,4]
            
            shared_mem[hess_start + 12] = hess_xb # H[4,1]
            shared_mem[hess_start + 13] = hess_yb # H[4,2]
            shared_mem[hess_start + 14] = hess_nb # H[4,3]
            shared_mem[hess_start + 15] = hess_bb # H[4,4]
            
            # Build gradient vector
            shared_mem[grad_start + 0] = grad_x
            shared_mem[grad_start + 1] = grad_y
            shared_mem[grad_start + 2] = grad_n
            shared_mem[grad_start + 3] = grad_bg
            
            # Regularize diagonal
            for i in 0:3
                idx = hess_start + i * 5  # Diagonal elements
                shared_mem[idx] = max(shared_mem[idx], T(1e-6))
            end
            
            # Solve using simplified approach for stability
            # Separate 2x2 system for position
            det_xy = hess_xx * hess_yy - hess_xy * hess_xy
            if abs(det_xy) > T(1e-10)
                inv_det = T(1) / det_xy
                delta_x = inv_det * (hess_yy * grad_x - hess_xy * grad_y)
                delta_y = inv_det * (hess_xx * grad_y - hess_xy * grad_x)
            else
                delta_x = T(0)
                delta_y = T(0)
            end
            
            # Separate updates for intensity and background
            delta_n = grad_n / max(hess_nn, T(1e-6))
            delta_bg = grad_bg / max(hess_bb, T(1e-6))
            
            # Apply updates with damping
            damping = T(0.7)
            x_new = x + damping * delta_x
            y_new = y + damping * delta_y
            n_new = intensity + damping * delta_n
            bg_new = bg + damping * delta_bg
            
            # Bounds constraints
            x_new = clamp(x_new, T(-0.5), T(roi_size - 0.5))
            y_new = clamp(y_new, T(-0.5), T(roi_size - 0.5))
            n_new = max(n_new, T(1))
            bg_new = max(bg_new, T(0.01))
            
            # Check convergence
            update_norm = sqrt(delta_x^2 + delta_y^2 + 
                             (delta_n/intensity)^2 + (delta_bg/bg)^2)
            
            converged_iter = update_norm < tolerance
            
            # Update parameters
            x = x_new
            y = y_new
            intensity = n_new
            bg = bg_new
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
        
        # Store uncertainties (sqrt of diagonal of inverse Hessian)
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
    
    # Shared memory size: 3 arrays of roi_size² + space for 4x4 matrix ops
    shmem_size = sizeof(T) * (roi_size * roi_size * 3 + 16 + 4 + 4)
    
    # Launch kernel
    @cuda threads=threads_per_block blocks=blocks shmem=shmem_size cuda_fit_kernel_fixed!(
        d_data, d_params, d_crlb, roi_size, Int32(n_rois), tolerance, max_iter
    )
    
    return nothing
end