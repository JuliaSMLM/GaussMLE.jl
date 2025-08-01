"""
Clean CUDA kernel implementation for GaussMLE
Optimized for compilation and performance
"""

using CUDA
using StaticArrays

# Constants
const WARP_SIZE = Int32(32)
const MAX_ITERATIONS = 50
const TOLERANCE = Float32(1e-6)

# Kernel for 2D Gaussian fitting using Newton-Raphson
function gaussian_fit_kernel!(data::CuDeviceArray{Float32,3}, 
                             params::CuDeviceArray{Float32,2},
                             crlb::CuDeviceArray{Float32,2},
                             roi_size::Int32)
    # Global thread index
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    n_rois = size(data, 3)
    
    # One thread per ROI
    if tid > n_rois
        return nothing
    end
    
    # Working with Float32 for GPU efficiency
    T = Float32
    
    # PSF width
    sigma = T(1.5)
    sigma2 = sigma * sigma
    two_sigma2 = T(2) * sigma2
    norm_factor = T(1) / (T(2π) * sigma2)
    
    # Initialize parameters using center of mass
    # First pass: find min (background estimate)
    bg = typemax(T)
    @inbounds for j in 1:roi_size
        for i in 1:roi_size
            val = data[i, j, tid]
            bg = min(bg, val)
        end
    end
    
    # Second pass: center of mass
    sum_val = T(0)
    sum_x = T(0)
    sum_y = T(0)
    
    @inbounds for j in 1:roi_size
        for i in 1:roi_size
            val = max(data[i, j, tid] - bg, T(0))
            sum_val += val
            sum_x += val * T(j)  # x corresponds to column j
            sum_y += val * T(i)  # y corresponds to row i
        end
    end
    
    # Initial parameter estimates
    x = sum_val > T(0) ? sum_x / sum_val : T(roi_size + 1) / T(2)
    y = sum_val > T(0) ? sum_y / sum_val : T(roi_size + 1) / T(2)
    intensity = max(sum_val, T(100))
    
    # Newton-Raphson iterations
    for iter in 1:MAX_ITERATIONS
        # Initialize gradients and Hessian
        g_x = T(0)
        g_y = T(0)
        g_n = T(0)
        g_bg = T(0)
        
        h_xx = T(0)
        h_yy = T(0)
        h_nn = T(0)
        h_bb = T(0)
        h_xy = T(0)
        h_xn = T(0)
        h_xb = T(0)
        h_yn = T(0)
        h_yb = T(0)
        h_nb = T(0)
        
        # Accumulate over pixels
        @inbounds for j in 1:roi_size
            for i in 1:roi_size
                # Pixel coordinates (1-based)
                xi = T(j)
                yi = T(i)
                
                # Distance from center
                dx = xi - x
                dy = yi - y
                dist2 = dx*dx + dy*dy
                
                # Gaussian value
                gauss = intensity * norm_factor * exp(-dist2 / two_sigma2)
                model = bg + gauss
                
                # Skip if model too small
                if model < T(1e-10)
                    continue
                end
                
                # Data value
                data_val = data[i, j, tid]
                
                # Derivatives
                d_x = gauss * dx / sigma2
                d_y = gauss * dy / sigma2
                d_n = gauss / intensity
                d_bg = T(1)
                
                # Residual and weight
                residual = (data_val - model) / model
                weight = T(1) / model
                
                # Accumulate gradient
                g_x += residual * d_x
                g_y += residual * d_y
                g_n += residual * d_n
                g_bg += residual * d_bg
                
                # Accumulate Hessian (Fisher information)
                h_xx += weight * d_x * d_x
                h_yy += weight * d_y * d_y
                h_nn += weight * d_n * d_n
                h_bb += weight * d_bg * d_bg
                h_xy += weight * d_x * d_y
                h_xn += weight * d_x * d_n
                h_xb += weight * d_x * d_bg
                h_yn += weight * d_y * d_n
                h_yb += weight * d_y * d_bg
                h_nb += weight * d_n * d_bg
            end
        end
        
        # Add regularization
        reg = T(1e-6)
        h_xx += reg
        h_yy += reg
        h_nn += reg
        h_bb += reg
        
        # Solve 2x2 position system
        det_xy = h_xx * h_yy - h_xy * h_xy
        if abs(det_xy) < T(1e-10)
            break
        end
        
        # Decouple from intensity/background
        g_x_adj = g_x - (h_xn * g_n) / h_nn - (h_xb * g_bg) / h_bb
        g_y_adj = g_y - (h_yn * g_n) / h_nn - (h_yb * g_bg) / h_bb
        
        # Newton step for position
        inv_det = T(1) / det_xy
        dx = inv_det * (h_yy * g_x_adj - h_xy * g_y_adj)
        dy = inv_det * (-h_xy * g_x_adj + h_xx * g_y_adj)
        
        # Update intensity and background
        g_n_adj = g_n - h_xn * dx - h_yn * dy
        g_bg_adj = g_bg - h_xb * dx - h_yb * dy
        
        dn = g_n_adj / h_nn
        dbg = g_bg_adj / h_bb
        
        # Apply updates
        x_new = x + dx
        y_new = y + dy
        n_new = intensity + dn
        bg_new = bg + dbg
        
        # Bounds check
        x_new = clamp(x_new, T(1), T(roi_size))
        y_new = clamp(y_new, T(1), T(roi_size))
        n_new = max(n_new, T(1))
        bg_new = max(bg_new, T(0.01))
        
        # Check convergence
        rel_change = (abs(dx) + abs(dy)) / T(roi_size) + 
                     abs(dn) / max(intensity, T(1)) + 
                     abs(dbg) / max(bg, T(1))
        
        # Update parameters
        x = x_new
        y = y_new
        intensity = n_new
        bg = bg_new
        
        if rel_change < TOLERANCE
            break
        end
    end
    
    # Store results
    @inbounds begin
        params[1, tid] = x
        params[2, tid] = y
        params[3, tid] = intensity
        params[4, tid] = bg
        
        # Placeholder CRLB values
        crlb[1, tid] = T(0.1)
        crlb[2, tid] = T(0.1)
        crlb[3, tid] = T(10)
        crlb[4, tid] = T(0.5)
    end
    
    return nothing
end

# Launch function
function launch_gaussian_fit!(d_data::CuArray{Float32,3}, 
                             d_params::CuArray{Float32,2}, 
                             d_crlb::CuArray{Float32,2})
    n_rois = size(d_data, 3)
    roi_size = Int32(size(d_data, 1))
    
    # Configure launch
    threads = 256
    blocks = cld(n_rois, threads)
    
    # Launch kernel
    @cuda threads=threads blocks=blocks gaussian_fit_kernel!(
        d_data, d_params, d_crlb, roi_size
    )
    
    return nothing
end

# Export the launch function
export launch_gaussian_fit!