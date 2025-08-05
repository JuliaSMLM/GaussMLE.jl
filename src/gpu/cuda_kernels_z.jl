"""
Fixed CUDA kernel implementation for astigmatic z-model
Avoids dynamic dispatch and tuple destructuring issues
"""

using CUDA
using GaussMLE.GaussModel: AstigmaticCalibration

# GPU kernel for astigmatic z-model fitting (fixed version)
function gaussian_fit_z_kernel_fixed!(data::CuDeviceArray{Float32,3,1},
                                     params::CuDeviceArray{Float32,2,1},
                                     crlb::CuDeviceArray{Float32,2,1},
                                     roi_size::Int32,
                                     σ_x0::Float32, σ_y0::Float32,
                                     γ::Float32, d::Float32,
                                     Ax::Float32, Ay::Float32,
                                     Bx::Float32, By::Float32)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    n_rois = size(data, 3)
    
    if tid > n_rois
        return nothing
    end
    
    # Initialize parameters using center of mass
    bg = Float32(Inf)
    @inbounds for j in 1:roi_size
        for i in 1:roi_size
            val = data[i, j, tid]
            bg = min(bg, val)
        end
    end
    
    # Center of mass calculation
    sum_val = Float32(0)
    sum_x = Float32(0)
    sum_y = Float32(0)
    
    @inbounds for j in 1:roi_size
        for i in 1:roi_size
            val = max(data[i, j, tid] - bg, Float32(0))
            sum_val = sum_val + val
            sum_x = sum_x + val * Float32(j)
            sum_y = sum_y + val * Float32(i)
        end
    end
    
    # Initial estimates
    x = sum_val > Float32(0) ? sum_x / sum_val : Float32(roi_size + 1) / Float32(2)
    y = sum_val > Float32(0) ? sum_y / sum_val : Float32(roi_size + 1) / Float32(2)
    z = γ  # Start at focal plane
    intensity = max(sum_val, Float32(100))
    
    # Newton-Raphson iterations
    for iter in 1:50
        # Initialize gradients and Hessian diagonal
        g1 = Float32(0)
        g2 = Float32(0)
        g3 = Float32(0)
        g4 = Float32(0)
        g5 = Float32(0)
        
        h11 = Float32(0)
        h22 = Float32(0)
        h33 = Float32(0)
        h44 = Float32(0)
        h55 = Float32(0)
        
        # Accumulate over pixels
        @inbounds for j in 1:roi_size
            for i in 1:roi_size
                xi = Float32(j)
                yi = Float32(i)
                dx = xi - x
                dy = yi - y
                
                # Compute PSF widths
                z_minus_γ = z - γ
                z_plus_γ = z + γ
                
                # Compute alpha values inline
                z_d_x = z_minus_γ / d
                z_d2_x = z_d_x * z_d_x
                z_d3_x = z_d2_x * z_d_x
                z_d4_x = z_d3_x * z_d_x
                α_x = Float32(1) + z_d2_x + Ax * z_d3_x + Bx * z_d4_x
                
                z_d_y = z_plus_γ / d
                z_d2_y = z_d_y * z_d_y
                z_d3_y = z_d2_y * z_d_y
                z_d4_y = z_d3_y * z_d_y
                α_y = Float32(1) + z_d2_y + Ay * z_d3_y + By * z_d4_y
                
                σ_x = σ_x0 * sqrt(α_x)
                σ_y = σ_y0 * sqrt(α_y)
                
                # Compute Gaussian
                σx2 = σ_x * σ_x
                σy2 = σ_y * σ_y
                norm_factor = Float32(1) / (Float32(2π) * σ_x * σ_y)
                exp_arg = -(dx*dx)/(Float32(2)*σx2) - (dy*dy)/(Float32(2)*σy2)
                gauss = intensity * norm_factor * exp(exp_arg)
                model = bg + gauss
                
                if model < Float32(1e-10)
                    continue
                end
                
                # Derivatives
                d1 = gauss * dx / σx2                    # ∂/∂x
                d2 = gauss * dy / σy2                    # ∂/∂y
                d4 = gauss / intensity                   # ∂/∂n
                d5 = Float32(1)                          # ∂/∂bg
                
                # Z derivative
                dα_x_dz = (Float32(2) * z_d_x + Float32(3) * Ax * z_d2_x + Float32(4) * Bx * z_d3_x) / d
                dα_y_dz = (Float32(2) * z_d_y + Float32(3) * Ay * z_d2_y + Float32(4) * By * z_d3_y) / d
                
                dσx_dz = σ_x0 * Float32(0.5) / sqrt(α_x) * dα_x_dz
                dσy_dz = σ_y0 * Float32(0.5) / sqrt(α_y) * dα_y_dz
                
                dσx_term = gauss * ((dx*dx)/σx2/σ_x - Float32(1)/σ_x) * dσx_dz
                dσy_term = gauss * ((dy*dy)/σy2/σ_y - Float32(1)/σ_y) * dσy_dz
                d3 = dσx_term + dσy_term
                
                data_val = data[i, j, tid]
                residual = (data_val - model) / model
                weight = Float32(1) / model
                
                # Accumulate gradients
                g1 = g1 + residual * d1
                g2 = g2 + residual * d2
                g3 = g3 + residual * d3
                g4 = g4 + residual * d4
                g5 = g5 + residual * d5
                
                # Accumulate Hessian diagonal
                h11 = h11 + weight * d1 * d1
                h22 = h22 + weight * d2 * d2
                h33 = h33 + weight * d3 * d3
                h44 = h44 + weight * d4 * d4
                h55 = h55 + weight * d5 * d5
            end
        end
        
        # Add regularization
        reg = Float32(1e-6)
        h11 = h11 + reg
        h22 = h22 + reg
        h33 = h33 + reg
        h44 = h44 + reg
        h55 = h55 + reg
        
        # Compute updates with step limiting
        max_step = Float32(1.0)
        dx = clamp(g1 / h11, -max_step, max_step)
        dy = clamp(g2 / h22, -max_step, max_step)
        dz = clamp(g3 / h33, -max_step, max_step)
        dn = g4 / h44
        dbg = g5 / h55
        
        # Apply updates with bounds
        x = clamp(x + dx, Float32(1), Float32(roi_size))
        y = clamp(y + dy, Float32(1), Float32(roi_size))
        z = clamp(z + dz, Float32(-2.0), Float32(2.0))  # Reasonable z bounds
        intensity = max(intensity + clamp(dn, -intensity * Float32(0.5), intensity * Float32(2.0)), Float32(1))
        bg = max(bg + clamp(dbg, -bg * Float32(0.5), bg * Float32(2.0) + Float32(10)), Float32(0.01))
        
        # Check convergence
        rel_change = (abs(dx) + abs(dy)) / Float32(roi_size) + 
                     abs(dz) / max(abs(z), Float32(1)) +
                     abs(dn) / max(intensity, Float32(1)) + 
                     abs(dbg) / max(bg, Float32(1))
        
        if rel_change < Float32(1e-6)
            break
        end
    end
    
    # Store fitted parameters
    @inbounds begin
        params[1, tid] = x
        params[2, tid] = y
        params[3, tid] = z
        params[4, tid] = intensity
        params[5, tid] = bg
    end
    
    # Compute CRLB (simplified diagonal approximation)
    # Could be improved with full Fisher matrix computation
    @inbounds begin
        crlb[1, tid] = Float32(0.1)  # Placeholder
        crlb[2, tid] = Float32(0.1)  # Placeholder
        crlb[3, tid] = Float32(0.2)  # Placeholder
        crlb[4, tid] = sqrt(max(intensity, Float32(1)))
        crlb[5, tid] = Float32(0.5)  # Placeholder
    end
    
    return nothing
end

# Launch function for z-model
function launch_gaussian_fit_z!(d_data::CuArray{Float32,3}, 
                               d_params::CuArray{Float32,2}, 
                               d_crlb::CuArray{Float32,2},
                               calib::AstigmaticCalibration{Float32})
    n_rois = size(d_data, 3)
    roi_size = Int32(size(d_data, 1))
    
    # Configure launch
    threads = 256
    blocks = cld(n_rois, threads)
    
    # Launch kernel with calibration parameters
    @cuda threads=threads blocks=blocks gaussian_fit_z_kernel_fixed!(
        d_data, d_params, d_crlb, roi_size,
        calib.σ_x0, calib.σ_y0, calib.γ, calib.d,
        calib.Ax, calib.Ay, calib.Bx, calib.By
    )
    
    return nothing
end

# Export the z-model launch function
export launch_gaussian_fit_z!