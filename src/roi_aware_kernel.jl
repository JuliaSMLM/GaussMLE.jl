"""
ROI-aware kernel that handles corner offsets for sCMOS variance maps
"""

using KernelAbstractions
using StaticArrays
using LinearAlgebra

# ROI-aware kernel with variance map indexing
@kernel function roi_aware_gaussian_mle_kernel!(
    results::AbstractArray{T,2},
    uncertainties::AbstractArray{T,2},
    log_likelihoods::AbstractArray{T,1},
    @Const(data::AbstractArray{T,3}),
    @Const(roi_corners::AbstractMatrix{Int32}),  # 2×n_rois
    @Const(variance_map::AbstractMatrix{T}),
    @Const(psf_model),
    @Const(constraints),
    iterations::Int
) where T
    idx = @index(Global)
    
    # Get the data for this fit
    box_size = size(data, 1)
    roi = @inbounds view(data, :, :, idx)
    
    # Get ROI corner offsets for variance map indexing
    x_corner = @inbounds roi_corners[1, idx]
    y_corner = @inbounds roi_corners[2, idx]
    
    # Initialize parameters
    θ = initialize_parameters(roi, psf_model)
    
    # Allocate small working arrays
    N = length(psf_model)
    ∇L = zeros(T, N)  # Gradient
    H = zeros(T, N, N) # Hessian
    
    # Newton-Raphson iterations
    for iter in 1:iterations
        fill!(∇L, zero(T))
        fill!(H, zero(T))
        
        # Compute derivatives over all pixels
        for j in 1:box_size, i in 1:box_size
            # Model and derivatives at this pixel
            model, dudt, d2udt2 = compute_pixel_derivatives(i, j, θ, psf_model)
            
            # Get variance at this pixel location in camera coordinates
            # ROI pixel (i,j) corresponds to camera pixel (i+x_corner-1, j+y_corner-1)
            cam_i = i + x_corner - 1
            cam_j = j + y_corner - 1
            pixel_variance = @inbounds variance_map[cam_i, cam_j]
            
            # Likelihood terms for sCMOS camera
            data_ij = roi[i, j]
            total_var = model + pixel_variance
            cf = (data_ij - model) / total_var
            df = (data_ij + pixel_variance) / (total_var * total_var)
            
            # Accumulate gradient and Hessian
            for k in 1:N
                ∇L[k] += dudt[k] * cf
                for l in k:N
                    H_kl = d2udt2[k,l] * cf - dudt[k] * dudt[l] * df
                    H[k,l] += H_kl
                    k != l && (H[l,k] += H_kl)  # Symmetric
                end
            end
        end
        
        # Newton-Raphson update with constraints
        det_H = det(H)
        if abs(det_H) > 1e-10
            H_inv = inv(H)
            Δθ = H_inv * ∇L
            θ = apply_constraints!(SVector{N,T}(θ), SVector{N,T}(Δθ), constraints)
        end
    end
    
    # Compute final log-likelihood and CRLB
    log_likelihood = zero(T)
    fill!(H, zero(T))  # Reuse for Fisher Information matrix
    
    for j in 1:box_size, i in 1:box_size
        model, dudt, _ = compute_pixel_derivatives(i, j, θ, psf_model)
        data_ij = roi[i, j]
        
        # Camera pixel coordinates for variance lookup
        cam_i = i + x_corner - 1
        cam_j = j + y_corner - 1
        pixel_variance = @inbounds variance_map[cam_i, cam_j]
        
        # Log-likelihood contribution for sCMOS
        if model > zero(T)
            total_var = model + pixel_variance
            log_likelihood += -T(0.5) * (
                log(T(2π) * total_var) + 
                (data_ij - model)^2 / total_var
            )
        end
        
        # Fisher Information Matrix (for CRLB)
        if model > zero(T)
            total_var = model + pixel_variance
            for k in 1:N, l in k:N
                # Fisher information for sCMOS model
                F_kl = dudt[k] * dudt[l] / total_var
                H[k,l] += F_kl
                k != l && (H[l,k] += F_kl)
            end
        end
    end
    
    # Invert Fisher matrix for uncertainties (CRLB)
    det_H = det(H)
    H_inv = if abs(det_H) > 1e-10
        inv(H)
    else
        # If matrix is singular, set uncertainties to infinity
        fill(T(Inf), N, N)
    end
    
    # Store results
    @inbounds for k in 1:N
        results[k, idx] = θ[k]
        uncertainties[k, idx] = sqrt(max(zero(T), H_inv[k,k]))
    end
    @inbounds log_likelihoods[idx] = log_likelihood
end

# Export the kernel
export roi_aware_gaussian_mle_kernel!