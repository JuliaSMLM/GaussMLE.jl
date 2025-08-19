"""
Main fitting kernel using KernelAbstractions
"""

using KernelAbstractions
using StaticArrays
using LinearAlgebra

# Analytical derivatives are included in main module

@kernel function gaussian_mle_kernel!(
    results::AbstractArray{T,2},
    uncertainties::AbstractArray{T,2},
    log_likelihoods::AbstractArray{T,1},
    @Const(data::AbstractArray{T,3}),
    @Const(psf_model),
    @Const(camera_model),
    @Const(constraints),
    iterations::Int
) where T
    idx = @index(Global)
    
    # Get the data for this fit
    box_size = size(data, 1)
    roi = @inbounds view(data, :, :, idx)
    
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
            
            # Likelihood terms based on camera model
            data_ij = roi[i, j]
            cf, df = if camera_model isa IdealCamera
                compute_likelihood_terms(data_ij, model, camera_model)
            else
                compute_likelihood_terms(data_ij, model, camera_model, i, j)
            end
            
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
        # Use simple inverse for small matrices
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
        
        # Log-likelihood contribution
        if camera_model isa IdealCamera
            log_likelihood += compute_log_likelihood(data_ij, model, camera_model)
        else
            log_likelihood += compute_log_likelihood(data_ij, model, camera_model, i, j)
        end
        
        # Fisher Information Matrix (for CRLB)
        if model > zero(T)
            for k in 1:N, l in k:N
                F_kl = dudt[k] * dudt[l] / model
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
export gaussian_mle_kernel!