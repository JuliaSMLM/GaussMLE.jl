"""
CPU-specific fitting kernel using standard arrays
"""

using LinearAlgebra

function cpu_fit_single_roi!(
    params_out::AbstractVector{T},
    uncertainty_out::AbstractVector{T},
    log_likelihood_out::Ref{T},
    roi::AbstractMatrix{T},
    psf_model,
    camera_model,
    constraints,
    iterations::Int
) where T
    
    # Get box size
    box_size = size(roi, 1)
    
    # Initialize parameters
    θ = Vector(initialize_parameters(roi, psf_model))
    N = length(psf_model)
    
    # Allocate working arrays
    ∇L = zeros(T, N)  # Gradient
    H = zeros(T, N, N) # Hessian
    
    # Newton-Raphson iterations
    for iter in 1:iterations
        fill!(∇L, zero(T))
        fill!(H, zero(T))
        
        # Compute derivatives over all pixels
        for j in 1:box_size, i in 1:box_size
            # Model and derivatives at this pixel
            model, dudt, d2udt2 = compute_pixel_derivatives(i, j, Params{N}(θ), psf_model)
            
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
        if det(H) > 1e-10
            Δθ = H \ ∇L
            θ_new = apply_constraints!(Params{N}(θ), Params{N}(Δθ), constraints)
            θ = Vector(θ_new)
        end
    end
    
    # Compute final log-likelihood and CRLB
    log_likelihood = zero(T)
    fill!(H, zero(T))  # Reuse for Fisher matrix
    
    for j in 1:box_size, i in 1:box_size
        model, dudt, _ = compute_pixel_derivatives(i, j, Params{N}(θ), psf_model)
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
    if det(H) > 1e-10
        H_inv = inv(H)
        for k in 1:N
            uncertainty_out[k] = sqrt(max(zero(T), H_inv[k,k]))
        end
    else
        fill!(uncertainty_out, T(Inf))
    end
    
    # Store results
    params_out .= θ
    log_likelihood_out[] = log_likelihood
end

# CPU batch fitting function
function cpu_fit_batch!(
    results::AbstractArray{T,2},
    uncertainties::AbstractArray{T,2},
    log_likelihoods::AbstractArray{T,1},
    data::AbstractArray{T,3},
    psf_model,
    camera_model,
    constraints,
    iterations::Int
) where T
    
    n_fits = size(data, 3)
    n_params = length(psf_model)
    
    # Process each ROI
    Threads.@threads for idx in 1:n_fits
        roi = @view data[:, :, idx]
        params_out = @view results[:, idx]
        uncertainty_out = @view uncertainties[:, idx]
        log_likelihood_out = Ref{T}(zero(T))
        
        cpu_fit_single_roi!(
            params_out,
            uncertainty_out,
            log_likelihood_out,
            roi,
            psf_model,
            camera_model,
            constraints,
            iterations
        )
        
        log_likelihoods[idx] = log_likelihood_out[]
    end
end

export cpu_fit_batch!, cpu_fit_single_roi!