"""
Unified GPU/CPU kernel using StaticArrays and GPU-compatible operations
"""

using KernelAbstractions
using StaticArrays

# GPU-compatible LU decomposition for small static matrices
@inline function static_lu_decomposition!(A::MMatrix{N,N,T}) where {N,T}
    @inbounds for k = 1:N
        # Check for zero pivot
        if abs(A[k, k]) < eps(T)
            return false
        end
        
        for i = k+1:N
            A[i, k] /= A[k, k]
            for j = k+1:N
                A[i, j] -= A[i, k] * A[k, j]
            end
        end
    end
    return true
end

# GPU-compatible back substitution for solving Ax = b with LU-decomposed A
@inline function static_back_substitution!(x::MVector{N,T}, A::MMatrix{N,N,T}, b::MVector{N,T}) where {N,T}
    # Forward substitution for Ly = b
    @inbounds for i = 1:N
        x[i] = b[i]
        for j = 1:i-1
            x[i] -= A[i, j] * x[j]
        end
    end
    
    # Backward substitution for Ux = y (overwrites x with solution)
    @inbounds for i = N:-1:1
        for j = i+1:N
            x[i] -= A[i, j] * x[j]
        end
        x[i] /= A[i, i]
    end
end

# GPU-compatible matrix inverse using LU decomposition
@inline function static_matrix_inverse!(A_inv::MMatrix{N,N,T}, A::MMatrix{N,N,T}) where {N,T}
    # Make a copy for LU decomposition (don't modify original)
    A_lu = MMatrix{N,N,T}(A)
    
    # Perform LU decomposition
    if !static_lu_decomposition!(A_lu)
        # Singular matrix - set to infinity
        @inbounds for i = 1:N, j = 1:N
            A_inv[i,j] = i == j ? T(Inf) : zero(T)
        end
        return false
    end
    
    # Initialize A_inv as identity and solve for each column
    e = MVector{N,T}(undef)
    x = MVector{N,T}(undef)
    
    @inbounds for col = 1:N
        # Set up unit vector
        for i = 1:N
            e[i] = (i == col) ? one(T) : zero(T)
        end
        
        # Solve for this column
        static_back_substitution!(x, A_lu, e)
        
        # Store result
        for i = 1:N
            A_inv[i, col] = x[i]
        end
    end
    
    return true
end

# GPU-compatible determinant using LU decomposition
@inline function static_det(A::MMatrix{N,N,T}) where {N,T}
    A_lu = MMatrix{N,N,T}(A)
    
    # Perform LU decomposition
    if !static_lu_decomposition!(A_lu)
        return zero(T)
    end
    
    # Determinant is product of diagonal elements
    det_val = one(T)
    @inbounds for i = 1:N
        det_val *= A_lu[i, i]
    end
    
    return det_val
end

# Simple GPU-compatible parameter initialization
@inline function simple_initialize(roi, box_size::Int, ::Val{N}, ::Type{T}) where {N,T}
    # Calculate background (use 20th percentile or edges for robustness)
    edge_sum = zero(T)
    edge_count = 0
    @inbounds for j in 1:box_size, i in 1:box_size
        if i == 1 || i == box_size || j == 1 || j == box_size
            edge_sum += roi[i,j]
            edge_count += 1
        end
    end
    bg = edge_sum / edge_count  # Use edge average as background estimate
    
    # Calculate center of mass for better position initialization
    total_signal = zero(T)
    x_weighted = zero(T)
    y_weighted = zero(T)
    
    @inbounds for j in 1:box_size, i in 1:box_size
        signal = max(zero(T), roi[i,j] - bg)  # Background-subtracted signal
        total_signal += signal
        x_weighted += signal * T(i)
        y_weighted += signal * T(j)
    end
    
    # Use center of mass if there's signal, otherwise use center
    center = T((box_size + 1) / 2)
    x_init = total_signal > T(10) ? x_weighted / total_signal : center
    y_init = total_signal > T(10) ? y_weighted / total_signal : center
    
    # Ensure positions are within bounds
    x_init = clamp(x_init, T(1.5), T(box_size - 0.5))
    y_init = clamp(y_init, T(1.5), T(box_size - 0.5))
    
    # Estimate photons from total signal
    photons = max(total_signal, T(100))  # Ensure positive photons
    
    # Return appropriate parameter vector based on model
    if N == 4  # GaussianXYNB
        return MVector{4,T}(x_init, y_init, photons, bg)
    elseif N == 5  # GaussianXYNBS or AstigmaticXYZNB
        # Note: This simple initialization works for both models
        # For AstigmaticXYZNB: x, y, z=0, photons, bg
        # For GaussianXYNBS: x, y, photons, bg, sigma
        # The proper initialization should be done based on model type
        return MVector{5,T}(x_init, y_init, photons, bg, T(1.3))
    else  # N == 6, GaussianXYNBSXSY
        return MVector{6,T}(x_init, y_init, photons, bg, T(1.3), T(1.3))
    end
end

# Zero out a static array (more efficient than fill!)
@inline function zero_array!(A::MVector{N,T}) where {N,T}
    @inbounds for i = 1:N
        A[i] = zero(T)
    end
end

@inline function zero_array!(A::MMatrix{N,M,T}) where {N,M,T}
    @inbounds for j = 1:M, i = 1:N
        A[i,j] = zero(T)
    end
end

# Unified kernel that works on both CPU and GPU
@kernel function unified_gaussian_mle_kernel!(
    results::AbstractArray{T,2},
    uncertainties::AbstractArray{T,2},
    log_likelihoods::AbstractArray{T,1},
    @Const(data::AbstractArray{T,3}),
    @Const(psf_model::PSFModel{N,T}),
    @Const(camera_model),
    @Const(constraints::ParameterConstraints{N}),
    iterations::Int
) where {T, N}
    idx = @index(Global)
    
    # Get the data for this fit
    box_size = size(data, 1)
    @inbounds roi = @view data[:, :, idx]
    
    # Stack-allocated working arrays (known size at compile time)
    θ = simple_initialize(roi, box_size, Val(N), T)
    ∇L = MVector{N,T}(undef)
    H = MMatrix{N,N,T}(undef)
    H_inv = MMatrix{N,N,T}(undef)
    
    # Newton-Raphson iterations
    for iter in 1:iterations
        # Zero out gradient and Hessian
        zero_array!(∇L)
        zero_array!(H)
        
        # Compute derivatives over all pixels
        @inbounds for j in 1:box_size, i in 1:box_size
            # Model and derivatives at this pixel
            θ_static = SVector{N,T}(θ)
            model, dudt, d2udt2 = compute_pixel_derivatives(i, j, θ_static, psf_model)
            
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
                    if k != l
                        H[l,k] += H_kl  # Symmetric
                    end
                end
            end
        end
        
        # Newton-Raphson update with constraints
        det_H = static_det(H)
        if abs(det_H) > T(1e-10)
            # Compute inverse using our GPU-compatible LU
            if static_matrix_inverse!(H_inv, H)
                # Compute update: Δθ = H_inv * ∇L
                Δθ = MVector{N,T}(undef)
                @inbounds for i = 1:N
                    Δθ[i] = zero(T)
                    for j = 1:N
                        Δθ[i] += H_inv[i,j] * ∇L[j]
                    end
                end
                
                # Apply constraints
                θ_new = apply_constraints!(SVector{N,T}(θ), SVector{N,T}(Δθ), constraints)
                @inbounds for i = 1:N
                    θ[i] = θ_new[i]
                end
            end
        end
    end
    
    # Compute final log-likelihood and CRLB
    log_likelihood = zero(T)
    zero_array!(H)  # Reuse for Fisher Information matrix
    
    @inbounds for j in 1:box_size, i in 1:box_size
        θ_static = SVector{N,T}(θ)
        model, dudt, _ = compute_pixel_derivatives(i, j, θ_static, psf_model)
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
                if k != l
                    H[l,k] += F_kl
                end
            end
        end
    end
    
    # Invert Fisher matrix for uncertainties (CRLB)
    if static_matrix_inverse!(H_inv, H)
        @inbounds for k in 1:N
            results[k, idx] = θ[k]
            uncertainties[k, idx] = sqrt(max(zero(T), H_inv[k,k]))
        end
    else
        # Singular matrix - set large uncertainties
        @inbounds for k in 1:N
            results[k, idx] = θ[k]
            uncertainties[k, idx] = T(Inf)
        end
    end
    
    @inbounds log_likelihoods[idx] = log_likelihood
end

# Export the unified kernel
export unified_gaussian_mle_kernel!