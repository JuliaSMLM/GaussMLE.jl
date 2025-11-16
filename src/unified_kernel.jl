"""
Unified GPU/CPU kernel using StaticArrays and GPU-compatible operations
"""

using KernelAbstractions
using StaticArrays

# GPU-compatible Cholesky decomposition for symmetric positive definite matrices
# This is more robust than LU for Fisher Information matrices
@inline function static_cholesky_decomposition!(A::MMatrix{N,N,T}) where {N,T}
    # Cholesky: A = L * L^T where L is lower triangular
    @inbounds for j = 1:N
        for i = j:N
            sum_val = A[i, j]
            for k = 1:j-1
                sum_val -= A[i, k] * A[j, k]
            end

            if i == j
                # Diagonal element
                if sum_val <= zero(T)
                    return false  # Not positive definite
                end
                A[i, i] = sqrt(sum_val)
            else
                # Off-diagonal element
                A[i, j] = sum_val / A[j, j]
            end
        end
    end
    return true
end

# Invert a Cholesky-decomposed matrix (A = L * L^T)
@inline function static_cholesky_inverse!(A_inv::MMatrix{N,N,T}, L::MMatrix{N,N,T}) where {N,T}
    # First invert L (lower triangular)
    L_inv = MMatrix{N,N,T}(undef)
    @inbounds for j = 1:N
        L_inv[j, j] = one(T) / L[j, j]
        for i = j+1:N
            sum_val = zero(T)
            for k = j:i-1
                sum_val += L[i, k] * L_inv[k, j]
            end
            L_inv[i, j] = -sum_val / L[i, i]
        end
    end

    # A_inv = L_inv^T * L_inv
    @inbounds for i = 1:N
        for j = i:N  # Symmetric, only compute upper triangle
            sum_val = zero(T)
            for k = j:N  # Sum from max(i,j) to N
                sum_val += L_inv[k, i] * L_inv[k, j]
            end
            A_inv[i, j] = sum_val
            if i != j
                A_inv[j, i] = sum_val  # Fill lower triangle by symmetry
            end
        end
    end
    return true
end

# GPU-compatible LU decomposition for small static matrices (no pivoting)
# This matches SMITE's approach with a relative tolerance
@inline function static_lu_decomposition!(A::MMatrix{N,N,T}) where {N,T}
    # Use relative tolerance based on matrix scale (like SMITE)
    tol = T(1e-10) * maximum(abs, A)

    @inbounds for k = 1:N
        # Check for near-zero pivot
        if abs(A[k, k]) < tol
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

# Simple GPU-compatible parameter initialization - generic version
@inline function simple_initialize_common(roi, box_size::Int, ::Type{T}) where T
    # Calculate background (use edges for robustness)
    edge_sum = zero(T)
    edge_count = 0
    @inbounds for j in 1:box_size, i in 1:box_size
        if i == 1 || i == box_size || j == 1 || j == box_size
            edge_sum += roi[i,j]
            edge_count += 1
        end
    end
    bg = edge_sum / edge_count

    # Calculate center of mass
    total_signal = zero(T)
    x_weighted = zero(T)
    y_weighted = zero(T)

    @inbounds for j in 1:box_size, i in 1:box_size
        signal = max(zero(T), roi[i,j] - bg)
        total_signal += signal
        x_weighted += signal * T(i)
        y_weighted += signal * T(j)
    end

    center = T((box_size + 1) / 2)
    x_init = total_signal > T(10) ? x_weighted / total_signal : center
    y_init = total_signal > T(10) ? y_weighted / total_signal : center

    x_init = clamp(x_init, T(1.5), T(box_size - 0.5))
    y_init = clamp(y_init, T(1.5), T(box_size - 0.5))

    photons = max(total_signal, T(100))

    return (x_init, y_init, photons, bg)
end

# Model-specific initialization
@inline function simple_initialize(roi, box_size::Int, ::GaussianXYNB{T}) where T
    (x, y, photons, bg) = simple_initialize_common(roi, box_size, T)
    return MVector{4,T}(x, y, photons, bg)
end

@inline function simple_initialize(roi, box_size::Int, ::GaussianXYNBS{T}) where T
    (x, y, photons, bg) = simple_initialize_common(roi, box_size, T)
    return MVector{5,T}(x, y, photons, bg, T(1.3))
end

@inline function simple_initialize(roi, box_size::Int, ::GaussianXYNBSXSY{T}) where T
    (x, y, photons, bg) = simple_initialize_common(roi, box_size, T)
    return MVector{6,T}(x, y, photons, bg, T(1.3), T(1.3))
end

@inline function simple_initialize(roi, box_size::Int, psf::AstigmaticXYZNB{T}) where T
    (x, y, photons, bg) = simple_initialize_common(roi, box_size, T)

    # Estimate z from PSF width asymmetry
    # Compute second moments to estimate sigma_x and sigma_y
    total_signal = zero(T)
    Mxx = zero(T)
    Myy = zero(T)

    @inbounds for j in 1:box_size, i in 1:box_size
        signal = max(zero(T), roi[i,j] - bg)
        total_signal += signal
        dx = T(i) - x
        dy = T(j) - y
        Mxx += signal * dx * dx
        Myy += signal * dy * dy
    end

    if total_signal > T(10)
        sigma_x_est = sqrt(max(T(0.5), Mxx / total_signal))
        sigma_y_est = sqrt(max(T(0.5), Myy / total_signal))

        # Estimate z from width asymmetry
        # For astigmatic PSF: σ(z) = σ₀ * sqrt(α(z))
        # α(z±γ) = 1 + ((z±γ)/d)² + A((z±γ)/d)³ + B((z±γ)/d)⁴
        # αx = (σx/σx₀)², αy = (σy/σy₀)²
        # Δα = αx - αy ≈ -4zγ/d² → z ≈ -Δα*d²/(4γ)

        # Compute α values from measured widths
        αx_est = (sigma_x_est / psf.σx₀)^2
        αy_est = (sigma_y_est / psf.σy₀)^2
        Δα = αx_est - αy_est

        # Estimate z from Δα (negative because of sign convention)
        # When z < 0: αx > αy → Δα > 0 → z = -Δα*d²/(4γ) < 0 ✓
        # When z > 0: αx < αy → Δα < 0 → z = -Δα*d²/(4γ) > 0 ✓
        z_init = -Δα * psf.d^2 / (T(4) * psf.γ)

        # Clamp to reasonable range (±600nm for typical astigmatic systems)
        z_init = clamp(z_init, T(-600), T(600))
    else
        z_init = T(0)
    end

    return MVector{5,T}(x, y, z_init, photons, bg)
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
    @Const(variance_map),  # Separate variance_map for sCMOS (nothing for IdealCamera)
    @Const(constraints::ParameterConstraints{N}),
    iterations::Int
) where {T, N}
    idx = @index(Global)
    
    # Get the data for this fit
    box_size = size(data, 1)
    @inbounds roi = @view data[:, :, idx]
    
    # Stack-allocated working arrays (known size at compile time)
    θ = simple_initialize(roi, box_size, psf_model)
    ∇L = MVector{N,T}(undef)
    H_diag = MVector{N,T}(undef)  # Only diagonal elements for Newton-Raphson
    
    # Newton-Raphson iterations with scalar updates
    for iter in 1:iterations
        # Zero out gradient and diagonal Hessian
        zero_array!(∇L)
        zero_array!(H_diag)
        
        # Compute derivatives over all pixels
        @inbounds for j in 1:box_size, i in 1:box_size
            # Model and derivatives at this pixel
            θ_static = SVector{N,T}(θ)
            model, dudt, d2udt2_diag = compute_pixel_derivatives(i, j, θ_static, psf_model)
            
            # Likelihood terms based on camera model
            data_ij = roi[i, j]
            cf, df = if camera_model isa IdealCamera
                compute_likelihood_terms(data_ij, model, camera_model)
            else
                # sCMOS: use separate variance_map argument
                compute_likelihood_terms(data_ij, model, variance_map, i, j)
            end
            
            # Accumulate gradient and diagonal Hessian
            for k in 1:N
                ∇L[k] += dudt[k] * cf
                H_diag[k] += d2udt2_diag[k] * cf - dudt[k] * dudt[k] * df
            end
        end
        
        # Scalar Newton-Raphson updates with constraints
        Δθ = MVector{N,T}(undef)
        @inbounds for k in 1:N
            if abs(H_diag[k]) > T(1e-10)
                Δθ[k] = ∇L[k] / H_diag[k]
            else
                Δθ[k] = zero(T)
            end
        end
        
        # Apply constraints
        θ_new = apply_constraints!(SVector{N,T}(θ), SVector{N,T}(Δθ), constraints)
        @inbounds for i = 1:N
            θ[i] = θ_new[i]
        end
    end
    
    # Compute final log-likelihood and full Fisher Information Matrix for CRLB
    log_likelihood = zero(T)
    H = MMatrix{N,N,T}(undef)  # Full Fisher Information matrix
    H_inv = MMatrix{N,N,T}(undef)  # For inverse
    zero_array!(H)
    
    @inbounds for j in 1:box_size, i in 1:box_size
        θ_static = SVector{N,T}(θ)
        model, dudt, _ = compute_pixel_derivatives(i, j, θ_static, psf_model)
        data_ij = roi[i, j]

        # Log-likelihood contribution
        if camera_model isa IdealCamera
            log_likelihood += compute_log_likelihood(data_ij, model, camera_model)
        else
            # sCMOS: use separate variance_map argument
            log_likelihood += compute_log_likelihood(data_ij, model, variance_map, i, j)
        end

        # Fisher Information Matrix (for CRLB) - need full matrix here
        # For IdealCamera: variance = model (Poisson only)
        # For sCMOS: variance = model + readout_variance
        if model > zero(T)
            # Compute variance - check variance_map, not camera type (camera may be placeholder)
            variance = if isnothing(variance_map)
                model  # Poisson only (IdealCamera)
            else
                # sCMOS camera - use separate variance_map
                model + variance_map[i, j]
            end

            for k in 1:N, l in k:N
                F_kl = dudt[k] * dudt[l] / variance
                H[k,l] += F_kl
                if k != l
                    H[l,k] += F_kl  # Symmetric
                end
            end
        end
    end
    
    # Invert Fisher matrix for uncertainties (CRLB)
    # Fisher matrices should be symmetric positive definite
    # Add minimal regularization to diagonal for numerical stability
    # Use very small value to not bias uncertainty estimates
    reg = T(1e-10) * maximum(abs, H)
    @inbounds for k in 1:N
        H[k,k] += reg
    end

    # Make a copy for Cholesky decomposition
    H_chol = MMatrix{N,N,T}(H)

    # Try Cholesky decomposition (for symmetric positive definite matrices)
    if static_cholesky_decomposition!(H_chol) && static_cholesky_inverse!(H_inv, H_chol)
        @inbounds for k in 1:N
            results[k, idx] = θ[k]
            uncertainties[k, idx] = sqrt(max(zero(T), H_inv[k,k]))
        end
    else
        # Fallback: use LU decomposition
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
    end
    
    @inbounds log_likelihoods[idx] = log_likelihood
end

# Export the unified kernel
export unified_gaussian_mle_kernel!