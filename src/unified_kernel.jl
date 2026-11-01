"""
Unified GPU/CPU kernel using StaticArrays and GPU-compatible operations
"""

# GPU-compatible Cholesky decomposition for symmetric positive definite matrices
# This is more robust than LU for Fisher Information matrices
@inline function static_cholesky_decomposition!(A::MMatrix{N, N, T}) where {N, T}
    # Cholesky: A = L * L^T where L is lower triangular
    @inbounds for j in 1:N
        for i in j:N
            sum_val = A[i, j]
            for k in 1:(j - 1)
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
@inline function static_cholesky_inverse!(
        A_inv::MMatrix{N, N, T}, L::MMatrix{N, N, T}
    ) where {N, T}
    # First invert L (lower triangular)
    L_inv = MMatrix{N, N, T}(undef)
    @inbounds for j in 1:N
        L_inv[j, j] = one(T) / L[j, j]
        for i in (j + 1):N
            sum_val = zero(T)
            for k in j:(i - 1)
                sum_val += L[i, k] * L_inv[k, j]
            end
            L_inv[i, j] = -sum_val / L[i, i]
        end
    end

    # A_inv = L_inv^T * L_inv
    @inbounds for i in 1:N
        for j in i:N  # Symmetric, only compute upper triangle
            sum_val = zero(T)
            for k in j:N  # Sum from max(i,j) to N
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
@inline function static_lu_decomposition!(A::MMatrix{N, N, T}) where {N, T}
    # Use relative tolerance based on matrix scale (like SMITE)
    tol = T(1.0e-10) * maximum(abs, A)

    @inbounds for k in 1:N
        # Check for near-zero pivot
        if abs(A[k, k]) < tol
            return false
        end

        for i in (k + 1):N
            A[i, k] /= A[k, k]
            for j in (k + 1):N
                A[i, j] -= A[i, k] * A[k, j]
            end
        end
    end
    return true
end

# GPU-compatible back substitution for solving Ax = b with LU-decomposed A
@inline function static_back_substitution!(
        x::MVector{N, T}, A::MMatrix{N, N, T}, b::MVector{N, T}
    ) where {N, T}
    # Forward substitution for Ly = b
    @inbounds for i in 1:N
        x[i] = b[i]
        for j in 1:(i - 1)
            x[i] -= A[i, j] * x[j]
        end
    end

    # Backward substitution for Ux = y (overwrites x with solution)
    return @inbounds for i in N:-1:1
        for j in (i + 1):N
            x[i] -= A[i, j] * x[j]
        end
        x[i] /= A[i, i]
    end
end

# GPU-compatible matrix inverse using LU decomposition
@inline function static_matrix_inverse!(
        A_inv::MMatrix{N, N, T}, A::MMatrix{N, N, T}
    ) where {N, T}
    # Make a copy for LU decomposition (don't modify original)
    A_lu = MMatrix{N, N, T}(A)

    # Perform LU decomposition
    if !static_lu_decomposition!(A_lu)
        # Singular matrix - set to infinity
        @inbounds for i in 1:N, j in 1:N
            A_inv[i, j] = i == j ? T(Inf) : zero(T)
        end
        return false
    end

    # Initialize A_inv as identity and solve for each column
    e = MVector{N, T}(undef)
    x = MVector{N, T}(undef)

    @inbounds for col in 1:N
        # Set up unit vector
        for i in 1:N
            e[i] = (i == col) ? one(T) : zero(T)
        end

        # Solve for this column
        static_back_substitution!(x, A_lu, e)

        # Store result
        for i in 1:N
            A_inv[i, col] = x[i]
        end
    end

    return true
end

# GPU-compatible determinant using LU decomposition
@inline function static_det(A::MMatrix{N, N, T}) where {N, T}
    A_lu = MMatrix{N, N, T}(A)

    # Perform LU decomposition
    if !static_lu_decomposition!(A_lu)
        return zero(T)
    end

    # Determinant is product of diagonal elements
    det_val = one(T)
    @inbounds for i in 1:N
        det_val *= A_lu[i, i]
    end

    return det_val
end

# Simple GPU-compatible parameter initialization - generic version
@inline function simple_initialize_common(roi, box_size::Int, ::Type{T}) where {T}
    # Calculate background (use edges for robustness)
    edge_sum = zero(T)
    edge_count = 0
    @inbounds for j in 1:box_size, i in 1:box_size
        if i == 1 || i == box_size || j == 1 || j == box_size
            edge_sum += roi[i, j]
            edge_count += 1
        end
    end
    bg = edge_sum / edge_count

    # Calculate center of mass
    total_signal = zero(T)
    x_weighted = zero(T)
    y_weighted = zero(T)

    @inbounds for j in 1:box_size, i in 1:box_size
        signal = max(zero(T), roi[i, j] - bg)
        total_signal += signal
        x_weighted += signal * T(j)
        y_weighted += signal * T(i)
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
@inline function simple_initialize(roi, box_size::Int, ::GaussianXYNB{T}) where {T}
    (x, y, photons, bg) = simple_initialize_common(roi, box_size, T)
    return MVector{4, T}(x, y, photons, bg)
end

@inline function simple_initialize(roi, box_size::Int, ::GaussianXYNBS{T}) where {T}
    (x, y, photons, bg) = simple_initialize_common(roi, box_size, T)
    return MVector{5, T}(x, y, photons, bg, T(1.3))
end

@inline function simple_initialize(roi, box_size::Int, ::GaussianXYNBSXSY{T}) where {T}
    (x, y, photons, bg) = simple_initialize_common(roi, box_size, T)
    return MVector{6, T}(x, y, photons, bg, T(1.3), T(1.3))
end

@inline function simple_initialize(roi, box_size::Int, psf::AstigmaticXYZNB{T}) where {T}
    (x, y, photons, bg) = simple_initialize_common(roi, box_size, T)

    # Estimate z from PSF width asymmetry
    # Compute second moments to estimate sigma_x and sigma_y
    total_signal = zero(T)
    Mxx = zero(T)
    Myy = zero(T)

    @inbounds for j in 1:box_size, i in 1:box_size
        signal = max(zero(T), roi[i, j] - bg)
        total_signal += signal
        dx = T(j) - x
        dy = T(i) - y
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

    return MVector{5, T}(x, y, z_init, photons, bg)
end

# Zero out a static array (more efficient than fill!)
@inline function zero_array!(A::MVector{N, T}) where {N, T}
    return @inbounds for i in 1:N
        A[i] = zero(T)
    end
end

@inline function zero_array!(A::MMatrix{N, M, T}) where {N, M, T}
    return @inbounds for j in 1:M, i in 1:N
        A[i, j] = zero(T)
    end
end

# Likelihood dispatch helpers - compile-time Val{Bool} branching
# Val{false} → IdealCamera (Poisson only)
@inline function _dispatch_likelihood(
        ::Val{false}, data::T, model::T, variance_map, corner_x, corner_y, i, j
    ) where {T}
    return compute_likelihood_terms(data, model, IdealCameraInternal())
end

# Val{true} → SCMOSCamera (Poisson + variance)
@inline function _dispatch_likelihood(
        ::Val{true}, data::T, model::T, variance_map, corner_x, corner_y, i, j
    ) where {T}
    # Convert ROI-local (i,j) to camera coordinates for variance lookup
    cam_i = corner_y + i - 1
    cam_j = corner_x + j - 1
    return compute_likelihood_terms(data, model, variance_map, cam_i, cam_j)
end

# Objective the Levenberg-Marquardt loop climbs, one pixel, either camera. Ideal: the Poisson
# log-likelihood without constants. sCMOS: the Poisson-shifted likelihood
# (d + v) log(mu + v) - (mu + v), v the read-noise variance at the pixel's camera position;
# its gradient (d - mu) / (mu + v) and Fisher 1 / (mu + v) are what the update uses.
@inline function _fit_objective(
        ::Val{false}, data::T, model::T, variance_map, corner_x, corner_y, i, j
    ) where {T}
    return data * log(model) - model
end

@inline function _fit_objective(
        ::Val{true}, data::T, model::T, variance_map, corner_x, corner_y, i, j
    ) where {T}
    v = variance_map[corner_y + i - 1, corner_x + j - 1]
    return (data + v) * log(model + v) - (model + v)
end

# Variance of the fit objective's gradient and Fisher terms: model (Ideal), model + v (sCMOS)
@inline function _fit_variance(
        ::Val{false}, model::T, variance_map, corner_x, corner_y, i, j
    ) where {T}
    return model
end

@inline function _fit_variance(
        ::Val{true}, model::T, variance_map, corner_x, corner_y, i, j
    ) where {T}
    return model + variance_map[corner_y + i - 1, corner_x + j - 1]
end

# Compile-time trait: a model fitted by Levenberg-Marquardt on the full Fisher matrix
# (_lm_fit) instead of the diagonal Newton loop. The Gaussian models keep the diagonal loop.
_lm_steps(::PSFModel) = false

# Solve L L' x = b for the Cholesky factor L (lower triangle) of
# static_cholesky_decomposition!
@inline function static_cholesky_solve!(
        x::MVector{N, T}, L::MMatrix{N, N, T}, b::MVector{N, T}
    ) where {N, T}
    @inbounds for i in 1:N
        s = b[i]
        for k in 1:(i - 1)
            s -= L[i, k] * x[k]
        end
        x[i] = s / L[i, i]
    end
    return @inbounds for i in N:-1:1
        s = x[i]
        for k in (i + 1):N
            s -= L[k, i] * x[k]
        end
        x[i] = s / L[i, i]
    end
end

# Levenberg-Marquardt fit of one ROI on the full Fisher matrix, from the start θ0. Each
# iteration makes one pixel pass at the trial point: the objective, the gradient
# g_k = sum du_k (d - mu) / var and F_kl = sum du_k du_l / var. A trial that does not lower
# the objective is accepted (g and F stored, lambda / 10); otherwise lambda * 10. The next
# trial is theta_acc + delta with (F_acc + lambda diag(F_acc)) delta = g_acc, step-clamped
# and bounded. Returns the last accepted point. Registers only: no allocation, no throw.
@inline function _lm_fit(
        roi, box_size::Int, psf_model::PSFModel{N, T}, camera_model, variance_map,
        corner_x, corner_y, constraints::ParameterConstraints{N}, iterations::Int, θ0
    ) where {N, T}
    θ = MVector{N, T}(undef)
    @inbounds for k in 1:N
        θ[k] = clamp(θ0[k], constraints.lower[k], constraints.upper[k])
    end
    θ_acc = MVector{N, T}(θ)
    g_acc = MVector{N, T}(undef)
    F_acc = MMatrix{N, N, T}(undef)
    zero_array!(g_acc)
    zero_array!(F_acc)
    obj_acc = zero(T)
    λ = T(1.0e-3)
    g = MVector{N, T}(undef)
    F = MMatrix{N, N, T}(undef)
    A = MMatrix{N, N, T}(undef)
    Δ = MVector{N, T}(undef)

    for iter in 1:iterations
        zero_array!(g)
        zero_array!(F)
        obj = zero(T)
        @inbounds for j in 1:box_size, i in 1:box_size
            model, dudt, _ = compute_pixel_derivatives(i, j, SVector{N, T}(θ), psf_model)
            d = roi[i, j]
            var = _fit_variance(camera_model, model, variance_map, corner_x, corner_y, i, j)
            obj += _fit_objective(
                camera_model, d, model, variance_map, corner_x, corner_y, i, j
            )
            r = (d - model) / var
            w = one(T) / var
            for k in 1:N
                g[k] += dudt[k] * r
                for l in 1:N
                    F[k, l] += dudt[k] * dudt[l] * w
                end
            end
        end

        if iter == 1 || obj >= obj_acc
            @inbounds for k in 1:N
                θ_acc[k] = θ[k]
                g_acc[k] = g[k]
                for l in 1:N
                    F_acc[k, l] = F[k, l]
                end
            end
            obj_acc = obj
            if iter > 1
                λ = max(λ / 10, T(1.0e-7))
            end
        else
            λ = min(λ * 10, T(1.0e7))
        end

        @inbounds for k in 1:N, l in 1:N
            A[k, l] = F_acc[k, l] + (k == l ? λ * F_acc[k, k] : zero(T))
        end
        if static_cholesky_decomposition!(A)
            static_cholesky_solve!(Δ, A, g_acc)
        else
            zero_array!(Δ)
        end
        @inbounds for k in 1:N
            step = clamp(Δ[k], -constraints.max_step[k], constraints.max_step[k])
            θ[k] = clamp(θ_acc[k] + step, constraints.lower[k], constraints.upper[k])
        end
    end
    return θ_acc
end

# Unified kernel that works on both CPU and GPU
@kernel function unified_gaussian_mle_kernel!(
        results::AbstractArray{T, 2},
        uncertainties::AbstractArray{T, 2},
        # Off-diagonal: [σ_xy, σ_xz, σ_yz] = H_inv[1,2], H_inv[1,3], H_inv[2,3]
        covariances::AbstractArray{T, 2},
        log_likelihoods::AbstractArray{T, 1},
        @Const(data::AbstractArray{T, 3}),
        @Const(psf_model::PSFModel{N, T}),
        @Const(camera_model),  # Val{false} for Ideal, Val{true} for sCMOS
        @Const(variance_map),  # 2D variance map (camera-sized for sCMOS)
        @Const(x_corners::AbstractVector{Int32}),  # X corners for variance indexing
        @Const(y_corners::AbstractVector{Int32}),  # Y corners for variance indexing
        @Const(constraints::ParameterConstraints{N}),
        iterations::Int
    ) where {T, N}
    idx = @index(Global)

    # Get the data for this fit
    box_size = size(data, 1)
    @inbounds roi = @view data[:, :, idx]

    # Get corner for variance map indexing (sCMOS only)
    corner_x = x_corners[idx]
    corner_y = y_corners[idx]

    # Stack-allocated working arrays (known size at compile time)
    θ = simple_initialize(roi, box_size, psf_model)
    ∇L = MVector{N, T}(undef)
    H_diag = MVector{N, T}(undef)  # Only diagonal elements for Newton-Raphson
    newton_iterations = iterations
    if _lm_steps(psf_model)
        # Levenberg-Marquardt on the full Fisher matrix; the Newton loop below then runs 0 times
        θ = _lm_fit(
            roi, box_size, psf_model, camera_model, variance_map, corner_x, corner_y,
            constraints, iterations, θ
        )
        newton_iterations = 0
    end

    # Newton-Raphson iterations with scalar updates
    for iter in 1:newton_iterations
        # Zero out gradient and diagonal Hessian
        zero_array!(∇L)
        zero_array!(H_diag)

        # Compute derivatives over all pixels
        @inbounds for j in 1:box_size, i in 1:box_size
            # Model and derivatives at this pixel
            θ_static = SVector{N, T}(θ)
            model, dudt, d2udt2_diag = compute_pixel_derivatives(i, j, θ_static, psf_model)

            # Likelihood terms based on camera model (Val dispatch for compile-time
            # optimization)
            data_ij = roi[i, j]
            cf, df = _dispatch_likelihood(
                camera_model, data_ij, model, variance_map, corner_x, corner_y, i, j
            )

            # Accumulate gradient and diagonal Hessian
            for k in 1:N
                ∇L[k] += dudt[k] * cf
                H_diag[k] += d2udt2_diag[k] * cf - dudt[k] * dudt[k] * df
            end
        end

        # Scalar Newton-Raphson updates with constraints
        Δθ = MVector{N, T}(undef)
        @inbounds for k in 1:N
            if abs(H_diag[k]) > T(1.0e-10)
                Δθ[k] = ∇L[k] / H_diag[k]
            else
                Δθ[k] = zero(T)
            end
        end

        # Apply constraints
        θ_new = apply_constraints!(SVector{N, T}(θ), SVector{N, T}(Δθ), constraints)
        @inbounds for i in 1:N
            θ[i] = θ_new[i]
        end
    end

    # Compute final log-likelihood and full Fisher Information Matrix for CRLB
    log_likelihood = zero(T)
    H = MMatrix{N, N, T}(undef)  # Full Fisher Information matrix
    H_inv = MMatrix{N, N, T}(undef)  # For inverse
    zero_array!(H)

    @inbounds for j in 1:box_size, i in 1:box_size
        θ_static = SVector{N, T}(θ)
        model, dudt, _ = compute_pixel_derivatives(i, j, θ_static, psf_model)
        data_ij = roi[i, j]

        # Log-likelihood contribution (dispatch based on camera_model Val)
        # camera_model is Val{false} for IdealCamera, Val{true} for sCMOS
        log_likelihood += if camera_model isa Val{false}
            compute_log_likelihood(data_ij, model, IdealCameraInternal())
        else
            # sCMOS: convert ROI-local (i,j) to camera coordinates
            cam_i = corner_y + i - 1
            cam_j = corner_x + j - 1
            compute_log_likelihood(data_ij, model, variance_map, cam_i, cam_j)
        end

        # Fisher Information Matrix (for CRLB) - need full matrix here
        # For IdealCamera: variance = model (Poisson only)
        # For sCMOS: variance = model + readout_variance
        if model > zero(T)
            # Compute variance - dispatch based on camera_model Val
            variance = if camera_model isa Val{false}
                model  # Poisson only (IdealCamera)
            else
                # sCMOS camera - convert ROI-local (i,j) to camera coordinates
                cam_i = corner_y + i - 1
                cam_j = corner_x + j - 1
                model + variance_map[cam_i, cam_j]
            end

            for k in 1:N, l in k:N
                F_kl = dudt[k] * dudt[l] / variance
                H[k, l] += F_kl
                if k != l
                    H[l, k] += F_kl  # Symmetric
                end
            end
        end
    end

    # Invert Fisher matrix for uncertainties (CRLB)
    # Fisher matrices should be symmetric positive definite
    # Add minimal regularization to diagonal for numerical stability
    # Use very small value to not bias uncertainty estimates
    reg = T(1.0e-10) * maximum(abs, H)
    @inbounds for k in 1:N
        H[k, k] += reg
    end

    # Make a copy for Cholesky decomposition
    H_chol = MMatrix{N, N, T}(H)

    # Try Cholesky decomposition (for symmetric positive definite matrices)
    if static_cholesky_decomposition!(H_chol) && static_cholesky_inverse!(H_inv, H_chol)
        @inbounds for k in 1:N
            results[k, idx] = θ[k]
            uncertainties[k, idx] = sqrt(max(zero(T), H_inv[k, k]))
        end
        # Extract spatial covariances from Fisher matrix inverse
        # covariances[1,idx] = σ_xy (all models)
        # covariances[2,idx] = σ_xz (3D only, zero otherwise)
        # covariances[3,idx] = σ_yz (3D only, zero otherwise)
        @inbounds covariances[1, idx] = H_inv[1, 2]
        if _has_axial_position(typeof(psf_model))
            @inbounds covariances[2, idx] = H_inv[1, 3]
            @inbounds covariances[3, idx] = H_inv[2, 3]
        else
            @inbounds covariances[2, idx] = zero(T)
            @inbounds covariances[3, idx] = zero(T)
        end
    else
        # Fallback: use LU decomposition
        if static_matrix_inverse!(H_inv, H)
            @inbounds for k in 1:N
                results[k, idx] = θ[k]
                uncertainties[k, idx] = sqrt(max(zero(T), H_inv[k, k]))
            end
            @inbounds covariances[1, idx] = H_inv[1, 2]
            if _has_axial_position(typeof(psf_model))
                @inbounds covariances[2, idx] = H_inv[1, 3]
                @inbounds covariances[3, idx] = H_inv[2, 3]
            else
                @inbounds covariances[2, idx] = zero(T)
                @inbounds covariances[3, idx] = zero(T)
            end
        else
            # Singular matrix - set zero covariances
            @inbounds for k in 1:N
                results[k, idx] = θ[k]
                uncertainties[k, idx] = T(Inf)
            end
            @inbounds covariances[1, idx] = zero(T)
            @inbounds covariances[2, idx] = zero(T)
            @inbounds covariances[3, idx] = zero(T)
        end
    end

    @inbounds log_likelihoods[idx] = log_likelihood
end
