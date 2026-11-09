"""
Unified GPU/CPU kernel using StaticArrays and GPU-compatible operations
"""

# GPU-compatible Cholesky decomposition for symmetric positive definite matrices.
# A pivot (the squared diagonal of the factor, before the square root) at or below `tol`
# fails. A NaN pivot does not fail here (`NaN <= tol` is false): the factor carries the
# NaN and the caller's checks catch it. The spline Levenberg-Marquardt step passes no
# tolerance.
# Immutable matrices avoid heap allocation caused by bounds-check error paths.
@inline function static_cholesky_decomposition(
        A::SMatrix{N, N, T}, tol::T = zero(T)
    ) where {N, T}
    # Cholesky: A = L * L^T where L is lower triangular
    @inbounds for j in 1:N
        for i in j:N
            sum_val = A[i, j]
            for k in 1:(j - 1)
                sum_val -= A[i, k] * A[j, k]
            end

            if i == j
                # Diagonal element
                if sum_val <= tol
                    return false, A  # Not positive definite (to the tolerance)
                end
                A = Base.setindex(A, sqrt(sum_val), i, i)
            else
                # Off-diagonal element
                A = Base.setindex(A, sum_val / A[j, j], i, j)
            end
        end
    end
    return true, A
end

# Invert a Cholesky-decomposed matrix (A = L * L^T)
@inline function static_cholesky_inverse(L::SMatrix{N, N, T}) where {N, T}
    # First invert L (lower triangular)
    L_inv = zero(L)
    @inbounds for j in 1:N
        L_inv = Base.setindex(L_inv, one(T) / L[j, j], j, j)
        for i in (j + 1):N
            sum_val = zero(T)
            for k in j:(i - 1)
                sum_val += L[i, k] * L_inv[k, j]
            end
            L_inv = Base.setindex(L_inv, -sum_val / L[i, i], i, j)
        end
    end

    # A_inv = L_inv^T * L_inv
    A_inv = zero(L)
    @inbounds for i in 1:N
        for j in i:N  # Symmetric, only compute upper triangle
            sum_val = zero(T)
            for k in j:N  # Sum from max(i,j) to N
                sum_val += L_inv[k, i] * L_inv[k, j]
            end
            A_inv = Base.setindex(A_inv, sum_val, i, j)
            if i != j
                A_inv = Base.setindex(A_inv, sum_val, j, i)  # Fill lower triangle
            end
        end
    end
    return true, A_inv
end

# Inverse of a Fisher matrix F for the CRLB, `(ok, C)`. `n_pixels` is the number of pixels
# summed into F. The entries span many orders of magnitude (F_xx grows like N, F_NN
# shrinks like 1/N), so F is inverted in Jacobi-scaled form: with d_k = 1 / sqrt(F_kk) and
# G = D F D, which has a unit diagonal, G is inverted by Cholesky and C = D G^-1 D. F is
# positive semidefinite by construction, so a Cholesky pivot at or below
# `n_pixels * eps(T)` means F is singular to working precision and there is no fallback:
# the accumulation over the box's pixels perturbs the entries of G by about that much, so
# a pivot below it is not distinguishable from an exactly singular matrix, whose inverse
# is rounding. Nothing is added to the diagonal, so every uncertainty is the Cramér-Rao
# bound itself. `ok = false` for a diagonal entry of F that is not positive and finite, a
# failed Cholesky, or an inverse diagonal that is not positive and finite (which also
# catches a NaN pivot).
@inline function _invert_fisher(F::SMatrix{N, N, T}, n_pixels::Integer) where {N, T}
    positive = true
    @inbounds for k in 1:N
        positive &= F[k, k] > zero(T) && isfinite(F[k, k])
    end
    positive || return false, zero(F)
    d = SVector{N, T}(ntuple(k -> one(T) / sqrt(F[k, k]), Val(N)))
    G = (F .* d) .* d'
    ok, G_chol = static_cholesky_decomposition(G, T(n_pixels) * eps(T))
    ok || return false, zero(F)
    _, G_inv = static_cholesky_inverse(G_chol)
    C = (G_inv .* d) .* d'
    valid = true
    @inbounds for k in 1:N
        valid &= C[k, k] > zero(T) && isfinite(C[k, k])
    end
    valid || return false, zero(F)
    return true, C
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
    return SVector{4, T}(x, y, photons, bg)
end

@inline function simple_initialize(roi, box_size::Int, ::GaussianXYNBS{T}) where {T}
    (x, y, photons, bg) = simple_initialize_common(roi, box_size, T)
    return SVector{5, T}(x, y, photons, bg, T(1.3))
end

@inline function simple_initialize(roi, box_size::Int, ::GaussianXYNBSXSY{T}) where {T}
    (x, y, photons, bg) = simple_initialize_common(roi, box_size, T)
    return SVector{6, T}(x, y, photons, bg, T(1.3), T(1.3))
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
        # z in microns; the fit clamps the start to the z constraints
        z_init = -Δα * psf.d^2 / (T(4) * psf.γ)
    else
        z_init = T(0)
    end

    return SVector{5, T}(x, y, z_init, photons, bg)
end

# Read-noise variance at the pixel's camera position: zero for the ideal camera (Poisson
# only), the variance map's value for sCMOS.
@inline function _readout_variance(
        ::Val{false}, model::T, variance_map, corner_x, corner_y, i, j
    ) where {T}
    return zero(T)
end

@inline function _readout_variance(
        ::Val{true}, model::T, variance_map, corner_x, corner_y, i, j
    ) where {T}
    return variance_map[corner_y + i - 1, corner_x + j - 1]
end

# Objective the Levenberg-Marquardt loop climbs, one pixel, either camera: the
# Poisson-shifted likelihood (d + v) log(mu + v) - (mu + v) with v the read-noise variance
# (v = 0: the Poisson log-likelihood). Its gradient (d - mu) / (mu + v) and Fisher
# 1 / (mu + v) are what the update uses. Written as the deviance s log(1 + r / s) - r,
# s = d + v, r = mu - d, which has the same dependence on mu but a float32 error of about
# eps * |r| rather than eps * d.
@inline function _fit_objective(data::T, model::T, v::T) where {T}
    s = data + v
    r = model - data
    s > zero(T) || return s * log(model + v) - (model + v)
    return s * log1p(r / s) - r
end

# Solve (L L') x = b for the Cholesky factor L (lower triangle) of
# static_cholesky_decomposition
@inline function static_cholesky_solve(
        L::SMatrix{N, N, T}, b::SVector{N, T}
    ) where {N, T}
    x = b
    @inbounds for i in 1:N
        s = x[i]
        for k in 1:(i - 1)
            s -= L[i, k] * x[k]
        end
        x = Base.setindex(x, s / L[i, i], i)
    end
    @inbounds for i in N:-1:1
        s = x[i]
        for k in (i + 1):N
            s -= L[k, i] * x[k]
        end
        x = Base.setindex(x, s / L[i, i], i)
    end
    return x
end

# Log-likelihood gain below which a Gaussian fit stops: the gain the full Newton step
# predicts, g' F⁻¹ g / 2 over the free parameters. F⁻¹ is the CRLB covariance at θ, so
# g' F⁻¹ g is the squared distance to the maximum of the local quadratic model in CRLB
# units (Mahalanobis); a gain of 1e-4 is a distance of sqrt(2e-4) = 0.014 CRLB.
const _LM_GAIN_TOL = 1.0e-4

# Fit one ROI from the start θ0 and return θ: Levenberg-Marquardt on the full Fisher
# matrix (`_lm_fit`), stopping once the predicted gain is below `_LM_GAIN_TOL` or after
# `iterations` pixel passes. SplinePSFModel overrides this with several starts.
@inline function _optimize(
        roi, box_size::Int, psf_model::PSFModel{N, T}, camera_model, variance_map,
        corner_x, corner_y, constraints, iterations::Int, θ0
    ) where {N, T}
    θ, _, _ = _lm_fit(
        roi, box_size, psf_model, camera_model, variance_map, corner_x, corner_y,
        constraints, iterations, θ0, T(_LM_GAIN_TOL)
    )
    return θ
end

# Levenberg-Marquardt fit of one ROI on the full Fisher matrix, from the start θ0. Each
# iteration makes one pixel pass at the trial point: the objective, the gradient
# g_k = sum du_k (d - mu) / var and F_kl = sum du_k du_l / var (pixels whose variance is not
# positive and finite add nothing). A trial that raises the objective is accepted (g and F
# stored, lambda / 10); an equal objective is accepted with lambda unchanged; a lower one
# is rejected (lambda * 10). The next trial is theta_acc + delta with
# (F_acc + lambda diag(F_acc)) delta = g_acc, step-clamped and bounded, with a parameter
# that sits on a bound and whose gradient points out of the box held fixed; if the
# factorization fails, delta = 0 and lambda * 10. With `gain_tol > 0` the fit stops after
# an accepted pass whose predicted gain g' F⁻¹ g / 2 over the free parameters (undamped
# F_acc, held parameters removed) is below `gain_tol`; `gain_tol = 0` runs every pass.
# Returns the last accepted point, the objective there and the pixel passes made,
# `(θ_acc, obj_acc, passes)`: the first pass always accepts its start, so
# iterations = 1 returns the clamped start with its objective. The multi-start `_optimize`
# of SplinePSFModel compares these objectives.
# Immutable state only: no allocation, no throw.
@inline function _lm_fit(
        roi, box_size::Int, psf_model::PSFModel{N, T}, camera_model, variance_map,
        corner_x, corner_y, constraints::ParameterConstraints{N}, iterations::Int, θ0,
        gain_tol::T = zero(T)
    ) where {N, T}
    θ = clamp.(θ0, constraints.lower, constraints.upper)
    θ_acc = θ
    g_acc = zero(SVector{N, T})
    F_acc = zero(SMatrix{N, N, T})
    obj_acc = zero(T)
    λ = T(1.0e-3)
    passes = 0

    for iter in 1:iterations
        passes = iter
        g = zero(SVector{N, T})
        F = zero(SMatrix{N, N, T})
        obj = zero(T)
        @inbounds for j in 1:box_size, i in 1:box_size
            model, dudt = compute_pixel_derivatives(i, j, θ, psf_model)
            d = roi[i, j]
            v = _readout_variance(
                camera_model, model, variance_map, corner_x, corner_y, i, j
            )
            var = model + v
            obj += _fit_objective(d, model, v)
            if var > zero(T) && isfinite(var)
                g = g + dudt * ((d - model) / var)
                F = F + (dudt * dudt') * (one(T) / var)
            end
        end

        accepted = true
        if iter == 1 || obj > obj_acc
            θ_acc = θ
            g_acc = g
            F_acc = F
            obj_acc = obj
            if iter > 1
                λ = max(λ / 10, T(1.0e-7))
            end
        elseif obj == obj_acc
            θ_acc = θ
            g_acc = g
            F_acc = F
        else
            accepted = false
            λ = min(λ * 10, T(1.0e7))
        end

        # A parameter on a bound whose ascent gradient points out of the box is held:
        # its row and column become the identity and its gradient zero, so the solve
        # moves only the free parameters (the coupled photon and background steps
        # otherwise assume the held one moves, and the clamped trial is rejected).
        held = map(
            (θk, gk, lo, hi) -> (θk <= lo && gk < zero(T)) || (θk >= hi && gk > zero(T)),
            θ_acc, g_acc, constraints.lower, constraints.upper
        )
        g_free = g_acc
        @inbounds for k in 1:N
            if held[k]
                g_free = Base.setindex(g_free, zero(T), k)
            end
        end

        A = F_acc
        @inbounds for k in 1:N
            A = Base.setindex(A, F_acc[k, k] + λ * F_acc[k, k], k, k)
        end
        ok, Δ = _lm_solve(A, held, g_free)
        if !ok
            λ = min(λ * 10, T(1.0e7))
        end
        # Stop test. The damped gain g' A⁻¹ g never exceeds the full Newton gain g' F⁻¹ g
        # (A ⪰ F), so it is a necessary condition, and the undamped matrix is factored
        # only when it holds: the stop is exactly the full Newton gain below gain_tol.
        if accepted && ok && gain_tol > zero(T) && sum(g_free .* Δ) < 2 * gain_tol &&
                _newton_gain(F_acc, held, g_free) < gain_tol
            break
        end
        step = clamp.(Δ, -constraints.max_step, constraints.max_step)
        θ = clamp.(θ_acc + step, constraints.lower, constraints.upper)
    end
    return θ_acc, obj_acc, passes
end

# The two factorizations of an LM pass are separate, non-inlined functions: with both
# inlined into the kernel, the POCL CPU kernel (KernelAbstractions 0.10) ran Gaussian fits
# about 8 times slower than with them as calls. Results are the same either way.

# Damped step: solve (A with held rows and columns set to the identity's) Δ = g, `(ok, Δ)`;
# Δ = 0 when the factorization fails.
@noinline function _lm_solve(
        A::SMatrix{N, N, T}, held::SVector{N, Bool}, g::SVector{N, T}
    ) where {N, T}
    ok, L = static_cholesky_decomposition(_hold_rows(A, held))
    return ok, (ok ? static_cholesky_solve(L, g) : zero(SVector{N, T}))
end

# Log-likelihood gain of the full Newton step over the free parameters, g' F⁻¹ g / 2; Inf
# when F (held rows and columns set to the identity's) is not positive definite.
@noinline function _newton_gain(
        F::SMatrix{N, N, T}, held::SVector{N, Bool}, g::SVector{N, T}
    ) where {N, T}
    ok, L = static_cholesky_decomposition(_hold_rows(F, held))
    return ok ? sum(g .* static_cholesky_solve(L, g)) / 2 : T(Inf)
end

# A with the row and column of each held parameter replaced by the identity's
@inline function _hold_rows(A::SMatrix{N, N, T}, held::SVector{N, Bool}) where {N, T}
    @inbounds for k in 1:N
        if held[k]
            for l in 1:N
                A = Base.setindex(A, ifelse(k == l, one(T), zero(T)), k, l)
                A = Base.setindex(A, ifelse(k == l, one(T), zero(T)), l, k)
            end
        end
    end
    return A
end

# Unified kernel that works on both CPU and GPU
@kernel function unified_gaussian_mle_kernel!(
        results::AbstractArray{T, 2},
        uncertainties::AbstractArray{T, 2},
        # Off-diagonal: [σ_xy, σ_xz, σ_yz] = H_inv[1,2], H_inv[1,3], H_inv[2,3]
        covariances::AbstractArray{T, 2},
        log_likelihoods::AbstractArray{T, 1},
        # Pearson statistic sum (d - mu)^2 / (mu + readout variance), the p-value's input
        chi2::AbstractArray{T, 1},
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

    # Immutable working vectors avoid heap allocation from bounds-check error paths.
    θ = simple_initialize(roi, box_size, psf_model)
    θ = _optimize(
        roi, box_size, psf_model, camera_model, variance_map, corner_x, corner_y,
        constraints, iterations, θ
    )

    # Compute final log-likelihood ratio, Pearson statistic and full Fisher Information
    # Matrix for CRLB
    log_likelihood = zero(T)
    chi2_sum = zero(T)
    H = zero(SMatrix{N, N, T})  # Full Fisher Information matrix

    @inbounds for j in 1:box_size, i in 1:box_size
        model, dudt = compute_pixel_derivatives(i, j, θ, psf_model)
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

        # variance = model + readout variance (zero for IdealCamera)
        variance = model + _readout_variance(
            camera_model, model, variance_map, corner_x, corner_y, i, j
        )

        # Pearson statistic (d - mu)^2 / variance, for both cameras; for sCMOS it is
        # -2 times the log-likelihood ratio above.
        if variance > zero(T)
            chi2_sum += (data_ij - model)^2 / variance
        end

        # Fisher Information Matrix (for CRLB) - need full matrix here
        if model > zero(T)
            # Whole-matrix operations preserve multiply/divide/add order per element.
            H = H + (dudt * dudt') / variance
        end
    end

    # Invert the Fisher matrix for uncertainties (CRLB); a matrix singular to working
    # precision, or a parameter with no Fisher information, gives infinite uncertainties
    # and zero covariances.
    ok, H_inv = _invert_fisher(H, box_size * box_size)
    # Off-diagonal: covariances[1,idx] = σ_xy (all models), covariances[2,idx] = σ_xz and
    # covariances[3,idx] = σ_yz (3D only, zero otherwise)
    has_z = _has_axial_position(typeof(psf_model))
    @inbounds for k in 1:N
        results[k, idx] = θ[k]
        uncertainties[k, idx] = ok ? sqrt(H_inv[k, k]) : T(Inf)
    end
    @inbounds covariances[1, idx] = ok ? H_inv[1, 2] : zero(T)
    @inbounds covariances[2, idx] = ok && has_z ? H_inv[1, 3] : zero(T)
    @inbounds covariances[3, idx] = ok && has_z ? H_inv[2, 3] : zero(T)

    @inbounds log_likelihoods[idx] = log_likelihood
    @inbounds chi2[idx] = chi2_sum
end
