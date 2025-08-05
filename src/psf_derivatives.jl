"""
Analytical derivatives for PSF models
"""

using StaticArrays
using SpecialFunctions: erf

# Helper function for Gaussian derivative
@inline function gaussian_derivative_1d(x::T, σ::T) where T
    norm = one(T) / (sqrt(T(2π)) * σ)
    return norm * exp(-x^2 / (2 * σ^2))
end

# Fixed sigma model derivatives
@inline function compute_pixel_derivatives(i, j, θ::Params{4}, psf::GaussianXYNB)
    x, y, N, bg = θ
    
    # Integrated Gaussians
    dx = Float32(i) - x
    dy = Float32(j) - y
    
    psf_x = integrated_gaussian_1d(dx, psf.σ)
    psf_y = integrated_gaussian_1d(dy, psf.σ)
    
    # Derivatives of integrated Gaussians
    dpsf_x = gaussian_derivative_1d(dx, psf.σ)
    dpsf_y = gaussian_derivative_1d(dy, psf.σ)
    
    # Model value
    model = bg + N * psf_x * psf_y
    
    # First derivatives
    dudt = @SVector [
        -N * dpsf_x * psf_y,  # ∂/∂x
        -N * psf_x * dpsf_y,  # ∂/∂y
        psf_x * psf_y,        # ∂/∂N
        one(Float32)          # ∂/∂bg
    ]
    
    # Second derivatives (diagonal Hessian approximation)
    d2psf_x = -dx * dpsf_x / (psf.σ^2)
    d2psf_y = -dy * dpsf_y / (psf.σ^2)
    
    d2udt2 = @SMatrix [
        -N*d2psf_x*psf_y      -N*dpsf_x*dpsf_y     -dpsf_x*psf_y    0;
        -N*dpsf_x*dpsf_y      -N*psf_x*d2psf_y     -psf_x*dpsf_y    0;
        -dpsf_x*psf_y         -psf_x*dpsf_y         0                0;
        0                      0                     0                0
    ]
    
    return model, dudt, d2udt2
end

# Variable sigma model derivatives
@inline function compute_pixel_derivatives(i, j, θ::Params{5}, ::GaussianXYNBS)
    x, y, N, bg, σ = θ
    
    dx = Float32(i) - x
    dy = Float32(j) - y
    
    psf_x = integrated_gaussian_1d(dx, σ)
    psf_y = integrated_gaussian_1d(dy, σ)
    
    dpsf_x = gaussian_derivative_1d(dx, σ)
    dpsf_y = gaussian_derivative_1d(dy, σ)
    
    # Derivatives with respect to σ
    dpsf_x_dσ = dx * dpsf_x / σ^2
    dpsf_y_dσ = dy * dpsf_y / σ^2
    
    model = bg + N * psf_x * psf_y
    
    dudt = @SVector [
        -N * dpsf_x * psf_y,
        -N * psf_x * dpsf_y,
        psf_x * psf_y,
        one(Float32),
        N * (dpsf_x_dσ * psf_y + psf_x * dpsf_y_dσ)
    ]
    
    # Simplified second derivatives
    d2udt2 = @SMatrix zeros(Float32, 5, 5)
    
    return model, dudt, d2udt2
end

# Anisotropic model derivatives
@inline function compute_pixel_derivatives(i, j, θ::Params{6}, ::GaussianXYNBSXSY)
    x, y, N, bg, σx, σy = θ
    
    dx = Float32(i) - x
    dy = Float32(j) - y
    
    psf_x = integrated_gaussian_1d(dx, σx)
    psf_y = integrated_gaussian_1d(dy, σy)
    
    dpsf_x = gaussian_derivative_1d(dx, σx)
    dpsf_y = gaussian_derivative_1d(dy, σy)
    
    dpsf_x_dσx = dx * dpsf_x / σx^2
    dpsf_y_dσy = dy * dpsf_y / σy^2
    
    model = bg + N * psf_x * psf_y
    
    dudt = @SVector [
        -N * dpsf_x * psf_y,
        -N * psf_x * dpsf_y,
        psf_x * psf_y,
        one(Float32),
        N * dpsf_x_dσx * psf_y,
        N * psf_x * dpsf_y_dσy
    ]
    
    # Simplified second derivatives
    d2udt2 = @SMatrix zeros(Float32, 6, 6)
    
    return model, dudt, d2udt2
end

# Astigmatic model derivatives
@inline function compute_pixel_derivatives(i, j, θ::Params{5}, psf::AstigmaticXYZNB)
    x, y, z, N, bg = θ
    
    dx = Float32(i) - x
    dy = Float32(j) - y
    
    # Width calculation
    zx = (z - psf.γ) / psf.d
    zy = (z + psf.γ) / psf.d
    
    σx = psf.σx₀ * sqrt(one(Float32) + zx^2 + psf.Ax * zx^3 + psf.Bx * zx^4)
    σy = psf.σy₀ * sqrt(one(Float32) + zy^2 + psf.Ay * zy^3 + psf.By * zy^4)
    
    psf_x = integrated_gaussian_1d(dx, σx)
    psf_y = integrated_gaussian_1d(dy, σy)
    
    dpsf_x = gaussian_derivative_1d(dx, σx)
    dpsf_y = gaussian_derivative_1d(dy, σy)
    
    # Derivatives of σ with respect to z
    dσx_dz = psf.σx₀ / (2 * σx) * (2*zx + 3*psf.Ax*zx^2 + 4*psf.Bx*zx^3) / psf.d
    dσy_dz = psf.σy₀ / (2 * σy) * (2*zy + 3*psf.Ay*zy^2 + 4*psf.By*zy^3) / psf.d
    
    # PSF derivatives with respect to σ
    dpsf_x_dσx = dx * dpsf_x / σx^2
    dpsf_y_dσy = dy * dpsf_y / σy^2
    
    model = bg + N * psf_x * psf_y
    
    dudt = @SVector [
        -N * dpsf_x * psf_y,
        -N * psf_x * dpsf_y,
        N * (dpsf_x_dσx * dσx_dz * psf_y + psf_x * dpsf_y_dσy * dσy_dz),
        psf_x * psf_y,
        one(Float32)
    ]
    
    # Simplified second derivatives
    d2udt2 = @SMatrix zeros(Float32, 5, 5)
    
    return model, dudt, d2udt2
end

export compute_pixel_derivatives