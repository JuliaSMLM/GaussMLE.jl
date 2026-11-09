"""
Analytical derivatives for PSF models using GaussLib functions
"""

# Each method returns the expected count of pixel (i, j) and its first derivatives with
# respect to the parameters, `(model, dudt)`: the Levenberg-Marquardt fit and the Fisher
# matrix use no second derivatives.

# Fixed sigma model derivatives
@inline function compute_pixel_derivatives(i, j, θ::Params{4}, psf::GaussianXYNB)
    x, y, N, bg = θ

    # Compute PSF values
    psf_x = integral_gaussian_1d(j, x, psf.σ)
    psf_y = integral_gaussian_1d(i, y, psf.σ)

    # Get derivatives using GaussLib functions
    dudt_x = derivative_integral_gaussian_1d(j, x, psf.σ, N, psf_y)
    dudt_y = derivative_integral_gaussian_1d(i, y, psf.σ, N, psf_x)

    # Model value
    model = bg + N * psf_x * psf_y

    # First derivatives
    dudt = @SVector [
        dudt_x,               # ∂/∂x
        dudt_y,               # ∂/∂y
        psf_x * psf_y,        # ∂/∂N
        one(Float32),          # ∂/∂bg
    ]

    return model, dudt
end

# Variable sigma model derivatives
@inline function compute_pixel_derivatives(i, j, θ::Params{5}, ::GaussianXYNBS)
    x, y, N, bg, σ = θ

    # Compute PSF values
    psf_x = integral_gaussian_1d(j, x, σ)
    psf_y = integral_gaussian_1d(i, y, σ)

    # Get derivatives using GaussLib functions
    dudt_x = derivative_integral_gaussian_1d(j, x, σ, N, psf_y)
    dudt_y = derivative_integral_gaussian_1d(i, y, σ, N, psf_x)
    dudt_s = derivative_integral_gaussian_2d_sigma(i, j, x, y, σ, N, psf_x, psf_y)

    # Model value
    model = bg + N * psf_x * psf_y

    dudt = @SVector [
        dudt_x,
        dudt_y,
        psf_x * psf_y,
        one(Float32),
        dudt_s,
    ]

    return model, dudt
end

# Anisotropic model derivatives
@inline function compute_pixel_derivatives(i, j, θ::Params{6}, ::GaussianXYNBSXSY)
    x, y, N, bg, σx, σy = θ

    # Compute PSF values
    psf_x = integral_gaussian_1d(j, x, σx)
    psf_y = integral_gaussian_1d(i, y, σy)

    # Get derivatives using GaussLib functions
    dudt_x = derivative_integral_gaussian_1d(j, x, σx, N, psf_y)
    dudt_y = derivative_integral_gaussian_1d(i, y, σy, N, psf_x)
    dudt_sx = derivative_integral_gaussian_1d_sigma(j, x, σx, N, psf_y)
    dudt_sy = derivative_integral_gaussian_1d_sigma(i, y, σy, N, psf_x)

    # Model value
    model = bg + N * psf_x * psf_y

    dudt = @SVector [
        dudt_x,
        dudt_y,
        psf_x * psf_y,
        one(Float32),
        dudt_sx,
        dudt_sy,
    ]

    return model, dudt
end

# Astigmatic model derivatives
@inline function compute_pixel_derivatives(i, j, θ::Params{5}, psf::AstigmaticXYZNB)
    x, y, z, N, bg = θ

    # Compute z-dependent widths using astigmatic model
    # α(z) = 1 + (z/d)² + A(z/d)³ + B(z/d)⁴
    z_minus_gamma = z - psf.γ
    z_plus_gamma = z + psf.γ

    # Compute alpha values (inline to avoid function call overhead)
    z_d_x = z_minus_gamma / psf.d
    αx = one(Float32) + z_d_x^2 + psf.Ax * z_d_x^3 + psf.Bx * z_d_x^4

    z_d_y = z_plus_gamma / psf.d
    αy = one(Float32) + z_d_y^2 + psf.Ay * z_d_y^3 + psf.By * z_d_y^4

    # Widths at this z position
    σx = psf.σx₀ * sqrt(αx)
    σy = psf.σy₀ * sqrt(αy)

    # Compute PSF values
    psf_x = integral_gaussian_1d(j, x, σx)
    psf_y = integral_gaussian_1d(i, y, σy)

    # Get x, y derivatives (same as GaussianXYNBSXSY but with z-dependent sigmas)
    dudt_x = derivative_integral_gaussian_1d(j, x, σx, N, psf_y)
    dudt_y = derivative_integral_gaussian_1d(i, y, σy, N, psf_x)

    # Compute z derivatives via chain rule: dmodel/dz = dmodel/dσx * dσx/dz + dmodel/dσy *
    # dσy/dz
    dudt_sx = derivative_integral_gaussian_1d_sigma(j, x, σx, N, psf_y)
    dudt_sy = derivative_integral_gaussian_1d_sigma(i, y, σy, N, psf_x)

    # dσ/dα = σ₀/(2√α), dα/dz for astigmatic model
    dαx_dz = (2 * z_d_x + 3 * psf.Ax * z_d_x^2 + 4 * psf.Bx * z_d_x^3) / psf.d
    dαy_dz = (2 * z_d_y + 3 * psf.Ay * z_d_y^2 + 4 * psf.By * z_d_y^3) / psf.d

    dσx_dz = (psf.σx₀ / (2 * sqrt(αx))) * dαx_dz
    dσy_dz = (psf.σy₀ / (2 * sqrt(αy))) * dαy_dz

    dudt_z = dudt_sx * dσx_dz + dudt_sy * dσy_dz

    # Model value
    model = bg + N * psf_x * psf_y

    # First derivatives
    dudt = @SVector [
        dudt_x,          # ∂/∂x
        dudt_y,          # ∂/∂y
        dudt_z,          # ∂/∂z
        psf_x * psf_y,   # ∂/∂N
        one(Float32),     # ∂/∂bg
    ]

    return model, dudt
end
