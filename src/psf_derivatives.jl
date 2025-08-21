"""
Analytical derivatives for PSF models using GaussLib functions
"""

using StaticArrays
using .GaussLib: integral_gaussian_1d, derivative_integral_gaussian_1d, 
                 derivative_integral_gaussian_1d_sigma, derivative_integral_gaussian_2d_sigma,
                 derivative_integral_gaussian_2d_z, compute_alpha, derivative_alpha_z

# Fixed sigma model derivatives
@inline function compute_pixel_derivatives(i, j, θ::Params{4}, psf::GaussianXYNB)
    x, y, N, bg = θ
    
    # Compute PSF values
    psf_x = integral_gaussian_1d(i, x, psf.σ)
    psf_y = integral_gaussian_1d(j, y, psf.σ)
    
    # Get derivatives using GaussLib functions
    dudt_x, d2udt2_x = derivative_integral_gaussian_1d(i, x, psf.σ, N, psf_y)
    dudt_y, d2udt2_y = derivative_integral_gaussian_1d(j, y, psf.σ, N, psf_x)
    
    # Model value
    model = bg + N * psf_x * psf_y
    
    # First derivatives
    dudt = @SVector [
        dudt_x,               # ∂/∂x
        dudt_y,               # ∂/∂y
        psf_x * psf_y,        # ∂/∂N
        one(Float32)          # ∂/∂bg
    ]
    
    # Second derivatives (diagonal only for scalar Newton-Raphson)
    d2udt2 = @SMatrix [
        d2udt2_x            0               0             0;
        0                   d2udt2_y        0             0;
        0                   0               zero(Float32) 0;
        0                   0               0             zero(Float32)
    ]
    
    return model, dudt, d2udt2
end

# Variable sigma model derivatives
@inline function compute_pixel_derivatives(i, j, θ::Params{5}, ::GaussianXYNBS)
    x, y, N, bg, σ = θ
    
    # Compute PSF values
    psf_x = integral_gaussian_1d(i, x, σ)
    psf_y = integral_gaussian_1d(j, y, σ)
    
    # Get derivatives using GaussLib functions
    dudt_x, d2udt2_x = derivative_integral_gaussian_1d(i, x, σ, N, psf_y)
    dudt_y, d2udt2_y = derivative_integral_gaussian_1d(j, y, σ, N, psf_x)
    dudt_s, d2udt2_s = derivative_integral_gaussian_2d_sigma(i, j, x, y, σ, N, psf_x, psf_y)
    
    # Model value
    model = bg + N * psf_x * psf_y
    
    dudt = @SVector [
        dudt_x,
        dudt_y,
        psf_x * psf_y,
        one(Float32),
        dudt_s
    ]
    
    # Diagonal second derivatives only
    d2udt2 = @SMatrix [
        d2udt2_x  0         0  0  0;
        0         d2udt2_y  0  0  0;
        0         0         0  0  0;
        0         0         0  0  0;
        0         0         0  0  d2udt2_s
    ]
    
    return model, dudt, d2udt2
end

# Anisotropic model derivatives
@inline function compute_pixel_derivatives(i, j, θ::Params{6}, ::GaussianXYNBSXSY)
    x, y, N, bg, σx, σy = θ
    
    # Compute PSF values
    psf_x = integral_gaussian_1d(i, x, σx)
    psf_y = integral_gaussian_1d(j, y, σy)
    
    # Get derivatives using GaussLib functions
    dudt_x, d2udt2_x = derivative_integral_gaussian_1d(i, x, σx, N, psf_y)
    dudt_y, d2udt2_y = derivative_integral_gaussian_1d(j, y, σy, N, psf_x)
    dudt_sx, d2udt2_sx = derivative_integral_gaussian_1d_sigma(i, x, σx, N, psf_y)
    dudt_sy, d2udt2_sy = derivative_integral_gaussian_1d_sigma(j, y, σy, N, psf_x)
    
    # Model value
    model = bg + N * psf_x * psf_y
    
    dudt = @SVector [
        dudt_x,
        dudt_y,
        psf_x * psf_y,
        one(Float32),
        dudt_sx,
        dudt_sy
    ]
    
    # Diagonal second derivatives only
    d2udt2 = @SMatrix [
        d2udt2_x  0         0  0  0         0;
        0         d2udt2_y  0  0  0         0;
        0         0         0  0  0         0;
        0         0         0  0  0         0;
        0         0         0  0  d2udt2_sx 0;
        0         0         0  0  0         d2udt2_sy
    ]
    
    return model, dudt, d2udt2
end

# Astigmatic model derivatives
@inline function compute_pixel_derivatives(i, j, θ::Params{5}, psf::AstigmaticXYZNB)
    x, y, z, N, bg = θ
    
    # Use GaussLib function which computes all derivatives consistently
    # GaussLib now expects same parameter order as us: [x, y, z, N, bg]
    dudt_arr = @MVector zeros(Float32, 5)
    d2udt2_arr = @MVector zeros(Float32, 5)
    
    # This function computes PSF and all derivatives accounting for z-dependent widths
    PSFx, PSFy = derivative_integral_gaussian_2d_z(
        i, j, θ,  # Pass θ directly - same order now!
        psf.σx₀, psf.σy₀, psf.Ax, psf.Ay, psf.Bx, psf.By, psf.γ, psf.d, 
        dudt_arr, d2udt2_arr
    )
    
    # Model value
    model = bg + N * PSFx * PSFy
    
    # GaussLib now returns derivatives in our standard order [x,y,z,N,bg]
    # But it only computes x,y,z derivatives. N and bg we compute here.
    dudt = @SVector [
        dudt_arr[1],          # ∂/∂x (from GaussLib)
        dudt_arr[2],          # ∂/∂y (from GaussLib)
        dudt_arr[3],          # ∂/∂z (from GaussLib) - now in position 3!
        PSFx * PSFy,          # ∂/∂N
        one(Float32)          # ∂/∂bg
    ]
    
    # Diagonal second derivatives only
    d2udt2 = @SMatrix [
        d2udt2_arr[1]  0               0               0  0;
        0              d2udt2_arr[2]   0               0  0;
        0              0               d2udt2_arr[3]   0  0;  # z is now position 3!
        0              0               0               0  0;
        0              0               0               0  0
    ]
    
    return model, dudt, d2udt2
end

export compute_pixel_derivatives