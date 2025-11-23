

"""
    integral_gaussian_1d(ii::Int, position::T, sigma::T) where T <: Real

Calculate the integral of a 1D Gaussian function.
"""
function integral_gaussian_1d(i::Int, position::T, sigma::T) where T <: Real
    half = one(T) / 2
    two = one(T) + one(T)
    
    norm = half / sigma^two
    return half * (erf((i - position + half) * sqrt(norm)) - erf((i - position - half) * sqrt(norm)))
end

"""
    derivative_integral_gaussian_1d(ii::Int, x::T, sigma::T, N::T, PSFy::T) where T <: Real

Compute the derivative of the integral of a 1D Gaussian function with respect to x.
"""
function derivative_integral_gaussian_1d(ii::Int, x::T, sigma::T, N::T, PSFy::T) where T <: Real
    half = one(T) / 2
    two = one(T) + one(T)
    
    factor_a = exp(-half * ((ii + half - x) / sigma)^two)
    factor_b = exp(-half * ((ii - half - x) / sigma)^two)

    constant = -N / sqrt(two * one(T) * pi) / sigma
    dudt = constant * (factor_a - factor_b) * PSFy
    d2udt2 = constant / sigma^two * ((ii + half - x) * factor_a - (ii - half - x) * factor_b) * PSFy

    return (dudt, d2udt2)
end

"""
    derivative_integral_gaussian_1d_sigma(i::Int, x::T, Sx::T, N::T, PSFy::T) where T <: Real

Compute the derivative of the integral of a 1D Gaussian function with respect to sigma.
"""
function derivative_integral_gaussian_1d_sigma(i::Int, x::T, Sx::T, N::T, PSFy::T) where T <: Real
    half = one(T) / 2
    two = one(T) + one(T)
    pi_val = one(T) * pi

    ax = exp(-half * ((i + half - x) / Sx)^two)
    bx = exp(-half * ((i - half - x) / Sx)^two)

    dudt = -N / sqrt(two * pi_val) / Sx / Sx * (ax * (i - x + half) - bx * (i - x - half)) * PSFy
    d2udt2 = -two / Sx * dudt - N / sqrt(two * pi_val) / Sx^5 * (ax * (i - x + half)^3 - bx * (i - x - half)^3) * PSFy

    return (dudt, d2udt2)
end



"""
    derivative_integral_gaussian_2d_sigma(i::Int, j::Int, x::T, y::T, S::T, N::T, PSFx::T, PSFy::T) where T <: Real

Compute the derivative of the integral of a 2D Gaussian function with respect to sigma.
"""
function derivative_integral_gaussian_2d_sigma(i::Int, j::Int, x::T, y::T, S::T, N::T, PSFx::T, PSFy::T) where T <: Real
    (dSx, ddSx) = derivative_integral_gaussian_1d_sigma(j, x, S, N, PSFy)
    (dSy, ddSy) = derivative_integral_gaussian_1d_sigma(i, y, S, N, PSFx)
    dudt = dSx + dSy
    d2udt2 = ddSx + ddSy
    return (dudt, d2udt2)
end


"""
    compute_alpha(z::T, Ax::T, Bx::T, d::T) where T <: Real

Compute the alpha value based on the given parameters.
"""
function compute_alpha(z::T, Ax::T, Bx::T, d::T) where T <: Real
    one_val = one(T)
    two_val = one_val + one_val
    three_val = two_val + one_val
    four_val = three_val + one_val

    z_d_ratio = z / d
    return one_val + z_d_ratio^two_val + Ax * z_d_ratio^three_val + Bx * z_d_ratio^four_val
end


"""
    derivative_alpha_z(z::T, Ax::T, Bx::T, d::T) where T <: Real

Compute the derivative of alpha with respect to z.
"""
function derivative_alpha_z(z::T, Ax::T, Bx::T, d::T) where T <: Real
    two_val = one(T) + one(T)
    three_val = two_val + one(T)
    four_val = three_val + one(T)

    d_square = d^two_val
    d_cube = d^three_val
    d_quad = d^four_val

    return two_val * z / d_square + three_val * Ax * z^two_val / d_cube + four_val * Bx * z^three_val / d_quad
end

"""
    second_derivative_alpha_z(z::T, Ax::T, Bx::T, d::T) where T <: Real

Compute the second derivative of alpha with respect to z.
"""
function second_derivative_alpha_z(z::T, Ax::T, Bx::T, d::T) where T <: Real
    two_val = one(T) + one(T)
    three_val = two_val + one(T)
    four_val = three_val + one(T)
    six_val = three_val + three_val
    twelve_val = six_val + six_val

    d_square = d^two_val
    d_cube = d^three_val
    d_quad = d^four_val

    return two_val / d_square + six_val * Ax * z / d_cube + twelve_val * Bx * z^two_val / d_quad
end



"""
    center_of_mass_2d(sz::Int, data::Array{T}) where T <: Real

Compute the center of mass of a square 2D data array.
"""
function center_of_mass_2d(sz::Int, data::Array{T}) where T <: Real
    tmp_x = zero(T)
    tmp_y = zero(T)
    tmp_sum = zero(T)

    for i = 1:sz
        for j = 1:sz
            tmp_x += data[sz*(j-1)+i] * T(i)
            tmp_y += data[sz*(j-1)+i] * T(j)
            tmp_sum += data[sz*(j-1)+i]
        end
    end

    return (tmp_x / tmp_sum, tmp_y / tmp_sum)
end


"""
    gaussian_max_min_2d(sz::Int, sigma::T, data::Array{T}) where T <: Real

Compute the maximum and minimum values after applying a Gaussian filter to a 2D data array.
"""
function gaussian_max_min_2d(sz::Int, sigma::T, data::Array{T}) where T <: Real
    filtered_pixel = zero(T)
    sum_val = zero(T)
    max_n = zero(T)
    min_bg = typemax(T)
    norm = T(0.5) / sigma^T(2)

    for k = 0:sz-1
        for l = 0:sz-1
            filtered_pixel, sum_val = zero(T), zero(T)
            for i = 0:sz-1
                for j = 0:sz-1
                    filtered_pixel += exp(-(i - k)^T(2) * norm) * exp(-(l - j)^T(2) * norm) * data[i*sz+j+1]
                    sum_val += exp(-(i - k)^T(2) * norm) * exp(-(l - j)^T(2) * norm)
                end
            end

            filtered_pixel /= sum_val

            max_n = max(max_n, filtered_pixel)
            min_bg = min(min_bg, filtered_pixel)
        end
    end
    return (max_n, min_bg)
end


function derivative_integral_gaussian_2d_z(i::Int, j::Int, theta, PSFSigma_x::T, PSFSigma_y::T, Ax::T, Ay::T, Bx::T, By::T, gamma::T, d::T, dudt, d2udt2) where T <: Real
    # Changed: Now expects theta in order [x, y, z, N, bg] to match our standard
    x = theta[1]
    y = theta[2] 
    z = theta[3]  # Changed from theta[5]
    N = theta[4]  # Changed from theta[3]
    bg = theta[5] # Changed from theta[4]
    
    alphax = compute_alpha(z - gamma, Ax, Bx, d)
    alphay = compute_alpha(z + gamma, Ay, By, d)

    Sx = PSFSigma_x * sqrt(alphax)
    Sy = PSFSigma_y * sqrt(alphay)

    # FIXED: Use correct Julia convention - j for x (column), i for y (row)
    PSFx = integral_gaussian_1d(j, x, Sx)
    PSFy = integral_gaussian_1d(i, y, Sy)

    (dudt[1], d2udt2[1]) = derivative_integral_gaussian_1d(j, x, Sx, N, PSFy)
    (dudt[2], d2udt2[2]) = derivative_integral_gaussian_1d(i, y, Sy, N, PSFx)
    (dSx, ddSx) = derivative_integral_gaussian_1d_sigma(j, x, Sx, N, PSFy)
    (dSy, ddSy) = derivative_integral_gaussian_1d_sigma(i, y, Sy, N, PSFx)

    dSdalpha_x = PSFSigma_x / T(2) / sqrt(alphax)
    dSdalpha_y = PSFSigma_y / T(2) / sqrt(alphay)

    dSdzx = dSdalpha_x * derivative_alpha_z(z - gamma, Ax, Bx, d)
    dSdzy = dSdalpha_y * derivative_alpha_z(z + gamma, Ay, By, d)
    dudt[3] = dSx * dSdzx + dSy * dSdzy  # Changed from dudt[5] to dudt[3] for z position

    d2Sdalpha2_x = -PSFSigma_x / T(4) / alphax^T(1.5)
    d2Sdalpha2_y = -PSFSigma_y / T(4) / alphay^T(1.5)

    ddSddzx = d2Sdalpha2_x * derivative_alpha_z(z - gamma, Ax, Bx, d)^2 + dSdalpha_x * second_derivative_alpha_z(z - gamma, Ax, Bx, d)
    ddSddzy = d2Sdalpha2_y * derivative_alpha_z(z + gamma, Ay, By, d)^2 + dSdalpha_y * second_derivative_alpha_z(z + gamma, Ay, By, d)

    d2udt2[3] = ddSx * (dSdzx * dSdzx) + dSx * ddSddzx +
                ddSy * (dSdzy * dSdzy) + dSy * ddSddzy  # Changed from d2udt2[5] to d2udt2[3]

    return (PSFx, PSFy)
end
