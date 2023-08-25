

"""
    integral_gaussian_1d(ii::Int, position::Real, sigma::Real)

Calculate the integral of a 1D Gaussian function.
"""
function integral_gaussian_1d(ii::Int, position::Real, sigma::Real)
    norm = 0.5 / sigma^2
    return 0.5 * (erf((ii - position + 0.5) * sqrt(norm)) - erf((ii - position - 0.5) * sqrt(norm)))
end

"""
    compute_alpha(z::Real, Ax::Real, Bx::Real, d::Real)

Compute the alpha value based on the given parameters.
"""
function compute_alpha(z::Real, Ax::Real, Bx::Real, d::Real)
    return 1.0 + (z / d)^2 + Ax * (z / d)^3 + Bx * (z / d)^4
end

"""
    derivative_alpha_z(z::Real, Ax::Real, Bx::Real, d::Real)

Compute the derivative of alpha with respect to z.
"""
function derivative_alpha_z(z::Real, Ax::Real, Bx::Real, d::Real)
    return 2.0 * z / d^2 + 3.0 * Ax * z^2 / d^3 + 4.0 * Bx * z^3 / d^4
end

"""
    second_derivative_alpha_z(z::Real, Ax::Real, Bx::Real, d::Real)

Compute the second derivative of alpha with respect to z.
"""
function second_derivative_alpha_z(z::Real, Ax::Real, Bx::Real, d::Real)
    return 2.0 / d^2 + 6.0 * Ax * z / d^3 + 12.0 * Bx * z^2 / d^4
end

"""
    derivative_integral_gaussian_1d(ii::Int, x::Real, sigma::Real, N::Real, PSFy::Real)

Compute the derivative of the integral of a 1D Gaussian function.
"""
function derivative_integral_gaussian_1d(ii::Int, x::Real, sigma::Real, N::Real, PSFy::Real)
    factor_a = exp(-0.5 * ((ii + 0.5 - x) / sigma)^2)
    factor_b = exp(-0.5 * ((ii - 0.5 - x) / sigma)^2)

    constant = -N / sqrt(2.0 * pi) / sigma
    dudt = constant * (factor_a - factor_b) * PSFy
    d2udt2 = constant / sigma^2 * ((ii + 0.5 - x) * factor_a - (ii - 0.5 - x) * factor_b) * PSFy

    return (dudt, d2udt2)
end


"""
    center_of_mass_2d(sz::Int, data::Array{<:Real})

Compute the center of mass of a square 2D data array.
"""
function center_of_mass_2d(sz::Int, data::Array{<:Real})
    tmp_x, tmp_y, tmp_sum = 0.0, 0.0, 0.0

    for ii = 1:sz
        for jj = 1:sz
            tmp_x += data[sz*(jj-1)+ii] * ii
            tmp_y += data[sz*(jj-1)+ii] * jj
            tmp_sum += data[sz*(jj-1)+ii]
        end
    end

    return (tmp_x / tmp_sum, tmp_y / tmp_sum)
end


"""
    gaussian_max_min_2d(sz::Int, sigma::Real, data::Array{<:Real})

Compute the maximum and minimum values after applying a Gaussian filter to a 2D data array.
"""
function gaussian_max_min_2d(sz::Int, sigma::Real, data::Array{<:Real})
    filtered_pixel, sum_val, max_n, min_bg = 0.0, 0.0, 0.0, 1e10
    norm = 0.5 / sigma^2

    for kk = 0:sz-1
        for ll = 0:sz-1
            filtered_pixel, sum_val = 0.0, 0.0
            for ii = 0:sz-1
                for jj = 0:sz-1
                    filtered_pixel += exp(-(ii - kk)^2 * norm) * exp(-(ll - jj)^2 * norm) * data[ii*sz+jj+1]
                    sum_val += exp(-(ii - kk)^2 * norm) * exp(-(ll - jj)^2 * norm)
                end
            end

            filtered_pixel /= sum_val

            max_n = max(max_n, filtered_pixel)
            min_bg = min(min_bg, filtered_pixel)
        end
    end
    return (max_n, min_bg)
end

"""
    derivative_integral_gaussian_1d_sigma(ii::Int, x::Real, Sx::Real, N::Real, PSFy::Real)

Compute the derivative of the integral of a 1D Gaussian function with respect to sigma.
"""
function derivative_integral_gaussian_1d_sigma(ii::Int, x::Real, Sx::Real, N::Real, PSFy::Real)
    ax = exp(-0.5 * ((ii + 0.5 - x) / Sx)^2)
    bx = exp(-0.5 * ((ii - 0.5 - x) / Sx)^2)
    dudt = -N / sqrt(2.0 * pi) / Sx / Sx * (ax * (ii - x + 0.5) - bx * (ii - x - 0.5)) * PSFy
    d2udt2 = -2.0 / Sx * dudt - N / sqrt(2.0 * pi) / Sx^5 * (ax * (ii - x + 0.5)^3 - bx * (ii - x - 0.5)^3) * PSFy
    return (dudt, d2udt2)
end


"""
    derivative_integral_gaussian_2d_sigma(ii::Int, jj::Int, x::Real, y::Real, S::Real, N::Real, PSFx::Real, PSFy::Real)

Compute the derivative of the integral of a 2D Gaussian function with respect to sigma.
"""
function derivative_integral_gaussian_2d_sigma(ii::Int, jj::Int, x::Real, y::Real, S::Real, N::Real, PSFx::Real, PSFy::Real)
    (dSx, ddSx) = derivative_integral_gaussian_1d_sigma(ii, x, S, N, PSFy)
    (dSy, ddSy) = derivative_integral_gaussian_1d_sigma(jj, y, S, N, PSFx)
    dudt = dSx + dSy
    d2udt2 = ddSx + ddSy
    return (dudt, d2udt2)
end

"""
    derivative_integral_gaussian_2d_z(ii::Int, jj::Int, theta, PSFSigma_x::Real, PSFSigma_y::Real, Ax::Real, Ay::Real, Bx::Real, By::Real, gamma::Real, d::Real, dudt, d2udt2)

Compute the derivative of the integral of a 2D Gaussian function with respect to z.
"""
function derivative_integral_gaussian_2d_z(ii::Int, jj::Int, theta, PSFSigma_x::Real, PSFSigma_y::Real, Ax::Real, Ay::Real, Bx::Real, By::Real, gamma::Real, d::Real, dudt, d2udt2)
    z = theta[5]
    alphax = compute_alpha(z - gamma, Ax, Bx, d)
    alphay = compute_alpha(z + gamma, Ay, By, d)

    Sx = PSFSigma_x * sqrt(alphax)
    Sy = PSFSigma_y * sqrt(alphay)

    PSFx = integral_gaussian_1d(ii, theta[1], Sx)
    PSFy = integral_gaussian_1d(jj, theta[2], Sy)

    (dudt[1], d2udt2[1]) = derivative_integral_gaussian_1d(ii, theta[1], Sx, theta[3], PSFy)
    (dudt[2], d2udt2[2]) = derivative_integral_gaussian_1d(jj, theta[2], Sy, theta[3], PSFx)
    (dSx, ddSx) = derivative_integral_gaussian_1d_sigma(ii, theta[1], Sx, theta[3], PSFy)
    (dSy, ddSy) = derivative_integral_gaussian_1d_sigma(jj, theta[2], Sy, theta[3], PSFx)

    dSdalpha_x = PSFSigma_x / 2.0 / sqrt(alphax)
    dSdalpha_y = PSFSigma_y / 2.0 / sqrt(alphay)

    dSdzx = dSdalpha_x * derivative_alpha_z(z - gamma, Ax, Bx, d)
    dSdzy = dSdalpha_y * derivative_alpha_z(z + gamma, Ay, By, d)
    dudt[5] = dSx * dSdzx + dSy * dSdzy

    d2Sdalpha2_x = -PSFSigma_x / 4.0 / alphax^1.5
    d2Sdalpha2_y = -PSFSigma_y / 4.0 / alphay^1.5

    ddSddzx = d2Sdalpha2_x * derivative_alpha_z(z - gamma, Ax, Bx, d)^2 + dSdalpha_x * second_derivative_alpha_z(z - gamma, Ax, Bx, d)
    ddSddzy = d2Sdalpha2_y * derivative_alpha_z(z + gamma, Ay, By, d)^2 + dSdalpha_y * second_derivative_alpha_z(z + gamma, Ay, By, d)

    d2udt2[5] = ddSx * (dSdzx * dSdzx) + dSx * ddSddzx +
                ddSy * (dSdzy * dSdzy) + dSy * ddSddzy

    return (PSFx, PSFy)
end

