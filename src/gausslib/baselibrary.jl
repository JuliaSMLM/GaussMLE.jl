"""
    integral_gaussian_1d(ii::Int, position::T, sigma::T) where T <: Real

Calculate the integral of a 1D Gaussian function.
"""
function integral_gaussian_1d(i::Int, position::T, sigma::T) where {T <: Real}
    half = one(T) / 2
    two = one(T) + one(T)

    norm = half / sigma^two
    return half * (
        erf((i - position + half) * sqrt(norm)) -
            erf((i - position - half) * sqrt(norm))
    )
end

"""
    derivative_integral_gaussian_1d(ii::Int, x::T, sigma::T, N::T, PSFy::T) where T <: Real

Compute the derivative of the integral of a 1D Gaussian function with respect to x.
"""
function derivative_integral_gaussian_1d(
        ii::Int, x::T, sigma::T, N::T, PSFy::T
    ) where {T <: Real}
    half = one(T) / 2
    two = one(T) + one(T)

    factor_a = exp(-half * ((ii + half - x) / sigma)^two)
    factor_b = exp(-half * ((ii - half - x) / sigma)^two)

    constant = -N / sqrt(two * one(T) * pi) / sigma
    return constant * (factor_a - factor_b) * PSFy
end

"""
    derivative_integral_gaussian_1d_sigma(i::Int, x::T, Sx::T, N::T, PSFy::T) where
        T <: Real

Compute the derivative of the integral of a 1D Gaussian function with respect to sigma.
"""
function derivative_integral_gaussian_1d_sigma(
        i::Int, x::T, Sx::T, N::T, PSFy::T
    ) where {T <: Real}
    half = one(T) / 2
    two = one(T) + one(T)
    pi_val = one(T) * pi

    ax = exp(-half * ((i + half - x) / Sx)^two)
    bx = exp(-half * ((i - half - x) / Sx)^two)

    return -N / sqrt(two * pi_val) / Sx / Sx *
        (ax * (i - x + half) - bx * (i - x - half)) * PSFy
end


"""
    derivative_integral_gaussian_2d_sigma(i::Int, j::Int, x::T, y::T, S::T, N::T, PSFx::T,
        PSFy::T) where T <: Real

Compute the derivative of the integral of a 2D Gaussian function with respect to sigma.
"""
function derivative_integral_gaussian_2d_sigma(
        i::Int, j::Int, x::T, y::T, S::T, N::T, PSFx::T, PSFy::T
    ) where {T <: Real}
    dSx = derivative_integral_gaussian_1d_sigma(j, x, S, N, PSFy)
    dSy = derivative_integral_gaussian_1d_sigma(i, y, S, N, PSFx)
    return dSx + dSy
end


"""
    compute_alpha(z::T, Ax::T, Bx::T, d::T) where T <: Real

Compute the alpha value based on the given parameters.
"""
function compute_alpha(z::T, Ax::T, Bx::T, d::T) where {T <: Real}
    one_val = one(T)
    two_val = one_val + one_val
    three_val = two_val + one_val
    four_val = three_val + one_val

    z_d_ratio = z / d
    return one_val + z_d_ratio^two_val + Ax * z_d_ratio^three_val + Bx * z_d_ratio^four_val
end
