"""
    GaussLib

Low level functions for calculating the Gaussian model and its first derivatives.
"""
module GaussLib

using SpecialFunctions: SpecialFunctions, erf

include("baselibrary.jl")

export integral_gaussian_1d,
    compute_alpha,
    derivative_integral_gaussian_1d,
    derivative_integral_gaussian_1d_sigma,
    derivative_integral_gaussian_2d_sigma
end
