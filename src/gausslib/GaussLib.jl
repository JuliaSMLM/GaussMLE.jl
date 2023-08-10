module GaussLib

using LinearAlgebra
using SpecialFunctions

import GaussMLE.MAXPARAMS

export integral_gaussian_1d,
       compute_alpha,
       derivative_alpha_z,
       second_derivative_alpha_z,
       derivative_integral_gaussian_1d,
       center_of_mass_2d,
       gaussian_max_min_2d,
       derivative_integral_gaussian_1d_sigma,
       derivative_integral_gaussian_2d_sigma,
       derivative_integral_gaussian_2d_z,
       matrix_inverse!

       

include("baselibrary.jl")


end