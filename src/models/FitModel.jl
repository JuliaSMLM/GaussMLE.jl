module FitModel

using GaussMLE.GaussLib

export model, gradient, curvature, genargs, genparams, initialize_parameters!
export GaussMLEParams, GaussMLEArgs, GaussMLEΣ
export θ_xynb, Args_xynb, Σ_xynb


include("types.jl")
include("gauss_xynb.jl")


end
