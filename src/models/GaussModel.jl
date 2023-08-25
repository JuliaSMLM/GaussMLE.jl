module GaussModel

using GaussMLE.GaussLib

#
# Each model must define these methods for thier type of model
export model, gradient!, curvature!, genargs, genθ, genΣ, initialize_parameters!, update!

# Each model must define a type inheriting from these abstract types and export
export GaussMLEParams, GaussMLEArgs, GaussMLEΣ

# Exports for the xynb model 
export θ_xynb, Args_xynb, Σ_xynb
export θ_xynbs, Args_xynbs, Σ_xynbs

# Used internally to map a symbol to a model type.
export MODEL_MAP

include("types.jl")
include("gauss_xynb.jl")
include("gauss_xynbs.jl")

# Mapping from a symbol to the new model type. Add new models here
const MODEL_MAP = Dict(
    :xynb => θ_xynb,
    :xynbs => θ_xynbs
)



end
