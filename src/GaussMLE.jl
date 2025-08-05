module GaussMLE

include("constants.jl")
include("gausslib/GaussLib.jl")
include("models/GaussModel.jl")
include("sim/GaussSim.jl")
include("fit/GaussFit.jl")
include("gpu/GaussGPU.jl")

using .GaussFit: fitstack
using .GaussGPU
using .GaussModel: θ_xynb, θ_xynbs, θ_xynbsxsy, θ_xynbz, AstigmaticCalibration, GaussMLEParams, GaussMLEArgs, GaussMLEΣ,
                   genargs, genθ, genΣ, model, gradient!, curvature!, initialize_parameters!, update!, compute_all!

# Export the unified fitstack function
export fitstack

# Export model types and abstract types
export θ_xynb, θ_xynbs, θ_xynbsxsy, θ_xynbz, AstigmaticCalibration, GaussMLEParams, GaussMLEArgs, GaussMLEΣ

# Export model functions
export genargs, genθ, genΣ, model, gradient!, curvature!, initialize_parameters!, update!, compute_all!

# Re-export backend types for advanced users (fitstack_gpu is now internal)
export select_backend, FittingBackend,
       CPUBackend, CUDABackend, MetalBackend, BatchConfig

end
