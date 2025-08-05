module GaussMLE

using KernelAbstractions
using CUDA
using StaticArrays
using LinearAlgebra
using Statistics
using SpecialFunctions

# Core modules
include("devices.jl")
include("camera_models.jl")
include("psf_models.jl")
include("constraints.jl")
include("kernels.jl")
include("results.jl")
include("api.jl")

# Legacy compatibility layer (can be removed later)
include("legacy_compat.jl")

# Main exports

# Device management
export ComputeDevice, CPU, GPU
export auto_device, select_device

# Camera models
export CameraModel, IdealCamera, SCMOSCamera

# PSF models
export PSFModel, GaussianXYNB, GaussianXYNBS, GaussianXYNBSXSY, AstigmaticXYZNB

# Main API
export GaussMLEFitter, fit, GaussMLEResults

# Convenience exports
export ParameterConstraints, default_constraints

end # module