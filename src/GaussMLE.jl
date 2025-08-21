module GaussMLE

using KernelAbstractions
using CUDA
using StaticArrays
using LinearAlgebra
using Statistics
using SpecialFunctions

# Constants
include("constants.jl")

# Original modules needed for GaussLib
include("gausslib/GaussLib.jl")
using .GaussLib

# Core modules for refactored API
include("devices.jl")
include("camera_models.jl")
include("psf_models.jl")
include("psf_derivatives.jl")
include("constraints.jl")
include("kernels.jl")
include("unified_kernel.jl")  # New unified GPU/CPU kernel
include("results.jl")
include("simulator.jl")
include("api.jl")

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

# Simulator exports
export SimulatedBlob, generate_blobs_data, generate_random_blobs

end # module