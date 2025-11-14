module GaussMLE

using KernelAbstractions
using CUDA
using StaticArrays
using LinearAlgebra
using Statistics
using SpecialFunctions
using SMLMData

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
include("roi_batch.jl")  # ROI batch data structure
include("unified_kernel.jl")  # Unified GPU/CPU kernel
include("results.jl")
include("simulator.jl")
include("api.jl")

# Main exports

# Device management
export ComputeDevice, CPU, GPU
export auto_device, select_device

# Camera models (internal types)
export CameraModel, IdealCamera, SCMOSCameraInternal
export to_electrons, get_variance_map  # Preprocessing helpers for SMLMData.SCMOSCamera

# PSF models
export PSFModel, GaussianXYNB, GaussianXYNBS, GaussianXYNBSXSY, AstigmaticXYZNB

# Main API
export GaussMLEFitter, fit, GaussMLEResults

# Convenience exports
export ParameterConstraints, default_constraints

# Simulator exports
export generate_roi_batch

# ROI batch exports
export ROIBatch, SingleROI, LocalizationResult
export roi_to_camera_coords, create_localization_result
export to_emitter2dfit, to_smld

end # module