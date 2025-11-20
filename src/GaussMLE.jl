module GaussMLE

using KernelAbstractions
using CUDA
using StaticArrays
using LinearAlgebra
using Statistics
using SpecialFunctions
using SMLMData
using Random
using Distributions

# Import ROIBatch and SingleROI from SMLMData (ecosystem standard)
using SMLMData: ROIBatch, SingleROI

import Adapt

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
include("emitters.jl")  # Custom emitter types with PSF parameters
include("roi_batch.jl")  # ROI batch data structure
include("unified_kernel.jl")  # Unified GPU/CPU kernel
include("results.jl")
include("simulator.jl")
include("interface.jl")  # User-facing API

# Main exports - minimal API for common workflows
# Camera types come from SMLMData (use SMLMData.SCMOSCamera, etc.)
# ROIBatch and SingleROI come from SMLMData (ecosystem standard)
export fit, GaussMLEFitter
export GaussianXYNB, GaussianXYNBS, GaussianXYNBSXSY, AstigmaticXYZNB
export generate_roi_batch

# Export custom emitter types (subtype AbstractEmitter)
export Emitter2DFitSigma, Emitter2DFitSigmaXY

# Re-export SMLMData types for convenience
export ROIBatch, SingleROI

end # module