"""
    GaussMLE

Maximum-likelihood fitting of Gaussian PSF models to single-molecule ROIs, on the CPU or a CUDA
GPU, with Cramér-Rao lower bound uncertainties. Noise models: Poisson (`IdealCamera`) and sCMOS
(`SCMOSCamera`, per-pixel readout noise).

Entry points: `fit` with a `GaussMLEConfig` or keywords; PSF models `GaussianXYNB`,
`GaussianXYNBS`, `GaussianXYNBSXSY`, `AstigmaticXYZNB` and `SplinePSFModel` (a tabulated 3D
PSF, CPU only); `generate_roi_batch` for simulated data. Results are `SMLMData.BasicSMLD` with
model-specific emitter types.
"""
module GaussMLE

using KernelAbstractions: KernelAbstractions, @Const, @index, @kernel
using CUDA: CUDA, CUDABackend
using StaticArrays: StaticArrays, @SVector, MMatrix, MVector, SVector
using LinearAlgebra: LinearAlgebra
using Statistics: Statistics, mean
using SpecialFunctions: SpecialFunctions
using SMLMData: SMLMData
using Random: Random
using Distributions: Chisq, cdf, Poisson  # Only import what we need

import StatsAPI: fit  # Extend canonical Julia fit function

# Import commonly used types from SMLMData (ecosystem standard)
using SMLMData: ROIBatch, SingleROI, IdealCamera, SCMOSCamera, @filter,
    AbstractSMLMConfig, AbstractSMLMInfo

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
include("spline_psf.jl")  # Tabulated 3D PSF (CPU only)
include("unified_kernel.jl")  # Unified GPU/CPU kernel
include("results.jl")
include("simulator.jl")
include("interface.jl")  # User-facing API

# Main exports - minimal API for common workflows
# Camera types come from SMLMData (use SMLMData.SCMOSCamera, etc.)
# ROIBatch and SingleROI come from SMLMData (ecosystem standard)
export fit, GaussMLEConfig, GaussMLEFitInfo
export GaussianXYNB, GaussianXYNBS, GaussianXYNBSXSY, AstigmaticXYZNB, SplinePSFModel
export generate_roi_batch

# Export custom emitter types (subtype AbstractEmitter)
export Emitter2DFitGaussMLE, Emitter2DFitSigma, Emitter2DFitSigmaXY, Emitter3DFitGaussMLE

# Re-export SMLMData types for convenience
export ROIBatch, SingleROI, IdealCamera, SCMOSCamera, @filter

end # module
