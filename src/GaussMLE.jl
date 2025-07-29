module GaussMLE

export fitstack

include("constants.jl")
include("gausslib/GaussLib.jl")
include("models/GaussModel.jl")
include("sim/GaussSim.jl")
include("fit/GaussFit.jl")
include("gpu/GaussGPU.jl")

using .GaussFit
using .GaussGPU

# Re-export GPU functions
export fitstack_gpu, select_backend, FittingBackend,
       CPUBackend, CUDABackend, MetalBackend, BatchConfig

end
