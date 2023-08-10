module GaussMLE

using CUDA

include("constants.jl")
include("gausslib/GaussLib.jl")
include("models/FitModel.jl")
include("sim/SimGauss.jl")


end
