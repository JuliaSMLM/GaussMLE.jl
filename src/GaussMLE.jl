module GaussMLE

export fitstack

include("constants.jl")
include("gausslib/GaussLib.jl")
include("models/GaussModel.jl")
include("sim/GaussSim.jl")
include("fit/GaussFit.jl")

using .GaussFit



end
