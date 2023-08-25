module GaussFit

using LinearAlgebra
using GaussMLE
using GaussMLE.GaussModel
using GaussMLE.GaussLib

import GaussMLE.MAXPARAMS

include("crlb.jl")
include("newtonraphson.jl")
include("fitstack.jl")

end