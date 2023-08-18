module GaussFit

using GaussMLE
using GaussMLE.GaussModel
using GaussMLE.GaussLib

import GaussMLE.MAXPARAMS

include("newtonraphson.jl")
include("fitstack.jl")

end