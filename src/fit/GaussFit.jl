"""
# GaussFit

The `GaussFit` module provides functions for fitting Gaussian models to a 2D box or a stack of 2D "boxes" of data.

The `fitstack` function is the primary exported function of *GaussMLE.jl*. 
It takes in a 3D array containing the data to be fitted, 
a symbol indicating the type of Gaussian model to fit, 
and additional model-specific arguments required for fitting. 
It returns an array of fitted parameters for each 2D box and 
an array of uncertainties for each set of fitted parameters.

The `fitstack` function also accepts optional keyword arguments for specifying 
a camera variance image and the top left corners of the boxes in relation to the variance image.
These are used to implement a variance-weighted fit as described by Huang et al:

- Huang, F., Hartwich, T., Rivera-Molina, F. et al. "Video-rate nanoscopy using sCMOS camera–specific single-molecule localization algorithms." Nat Methods 10, 653–658 (2013). [DOI: 10.1038/nmeth.2488](https://doi.org/10.1038/nmeth.2488)


# Exported Functions
- `fitstack`: Fit Gaussian models to each box in a stack of 2D "boxes" of data.

# Notes
- Data and Variance image must be in units of photons (Poisson distributed).
"""
module GaussFit

using LinearAlgebra
using GaussMLE
using GaussMLE.GaussModel
using GaussMLE.GaussLib

import GaussMLE.MAXPARAMS

include("crlb.jl")
include("newtonraphson.jl")
include("fitstack.jl")

export fitstack

end