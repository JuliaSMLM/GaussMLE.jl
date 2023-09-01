# GaussMLE

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/GaussMLE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/GaussMLE.jl/dev/)
[![Build Status](https://github.com/JuliaSMLM/GaussMLE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSMLM/GaussMLE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSMLM/GaussMLE.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/GaussMLE.jl)

## Overview

**GaussMLE.jl** is a Julia package for performing Maximum Likelihood Estimation (MLE) of Gaussian blob parameters in 2D images under a Poisson noise model. 

The main functionality is provided through the `GaussFit.fitstack` function, which takes a stack of 2D image boxes and returns the estimated Gaussian parameters for each box along with the uncertainties for each parameter.  Uncertainties are calculated using the Cramér-Rao Lower Bound.

### Features

- Fast MLE of Gaussian blob parameters
- Supports different forms of Gaussian expectation models
- Easy-to-use API

## Installation

To install GaussMLE.jl, start Julia and run:

```julia
using Pkg
Pkg.add("GaussMLE")
```

## Basic Usage

A basic example that demonstrates how to use `GaussFit.fitstack` to fit Gaussian blobs in a stack of 2D image boxes:

```julia
using GaussMLE
using Statistics 

# Simulate a stack of boxes with Poisson noise
T = Float32 # Data type
boxsz = 7 # Box size
nboxes = Int(1e5) # Number of boxes
modeltype = :xynb # Fit model type 
out, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynb; T=T, poissonnoise=true)

# Fit all boxes in the stack
θ_found, Σ_found = GaussMLE.GaussFit.fitstack(out, modeltype, args);

```

For more details and advanced usage, check out the [docs](https://JuliaSMLM.github.io/GaussMLE.jl/dev/).

## References

This package implements the algorithm described in:

- Smith, C., Joseph, N., Rieger, B. et al. "Fast, single-molecule localization that achieves theoretically minimum uncertainty." Nat Methods 7, 373–375 (2010). [DOI: 10.1038/nmeth.1449](https://doi.org/10.1038/nmeth.1449)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

