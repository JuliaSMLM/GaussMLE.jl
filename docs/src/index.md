# GaussMLE.jl

Maximum Likelihood Estimation for Gaussian blob parameters in 2D images under Poisson noise.

## Installation

```julia
using Pkg
Pkg.add("GaussMLE")
```

## Quick Start

```@example
using GaussMLE

# Generate synthetic data
boxsz = 7
nboxes = 100
out, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynb; poissonnoise=true)

# Fit the data
θ_found, Σ_found = GaussMLE.GaussFit.fitstack(out, :xynb, args)

# Display first result
println("First fit result: x=$(θ_found[1].x), y=$(θ_found[1].y)")
```

## Overview

GaussMLE.jl implements Maximum Likelihood Estimation for Gaussian blob parameters in 2D images under Poisson noise. The package is designed for Single Molecule Localization Microscopy (SMLM) applications where thousands of small image regions need to be fitted efficiently.

### Key Features

- **Fast fitting**: Optimized Newton-Raphson solver for small image regions
- **Multiple models**: Support for different Gaussian models (fixed/variable PSF width)
- **GPU acceleration**: CUDA support for large datasets (when available)
- **sCMOS support**: Variance-weighted fitting for scientific CMOS cameras
- **Uncertainty estimation**: Cramér-Rao Lower Bound uncertainties

### Supported Models

| Model | Parameters | Description |
|-------|------------|-------------|
| `GaussXyNb` | x, y, intensity, background | Fixed PSF width |
| `GaussXyNbS` | x, y, intensity, background, PSF width | Variable PSF width |

## Mathematical Foundation

The Gaussian expectation model is:

$$\mu(\theta)_{i,j} = \theta_b + \theta_n \int_{i-0.5}^{i+0.5} \int_{j-0.5}^{j+0.5} 
\mathcal{N}(\theta_x - x, \sigma^2)
\mathcal{N}(\theta_y - y, \sigma^2) \, dx \, dy$$

Where:
- $\theta_n$: Integrated intensity (photons)
- $\theta_x, \theta_y$: Blob center position (pixels)
- $\theta_b$: Background counts per pixel
- $\sigma$: PSF width (pixels)
- $i, j$: Pixel coordinates

The fitting process uses Newton-Raphson optimization to find maximum likelihood parameter estimates under Poisson noise assumptions.



