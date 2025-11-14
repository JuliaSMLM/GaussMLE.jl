# GaussMLE

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/GaussMLE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/GaussMLE.jl/dev/)
[![Build Status](https://github.com/JuliaSMLM/GaussMLE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSMLM/GaussMLE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSMLM/GaussMLE.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/GaussMLE.jl)

> **GPU Support:** Automatic GPU acceleration with CPU fallback. GPU tests run automatically when a CUDA GPU is detected.

## Overview

**GaussMLE.jl** is a Julia package for performing Maximum Likelihood Estimation (MLE) of Gaussian blob parameters in 2D images under a Poisson noise model. 

The package provides a modern, flexible API through the `GaussMLEFitter` type, which supports multiple PSF models, camera noise models, and automatic GPU acceleration. Parameter uncertainties are calculated using the Cramér-Rao Lower Bound.

### Features

- Fast MLE of Gaussian blob parameters with multiple PSF models
- Automatic GPU acceleration with CPU fallback
- Support for ideal and sCMOS camera noise models
- Flexible parameter constraints
- Type-stable, high-performance implementation using KernelAbstractions.jl

## Installation

GaussMLE.jl depends on SMLMData.jl, which is not yet registered in the Julia General registry. You need to install SMLMData.jl first:

```julia
using Pkg

# Install SMLMData.jl from GitHub
Pkg.add(url="https://github.com/JuliaSMLM/SMLMData.jl")

# Then install GaussMLE.jl
Pkg.add("GaussMLE")
```

**Note:** Once GaussMLE.jl is registered in the Julia General registry, you can install with:
```julia
using Pkg
Pkg.add("GaussMLE")
```

For development or to use the latest features:
```julia
using Pkg
Pkg.add(url="https://github.com/JuliaSMLM/SMLMData.jl")
Pkg.add(url="https://github.com/JuliaSMLM/GaussMLE.jl")
```

## Basic Usage

Here's a basic example demonstrating how to fit Gaussian blobs in a stack of 2D image boxes:

```julia
using GaussMLE
using Random
using Distributions

# Generate sample data (7x7 boxes, 1000 samples)
function generate_sample_data(n_samples=1000, roi_size=7)
    data = zeros(Float32, roi_size, roi_size, n_samples)
    true_params = Float32[4.0, 4.0, 1000.0, 10.0]  # x, y, photons, background
    
    for k in 1:n_samples
        # Add some position variation
        x = true_params[1] + 0.1f0 * randn(Float32)
        y = true_params[2] + 0.1f0 * randn(Float32)
        
        for j in 1:roi_size, i in 1:roi_size
            dx = Float32(i) - x
            dy = Float32(j) - y
            gaussian = true_params[3] * exp(-(dx^2 + dy^2)/(2*1.3f0^2))
            expected = true_params[4] + gaussian / (2π * 1.3f0^2)
            data[i, j, k] = rand(Poisson(expected))
        end
    end
    return data
end

# Generate data
data = generate_sample_data()

# Create fitter with automatic GPU detection
fitter = GaussMLEFitter(
    psf_model = GaussianXYNB(1.3f0),  # Fixed sigma Gaussian
    iterations = 20
)

# Perform fitting
results = fit(fitter, data)

# Access results
println("Found $(results.n_fits) localizations")
println("Mean x position: $(mean(results.x)) ± $(mean(results.x_error))")
println("Mean photons: $(mean(results.photons)) ± $(mean(results.photons_error))")
```

### Advanced Usage

```julia
# Use sCMOS camera from SMLMData with per-pixel calibration
using SMLMData

# Create sCMOS camera with realistic parameters
camera = SMLMData.SCMOSCamera(
    256, 256, 0.1,      # 256×256 pixels, 100nm pixel size
    5.0f0,              # 5 e⁻ rms readout noise
    offset = 100.0f0,   # 100 ADU dark level
    gain = 0.5f0,       # 0.5 e⁻/ADU conversion gain
    qe = 0.82f0         # 82% quantum efficiency
)

# Variable sigma model with custom constraints
psf_model = GaussianXYNBS()  # 5 parameters: x, y, N, bg, σ
constraints = ParameterConstraints{5}(
    SVector{5,Float32}(0.5, 0.5, 10.0, 0.0, 0.5),      # lower bounds
    SVector{5,Float32}(7.5, 7.5, 1e5, 100.0, 3.0),    # upper bounds
    SVector{5,Float32}(1.0, 1.0, Inf32, Inf32, 0.5)   # max step
)

# Generate simulated data or use ROIBatch from SMLMData
batch = generate_roi_batch(camera, psf_model; n_rois=100, roi_size=11)

# Create fitter with specific configuration
fitter = GaussMLEFitter(
    psf_model = psf_model,
    camera_model = camera,
    device = :gpu,  # Force GPU (will fallback to CPU if unavailable)
    constraints = constraints,
    iterations = 30,
    batch_size = 5000  # Process in batches of 5000
)

# Fit - preprocessing (ADU→electrons) happens automatically
results = fit(fitter, batch)
```

For more details and advanced usage, check out the [docs](https://JuliaSMLM.github.io/GaussMLE.jl/dev/).

## References

This package implements the algorithm described in:

- Smith, C., Joseph, N., Rieger, B. et al. "Fast, single-molecule localization that achieves theoretically minimum uncertainty." Nat Methods 7, 373–375 (2010). [DOI: 10.1038/nmeth.1449](https://doi.org/10.1038/nmeth.1449)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

