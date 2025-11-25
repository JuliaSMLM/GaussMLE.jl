# GaussMLE.jl

Fast Maximum Likelihood Estimation of Gaussian PSF parameters for single-molecule localization microscopy. Automatic GPU acceleration with CPU fallback.

## Installation

```julia
using Pkg
Pkg.add("GaussMLE")
```

## Quick Start

```julia
using GaussMLE
using Statistics

# ROIBatch typically comes from SMLMBoxer.jl which extracts ROIs from raw data
# For testing/development, use generate_roi_batch() or raw arrays:
data = rand(Float32, 11, 11, 100)  # (roi_size, roi_size, n_rois)

# Create fitter with PSF model (sigma must match your microscope PSF)
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))  # 130nm PSF width

# Fit data - returns BasicSMLD with emitters
smld = fit(fitter, data)

# Access results
println("Fitted $(length(smld.emitters)) localizations")
for e in smld.emitters[1:3]
    println("Position: ($(e.x), $(e.y)) μm, Photons: $(e.photons)")
end
```

!!! note "Re-exports from SMLMData"
    GaussMLE re-exports commonly needed types (ROIBatch, camera types) from SMLMData.jl,
    so you typically only need `using GaussMLE`.

## Overview

**GaussMLE.jl** provides a modern, minimal API for fitting Gaussian PSF models to single-molecule localization microscopy data. The package supports multiple PSF models, camera noise models, and automatic GPU acceleration.

### Key Features

- **Multiple PSF Models**: 2D Gaussian (fixed/variable sigma), 3D astigmatic
- **Physical Units**: All PSF parameters in microns (camera-independent)
- **Automatic GPU Acceleration**: CUDA support with automatic CPU fallback
- **Camera Models**: Ideal (Poisson) and sCMOS (pixel-dependent noise)
- **CRLB Uncertainties**: Cramer-Rao lower bound for each parameter
- **SMLMData Integration**: Returns `BasicSMLD` with ecosystem-compatible emitter types

### Unit Convention

**All user-facing parameters use physical units (microns)**:
- PSF widths: specified in microns (e.g., `GaussianXYNB(0.13)` for 130nm PSF)
- Output positions: microns
- Output uncertainties: microns
- Internally converted to pixels for computation based on camera pixel size

### Supported PSF Models

| Model | Parameters | Use Case |
|-------|------------|----------|
| `GaussianXYNB(sigma)` | x, y, N, bg | Fixed PSF width (fastest) |
| `GaussianXYNBS()` | x, y, N, bg, sigma | Variable PSF width |
| `GaussianXYNBSXSY()` | x, y, N, bg, sigma_x, sigma_y | Anisotropic PSF |
| `AstigmaticXYZNB{T}(...)` | x, y, z, N, bg | 3D astigmatic imaging |

## Typical SMLM Pipeline

In a real workflow, ROIs come from a boxer that detects candidates in raw movie frames:

```
Raw Movie → SMLMBoxer.jl → ROIBatch → GaussMLE.fit() → BasicSMLD → Analysis
```

## Examples

### Variable PSF Width

```julia
using GaussMLE
using Statistics

# Fit PSF width per localization
fitter = GaussMLEFitter(psf_model = GaussianXYNBS())
smld = fit(fitter, data)

# Access fitted sigma from Emitter2DFitSigma type
sigmas = [e.sigma for e in smld.emitters]
println("Mean sigma: $(mean(sigmas)) microns")
```

### GPU Acceleration

```julia
using GaussMLE

# Force GPU (auto-fallback if unavailable)
fitter = GaussMLEFitter(device = :gpu, batch_size = 5000)
smld = fit(fitter, large_dataset)
```

### sCMOS Camera

```julia
using GaussMLE

# Create sCMOS camera with calibration maps
camera = SCMOSCamera(
    offset_map,      # Per-pixel offset (ADU)
    gain_map,        # Per-pixel gain (e-/ADU)
    readnoise_map,   # Per-pixel readnoise (e-)
    0.065            # Pixel size (microns)
)

# Generate or load ROI data with camera
batch = generate_roi_batch(camera, GaussianXYNB(0.13f0), n_rois = 1000)

# Fit - automatically uses variance map from camera
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))
smld = fit(fitter, batch)
```

### 3D Astigmatic Localization

```julia
using GaussMLE

# Astigmatic PSF calibration (all spatial params in microns)
psf_3d = AstigmaticXYZNB{Float32}(
    0.13f0, 0.13f0,  # sigma_x0, sigma_y0
    0.05f0, 0.05f0,  # Ax, Ay
    0.3f0, 0.3f0,    # Bx, By
    0.05f0,          # gamma
    0.4f0            # d
)

fitter = GaussMLEFitter(psf_model = psf_3d)
smld = fit(fitter, data)

# Z positions from Emitter3DFitGaussMLE type
z_positions = [e.z for e in smld.emitters]
z_precision = [e.sigma_z for e in smld.emitters]
```

## Output Format

### BasicSMLD with Custom Emitter Types

`fit()` returns `SMLMData.BasicSMLD` with PSF-specific emitter types. All emitter types subtype `SMLMData.AbstractEmitter` for ecosystem compatibility.

| PSF Model | Emitter Type | Additional Fields |
|-----------|--------------|-------------------|
| `GaussianXYNB` | `Emitter2DFitGaussMLE` | `pvalue` |
| `GaussianXYNBS` | `Emitter2DFitSigma` | `sigma`, `sigma_sigma` |
| `GaussianXYNBSXSY` | `Emitter2DFitSigmaXY` | `sigma_x`, `sigma_y` |
| `AstigmaticXYZNB` | `Emitter3DFitGaussMLE` | `z`, `sigma_z`, `pvalue` |

### Common Emitter Fields

All emitter types include:
- `x`, `y`: Position (microns)
- `photons`, `bg`: Photometry
- `sigma_x`, `sigma_y`: Position uncertainties (CRLB, microns)
- `sigma_photons`, `sigma_bg`: Photometry uncertainties
- `frame`: Frame number
- `pvalue`: Goodness-of-fit p-value

## Mathematical Foundation

The Gaussian expectation model for pixel (i,j) is:

```math
\mu_{i,j}(\theta) = \theta_{bg} + \theta_N \int_{i-0.5}^{i+0.5} \int_{j-0.5}^{j+0.5}
\frac{1}{2\pi \sigma_x \sigma_y} \exp\left(-\frac{(x-\theta_x)^2}{2\sigma_x^2} - \frac{(y-\theta_y)^2}{2\sigma_y^2}\right) dx \, dy
```

The fitting uses Newton-Raphson optimization with:
- **Diagonal Hessian**: For fast parameter updates
- **Full Fisher Information**: For accurate CRLB uncertainties

## Performance

Typical performance on modern hardware:

- **CPU**: ~100K fits/second (AMD Ryzen 9 5950X, 11x11 ROIs)
- **GPU**: ~10M fits/second (NVIDIA RTX 4090, batch size 50K)

## Algorithm Reference

Implements the MLE algorithm from:

> Smith, C., Joseph, N., Rieger, B. et al. "Fast, single-molecule localization that achieves theoretically minimum uncertainty." *Nat Methods* **7**, 373-375 (2010). [DOI: 10.1038/nmeth.1449](https://doi.org/10.1038/nmeth.1449)

## Related Packages

- **[SMLMData.jl](https://github.com/JuliaSMLM/SMLMData.jl)** - Core SMLM data types
- **[SMLMSim.jl](https://github.com/JuliaSMLM/SMLMSim.jl)** - SMLM simulation
- **[MicroscopePSFs.jl](https://github.com/JuliaSMLM/MicroscopePSFs.jl)** - PSF models
