# GaussMLE.jl

Fast Maximum Likelihood Estimation of Gaussian PSF parameters for single-molecule localization microscopy. Automatic GPU acceleration with CPU fallback.

## Installation

```julia
using Pkg

# Install SMLMData.jl dependency first
Pkg.add(url="https://github.com/JuliaSMLM/SMLMData.jl")

# Then install GaussMLE.jl
Pkg.add(url="https://github.com/JuliaSMLM/GaussMLE.jl")
```

Once registered in Julia General:
```julia
using Pkg
Pkg.add("GaussMLE")
```

## Quick Start

```julia
using GaussMLE

# Your data: (roi_size, roi_size, n_rois)
data = rand(Float32, 11, 11, 100)

# Fit with defaults (fixed σ Gaussian, auto GPU/CPU)
fitter = GaussMLEFitter()
results = fit(fitter, data)

# Access results
println("Fitted $(results.n_fits) localizations")
println("Mean position: ($(mean(results.x)), $(mean(results.y)))")
println("Mean precision: $(mean(results.x_error)) pixels")
```

## Overview

**GaussMLE.jl** provides a modern, minimal API for fitting Gaussian PSF models to single-molecule localization microscopy data. The package supports multiple PSF models, camera noise models, and automatic GPU acceleration.

### Key Features

- **Multiple PSF Models**: 2D Gaussian (fixed/variable σ), 3D astigmatic
- **Automatic GPU Acceleration**: CUDA support with automatic CPU fallback
- **Camera Models**: Ideal (Poisson) and sCMOS (pixel-dependent noise)
- **CRLB Uncertainties**: Cramér-Rao lower bound for each parameter
- **SMLMData Integration**: Works seamlessly with SMLMData.jl types
- **Minimal API**: Only 7 exports - clean and focused interface

### Minimal Export Philosophy

GaussMLE.jl exports only what you need for common workflows:

**Exported (7):**
- `fit` - Main fitting function
- `GaussMLEFitter` - Fitter constructor
- `GaussianXYNB`, `GaussianXYNBS`, `GaussianXYNBSXSY`, `AstigmaticXYZNB` - PSF models
- `generate_roi_batch` - Synthetic data generation

**Qualified Access (use `GaussMLE.` prefix):**
- `GaussMLE.ParameterConstraints` - Custom constraints
- `GaussMLE.to_smld` - Result conversion
- `GaussMLE.GPU`, `GaussMLE.CPU` - Direct device types (prefer symbols `:gpu`, `:cpu`)

### Supported PSF Models

| Model | Parameters | Use Case |
|-------|------------|----------|
| `GaussianXYNB(σ)` | x, y, N, bg | Fixed PSF width (fastest) |
| `GaussianXYNBS(σ₀)` | x, y, N, bg, σ | Variable PSF width |
| `GaussianXYNBSXSY(σx₀, σy₀)` | x, y, N, bg, σx, σy | Asymmetric PSF |
| `AstigmaticXYZNB{T}(...)` | x, y, z, N, bg | 3D astigmatic imaging |

## Examples

### Variable PSF Width

```julia
using GaussMLE

# Fit PSF width per localization
fitter = GaussMLEFitter(psf_model=GaussianXYNBS())
results = fit(fitter, data)

println("Mean σ: $(mean(results.sigma)) pixels")
```

### GPU Acceleration

```julia
using GaussMLE

# Force GPU (auto-fallback if unavailable)
fitter = GaussMLEFitter(device=:gpu, batch_size=5000)
results = fit(fitter, large_dataset)
```

### sCMOS Camera

```julia
using GaussMLE
using SMLMData

# Realistic camera calibration
camera = SMLMData.SCMOSCamera(
    2048, 2048, 0.065,  # 2048×2048, 65nm pixels
    1.6f0,              # 1.6 e⁻ readnoise
    offset = 100.0f0,
    gain = 0.46f0
)

# Generate or load ROI data
batch = generate_roi_batch(camera, GaussianXYNB(1.3f0), n_rois=1000)

# Fit with automatic preprocessing
fitter = GaussMLEFitter(camera_model=camera)
results = fit(fitter, batch)
```

### 3D Astigmatic Localization

```julia
using GaussMLE

# Astigmatic PSF calibration
psf_3d = AstigmaticXYZNB{Float32}(
    1.3f0, 1.3f0,    # σx₀, σy₀
    0.05f0, 0.05f0,  # Ax, Ay
    0.3f0, 0.3f0,    # Bx, By
    50.0f0,          # γ
    100.0f0          # d
)

fitter = GaussMLEFitter(psf_model=psf_3d)
results = fit(fitter, data)

# Z positions
z_positions = results.z
z_precision = results.z_error
```

## Mathematical Foundation

The Gaussian expectation model for pixel (i,j) is:

```math
μ_{i,j}(θ) = θ_{bg} + θ_N ∫_{i-0.5}^{i+0.5} ∫_{j-0.5}^{j+0.5}
\frac{1}{2π σ_x σ_y} \exp\left(-\frac{(x-θ_x)^2}{2σ_x^2} - \frac{(y-θ_y)^2}{2σ_y^2}\right) dx \, dy
```

Where:
- θ_x, θ_y: Emitter position (pixels)
- θ_N: Integrated photon count
- θ_{bg}: Background per pixel
- σ_x, σ_y: PSF widths (pixels)

The fitting uses Newton-Raphson optimization with:
- **Diagonal Hessian**: For fast parameter updates
- **Full Fisher Information**: For accurate CRLB uncertainties

## Performance

Typical performance on modern hardware:

- **CPU**: ~100K fits/second (AMD Ryzen 9 5950X, 11×11 ROIs)
- **GPU**: ~10M fits/second (NVIDIA RTX 4090, batch size 50K)

Scales automatically with batch size and available memory.

## Algorithm Reference

Implements the MLE algorithm from:

> Smith, C., Joseph, N., Rieger, B. et al. "Fast, single-molecule localization that achieves theoretically minimum uncertainty." *Nat Methods* **7**, 373–375 (2010). [DOI: 10.1038/nmeth.1449](https://doi.org/10.1038/nmeth.1449)

## Related Packages

- **[SMLMData.jl](https://github.com/JuliaSMLM/SMLMData.jl)** - Core SMLM data types
- **[SMLMSim.jl](https://github.com/JuliaSMLM/SMLMSim.jl)** - SMLM simulation
- **[MicroscopePSFs.jl](https://github.com/JuliaSMLM/MicroscopePSFs.jl)** - PSF models
