# GaussMLE.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/GaussMLE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/GaussMLE.jl/dev/)
[![Build Status](https://github.com/JuliaSMLM/GaussMLE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSMLM/GaussMLE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSMLM/GaussMLE.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/GaussMLE.jl)

Fast Maximum Likelihood Estimation of Gaussian PSF parameters for single-molecule localization microscopy. Automatic GPU acceleration with CPU fallback.

## Features

- **Multiple PSF Models**: 2D Gaussian (fixed/variable σ), 3D astigmatic
- **Automatic GPU Acceleration**: CUDA GPU support with automatic CPU fallback
- **Camera Models**: Ideal (Poisson) and sCMOS (pixel-dependent noise)
- **CRLB Uncertainties**: Cramér-Rao lower bound for each parameter
- **SMLMData Integration**: Works with SMLMData.jl camera types and ROI structures
- **Minimal API**: Only 7 exports - clean and focused interface

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
Pkg.add("GaussMLE")
```

## Quick Start

### Basic Fitting

```julia
using GaussMLE
using SMLMData

# Your data: (roi_size, roi_size, n_rois)
data = rand(Float32, 11, 11, 100)

# Fit with defaults (fixed σ Gaussian, auto GPU/CPU)
fitter = GaussMLEFitter()
smld = fit(fitter, data)  # Returns BasicSMLD

# Access results (ecosystem-standard format)
println("Fitted $(length(smld.emitters)) localizations")
x_positions = [e.x for e in smld.emitters]
precisions = [e.σ_x for e in smld.emitters]
println("Mean position: $(mean(x_positions)) μm")
println("Mean precision: $(mean(precisions)*1000) nm")
```

### Variable PSF Width

```julia
using GaussMLE

# Fit PSF width per localization
fitter = GaussMLEFitter(psf_model=GaussianXYNBS())
smld = fit(fitter, data)

# Extract PSF widths from Emitter2DFitSigma
σ_values = [e.σ for e in smld.emitters]  # Microns
σ_uncertainties = [e.σ_σ for e in smld.emitters]
println("Mean PSF width: $(mean(σ_values)*1000) nm ± $(mean(σ_uncertainties)*1000) nm")
```

### GPU Acceleration

```julia
using GaussMLE

# Force GPU (auto-fallback to CPU if unavailable)
fitter = GaussMLEFitter(device=:gpu, batch_size=5000)
smld = fit(fitter, large_dataset)  # Returns BasicSMLD
```

### sCMOS Camera

```julia
using GaussMLE
using SMLMData

# Real camera calibration
camera = SMLMData.SCMOSCamera(...)

# Generate test data or use real ROIs
batch = generate_roi_batch(camera, GaussianXYNB(1.3f0), n_rois=1000)

# Fit - automatic ADU→electrons preprocessing
fitter = GaussMLEFitter(camera_model=camera)
smld = fit(fitter, batch)  # Returns BasicSMLD with camera coordinates
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
smld = fit(fitter, data)  # Returns BasicSMLD

# Access 3D positions
x_positions = [e.x for e in smld.emitters]
precisions = [e.σ_x for e in smld.emitters]
```

## Exported API (11 Functions/Types)

### Core Functions
- `fit(fitter, data)` → **Returns SMLMData.BasicSMLD** (ecosystem standard!)
- `generate_roi_batch(camera, psf; kwargs...)` - Generate synthetic data

### Main Type
- `GaussMLEFitter(; psf_model, camera_model, device, iterations, constraints, batch_size)`

### PSF Models
- `GaussianXYNB(σ)` - Fixed σ (4 params: x, y, N, bg)
- `GaussianXYNBS(σ₀)` - Variable σ (5 params: x, y, N, bg, σ)
- `GaussianXYNBSXSY(σx₀, σy₀)` - Independent σx, σy (6 params: x, y, N, bg, σx, σy)
- `AstigmaticXYZNB{T}(...)` - 3D astigmatic (5 params: x, y, z, N, bg)

### Custom Emitter Types
- `Emitter2DFitSigma{T}` - 2D emitter with fitted σ (for GaussianXYNBS)
- `Emitter2DFitSigmaXY{T}` - 2D emitter with fitted σx, σy (for GaussianXYNBSXSY)

### SMLMData Types (Re-exported)
- `ROIBatch` - Batch of ROIs with camera context
- `SingleROI` - Individual ROI

### Output Format

**fit() returns SMLMData.BasicSMLD** with model-specific emitter types:

```julia
smld = fit(fitter, data)

# All models: Access standard localization parameters
x_positions = [e.x for e in smld.emitters]  # Microns
photons = [e.photons for e in smld.emitters]
precisions = [e.σ_x for e in smld.emitters]  # Microns

# GaussianXYNBS: Access fitted PSF width (Emitter2DFitSigma)
fitter = GaussMLEFitter(psf_model=GaussianXYNBS())
smld = fit(fitter, data)
σ_values = [e.σ for e in smld.emitters]  # Microns
σ_errors = [e.σ_σ for e in smld.emitters]  # CRLB uncertainties

# GaussianXYNBSXSY: Access anisotropic PSF widths (Emitter2DFitSigmaXY)
fitter = GaussMLEFitter(psf_model=GaussianXYNBSXSY())
smld = fit(fitter, data)
σx_values = [e.σx for e in smld.emitters]  # Microns
σy_values = [e.σy for e in smld.emitters]  # Microns

# AstigmaticXYZNB: Access 3D positions (Emitter3DFit)
fitter = GaussMLEFitter(psf_model=AstigmaticXYZNB{Float32}(...))
smld = fit(fitter, data)
z_positions = [e.z for e in smld.emitters]  # Microns
z_errors = [e.σ_z for e in smld.emitters]  # CRLB uncertainties
```

**All emitter types subtype `SMLMData.AbstractEmitter`** for full ecosystem compatibility.

### Advanced Features (Qualified Access)

Internal functions use `GaussMLE.` prefix:
```julia
# Custom constraints
constraints = GaussMLE.ParameterConstraints{4}(lower, upper, max_step)

# Direct device types (prefer symbols :cpu/:gpu)
device = GaussMLE.GPU()
```

## Examples

See the `examples/` directory for complete working examples:
- `basic_fitting.jl` - Simple fitting workflow
- `scmos_camera.jl` - sCMOS noise model
- `gpu_acceleration.jl` - GPU batch processing
- `astigmatic_3d.jl` - 3D localization

Run examples:
```bash
julia --project=examples examples/basic_fitting.jl
```

## Performance

- CPU: ~100K fits/second (typical workstation, 11×11 ROIs)
- GPU: ~10M fits/second (NVIDIA RTX 4090, batch size 50K)
- Scales automatically with batch size and available memory

## Documentation

Full documentation available at: https://JuliaSMLM.github.io/GaussMLE.jl/dev/

Topics covered:
- API reference with all PSF models
- Coordinate systems and conventions
- Custom constraints and initialization
- Integration with SMLMData.jl ecosystem
- Performance optimization guide

## Algorithm Reference

Implements the MLE algorithm from:

> Smith, C., Joseph, N., Rieger, B. et al. "Fast, single-molecule localization that achieves theoretically minimum uncertainty." *Nat Methods* **7**, 373–375 (2010). [DOI: 10.1038/nmeth.1449](https://doi.org/10.1038/nmeth.1449)

## Related Packages

- **[SMLMData.jl](https://github.com/JuliaSMLM/SMLMData.jl)** - Core data types for SMLM
- **[SMLMSim.jl](https://github.com/JuliaSMLM/SMLMSim.jl)** - SMLM data simulation
- **[MicroscopePSFs.jl](https://github.com/JuliaSMLM/MicroscopePSFs.jl)** - PSF models

## License

MIT License - see [LICENSE](LICENSE) file for details.
