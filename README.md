# GaussMLE.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/GaussMLE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/GaussMLE.jl/dev/)
[![Build Status](https://github.com/JuliaSMLM/GaussMLE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSMLM/GaussMLE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSMLM/GaussMLE.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/GaussMLE.jl)

Fast Maximum Likelihood Estimation of Gaussian PSF parameters for single-molecule localization microscopy. Automatic GPU acceleration with CPU fallback.

## Features

- **Multiple PSF Models**: 2D Gaussian (fixed/variable σ), 3D astigmatic
- **Physical Units**: All PSF parameters in microns (camera-independent)
- **Automatic GPU Acceleration**: CUDA GPU support with automatic CPU fallback
- **Camera Models**: Ideal (Poisson) and sCMOS (pixel-dependent noise)
- **CRLB Uncertainties**: Cramér-Rao lower bound for each parameter
- **SMLMData Integration**: Works with SMLMData.jl camera types and ROI structures
- **Minimal API**: Only 11 exports - clean and focused interface

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
smld, info = fit(data, fitter)  # Returns (BasicSMLD, FitInfo)

# Access results (ecosystem-standard format)
println("Fitted $(info.n_fits) localizations in $(info.elapsed_ns / 1e6) ms on $(info.backend)")
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
smld, info = fit(data, fitter)

# Extract PSF widths from Emitter2DFitSigma
σ_values = [e.σ for e in smld.emitters]  # Microns
σ_uncertainties = [e.σ_σ for e in smld.emitters]
println("Mean PSF width: $(mean(σ_values)*1000) nm ± $(mean(σ_uncertainties)*1000) nm")
```

### GPU Acceleration

```julia
using GaussMLE

# Auto-detect backend (uses GPU if available)
fitter = GaussMLEFitter()

# Force GPU with custom timeout
fitter = GaussMLEFitter(backend=:gpu, batch_size=5000)

# Auto-detect with fallback timeout (waits 30s for GPU, then falls back to CPU)
fitter = GaussMLEFitter(backend=:auto, auto_timeout=30.0)

smld, info = fit(large_dataset, fitter)  # Returns (BasicSMLD, FitInfo)
println("Executed on $(info.backend)")  # :cpu or :gpu (never :auto)
```

### sCMOS Camera

```julia
using GaussMLE
using SMLMData

# Real camera calibration
camera = SMLMData.SCMOSCamera(...)

# Generate test data or use real ROIs
batch = generate_roi_batch(camera, GaussianXYNB(0.13f0), n_rois=1000)

# Fit - automatic ADU→electrons preprocessing
fitter = GaussMLEFitter()
smld, info = fit(batch, fitter)  # Returns (BasicSMLD, FitInfo) with camera coordinates
```

### 3D Astigmatic Localization

```julia
using GaussMLE

# Astigmatic PSF calibration (all spatial params in microns)
psf_3d = AstigmaticXYZNB{Float32}(
    0.13f0, 0.13f0,   # σx₀, σy₀ (μm)
    0.05f0, -0.05f0,  # Ax, Ay
    0.01f0, -0.01f0,  # Bx, By
    0.2f0,            # γ (μm)
    0.5f0             # d (μm)
)

fitter = GaussMLEFitter(psf_model=psf_3d)
smld, info = fit(data, fitter)  # Returns (BasicSMLD, FitInfo)

# Access 3D positions
z_positions = [e.z for e in smld.emitters]  # Microns
z_precisions = [e.σ_z for e in smld.emitters]
```

## Exported API (12 Functions/Types)

### Core Functions
- `fit(data, fitter)` → **Returns (SMLMData.BasicSMLD, FitInfo)** tuple
- `fit(batch; model=..., max_iterations=...)` → Convenience form with kwargs
- `generate_roi_batch(camera, psf; kwargs...)` - Generate synthetic data

### Main Types
- `GaussMLEFitter(; psf_model, backend, iterations, constraints, batch_size, auto_timeout, gpu_timeout, on_wait)`
- `FitInfo` - Metadata about fit: elapsed_ns, backend, device_id, n_fits, n_converged

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

**fit() returns (SMLMData.BasicSMLD, FitInfo)** tuple with model-specific emitter types:

```julia
smld, info = fit(data, fitter)

# FitInfo contains execution metadata
println("Elapsed: $(info.elapsed_ns / 1e6) ms")
println("Backend: $(info.backend)")  # :cpu or :gpu (never :auto)
println("Device: $(info.device_id)")  # -1 for CPU, 0+ for GPU
println("Fits: $(info.n_fits)")

# All models: Access standard localization parameters
x_positions = [e.x for e in smld.emitters]  # Microns
photons = [e.photons for e in smld.emitters]
precisions = [e.σ_x for e in smld.emitters]  # Microns

# GaussianXYNBS: Access fitted PSF width (Emitter2DFitSigma)
fitter = GaussMLEFitter(psf_model=GaussianXYNBS())
smld, info = fit(data, fitter)
σ_values = [e.σ for e in smld.emitters]  # Microns
σ_errors = [e.σ_σ for e in smld.emitters]  # CRLB uncertainties

# GaussianXYNBSXSY: Access anisotropic PSF widths (Emitter2DFitSigmaXY)
fitter = GaussMLEFitter(psf_model=GaussianXYNBSXSY())
smld, info = fit(data, fitter)
σx_values = [e.σx for e in smld.emitters]  # Microns
σy_values = [e.σy for e in smld.emitters]  # Microns

# AstigmaticXYZNB: Access 3D positions (Emitter3DFit)
fitter = GaussMLEFitter(psf_model=AstigmaticXYZNB{Float32}(...))
smld, info = fit(data, fitter)
z_positions = [e.z for e in smld.emitters]  # Microns
z_errors = [e.σ_z for e in smld.emitters]  # CRLB uncertainties
```

**All emitter types subtype `SMLMData.AbstractEmitter`** for full ecosystem compatibility.

### Advanced Features (Qualified Access)

Internal functions use `GaussMLE.` prefix:
```julia
# Custom constraints
constraints = GaussMLE.ParameterConstraints{4}(lower, upper, max_step)

# Direct device types (prefer backend symbols :cpu/:gpu/:auto)
device = GaussMLE.GPU()

# GPU wait callback for progress feedback
fitter = GaussMLEFitter(
    backend = :gpu,
    gpu_timeout = 60.0,
    on_wait = (elapsed, available, required) ->
        @info "Waiting for GPU memory..." elapsed available required
)
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
