# GaussMLE.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSMLM.github.io/GaussMLE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSMLM.github.io/GaussMLE.jl/dev/)
[![Build Status](https://github.com/JuliaSMLM/GaussMLE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSMLM/GaussMLE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSMLM/GaussMLE.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSMLM/GaussMLE.jl)

Fast Maximum Likelihood Estimation of Gaussian PSF parameters for single-molecule localization microscopy. Automatic GPU acceleration with CPU fallback.

## Installation

```julia
using Pkg
Pkg.add("GaussMLE")
```

## Quick Start

```julia
using GaussMLE

# Camera and ROI data (from your detection pipeline)
camera = IdealCamera(0:255, 0:255, 0.1)  # 100 nm pixels
batch = ROIBatch(data, x_corners, y_corners, frame_indices, camera)

# Configure and fit
fitter = GaussMLEConfig(psf_model=GaussianXYNB(0.13f0))  # σ = 130 nm
smld, info = fit(batch, fitter)

# Access results
x = [e.x for e in smld.emitters]       # positions (μm)
σ_x = [e.σ_x for e in smld.emitters]   # CRLB uncertainties (μm)
println("$(info.n_fits) fits in $(round(info.elapsed_s*1000, digits=1)) ms on $(info.backend)")
```

For complete SMLM workflows (detection + fitting + rendering), see [SMLMAnalysis.jl](https://github.com/JuliaSMLM/SMLMAnalysis.jl).

## GaussMLEConfig

All fitting is configured through `GaussMLEConfig`:

| Field | Default | Description |
|-------|---------|-------------|
| `psf_model` | `GaussianXYNB(0.13f0)` | PSF model (see below) |
| `backend` | `:auto` | Compute backend: `:cpu`, `:gpu`, or `:auto` |
| `iterations` | `20` | Newton-Raphson iterations |
| `constraints` | `nothing` | Parameter bounds/step limits (auto-generated if `nothing`) |
| `batch_size` | `10_000` | ROIs per GPU batch |
| `auto_timeout` | `300.0` | Seconds to wait for GPU before CPU fallback (`:auto` mode) |
| `gpu_timeout` | `Inf` | Seconds to wait for GPU (`:gpu` mode, errors on timeout) |
| `on_wait` | `nothing` | Callback `(elapsed, available, required) -> nothing` |

**Backend semantics:** `:cpu` runs immediately on CPU. `:gpu` waits for a GPU up to `gpu_timeout`, then errors. `:auto` tries GPU up to `auto_timeout`, then falls back to CPU with a warning.

GPU scheduling uses NVML polling to avoid OOM during device discovery in multi-process environments. See the [GPU guide](https://JuliaSMLM.github.io/GaussMLE.jl/dev/guide/gpu/) for details.

## PSF Models

All PSF parameters are specified in microns for camera-independence.

| Model | Constructor | Fitted Parameters | Use Case |
|-------|-------------|-------------------|----------|
| `GaussianXYNB` | `GaussianXYNB(σ)` | x, y, N, bg | Fixed-width 2D Gaussian |
| `GaussianXYNBS` | `GaussianXYNBS()` | x, y, N, bg, σ | Variable-width 2D Gaussian |
| `GaussianXYNBSXSY` | `GaussianXYNBSXSY()` | x, y, N, bg, σx, σy | Anisotropic 2D Gaussian |
| `AstigmaticXYZNB` | `AstigmaticXYZNB{T}(σx₀, σy₀, Ax, Ay, Bx, By, γ, d)` | x, y, z, N, bg | 3D astigmatic localization |

## Output Format

`fit()` returns `(BasicSMLD, GaussMLEFitInfo)`. The emitter type in `BasicSMLD` depends on the PSF model:

| PSF Model | Emitter Type | Extra Fields |
|-----------|-------------|--------------|
| `GaussianXYNB` | `Emitter2DFitGaussMLE` | pvalue |
| `GaussianXYNBS` | `Emitter2DFitSigma` | σ, σ_σ, pvalue |
| `GaussianXYNBSXSY` | `Emitter2DFitSigmaXY` | σx, σy, σ_σx, σ_σy, pvalue |
| `AstigmaticXYZNB` | `Emitter3DFitGaussMLE` | z, σ_z, σ_xz, σ_yz, pvalue |

All emitters include: `x`, `y`, `photons`, `bg`, `σ_x`, `σ_y`, `σ_xy`, `σ_photons`, `σ_bg`, `frame`.

`GaussMLEFitInfo` fields:

| Field | Description |
|-------|-------------|
| `elapsed_s` | Wall time (seconds) |
| `backend` | `:cpu` or `:gpu` (never `:auto`) |
| `device_id` | GPU index (0-based) or -1 for CPU |
| `n_fits` | Number of ROIs processed |
| `n_converged` | Number converged (currently = n_fits) |
| `batch_size` | Actual batch size used |
| `n_batches` | Number of batches |
| `memory_per_batch` | Estimated bytes per batch |

## Performance

Benchmarked on AMD Ryzen Threadripper PRO 5975WX / NVIDIA RTX A6000, 11x11 ROIs, 10K fits:

| Model | CPU (fits/s) | GPU (fits/s) |
|-------|-------------|--------------|
| GaussianXYNB (ideal) | 2,120 | 401,572 |
| GaussianXYNB (sCMOS) | 2,077 | 261,590 |
| GaussianXYNBS (ideal) | 1,633 | 215,745 |
| AstigmaticXYZNB (ideal) | 1,341 | 265,083 |

GPU throughput scales with batch size; these numbers use the default batch size of 10K. Run `Pkg.test("GaussMLE")` locally to benchmark your hardware.

## Algorithm Reference

MLE fitting algorithm:

> Smith, C., Joseph, N., Rieger, B. et al. "Fast, single-molecule localization that achieves theoretically minimum uncertainty." *Nat Methods* **7**, 373-375 (2010). [DOI: 10.1038/nmeth.1449](https://doi.org/10.1038/nmeth.1449)

sCMOS pixel-dependent noise model:

> Huang, F., Hartwich, T.M.P., Rivera-Molina, F.E. et al. "Video-rate nanoscopy using sCMOS camera-specific single-molecule localization algorithms." *Nat Methods* **10**, 653-658 (2013). [DOI: 10.1038/nmeth.2488](https://doi.org/10.1038/nmeth.2488)

## Related Packages

- **[SMLMAnalysis.jl](https://github.com/JuliaSMLM/SMLMAnalysis.jl)** - Complete SMLM workflow (detection + fitting + rendering)
- **[SMLMData.jl](https://github.com/JuliaSMLM/SMLMData.jl)** - Core data types for SMLM
- **[SMLMSim.jl](https://github.com/JuliaSMLM/SMLMSim.jl)** - SMLM data simulation
- **[MicroscopePSFs.jl](https://github.com/JuliaSMLM/MicroscopePSFs.jl)** - PSF models

## License

MIT License - see [LICENSE](LICENSE) file for details.
