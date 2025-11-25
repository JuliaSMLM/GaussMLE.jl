# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
```bash
# Run all tests (GPU auto-detected)
julia --project -e 'using Pkg; Pkg.test()'

# Run specific test file
julia --project -e 'using Test; include("test/model_validation_tests.jl")'
```

GPU tests run automatically when a CUDA GPU is detected.

### Documentation
```bash
# Build documentation
julia --project=docs docs/make.jl

# Serve documentation locally
julia --project=docs -e 'using LiveServer; serve(dir="docs/build")'
```

### Examples
```bash
# Run example scripts
julia --project examples/basic_fitting.jl
julia --project examples/scmos_camera.jl
julia --project examples/gpu_acceleration.jl
julia --project examples/astigmatic_3d.jl
```

## Architecture

### Core Implementation

The package uses a unified kernel approach via KernelAbstractions.jl:
- Single implementation works on both CPU and GPU via `@kernel` macro
- StaticArrays (MVector/MMatrix) for stack allocation and GPU compatibility
- Custom GPU-compatible linear algebra (Cholesky/LU decomposition, no pivoting)
- **Diagonal Newton-Raphson for optimization** (element-wise, not full matrix inverse)
- **Full Fisher Information Matrix for CRLB** (symmetric, requires full matrix inverse)

### Data Types and Conventions

#### Julia Array Indexing Convention

**CRITICAL**: Julia arrays use **[y, x]** indexing (row, column):
```julia
# Array dimensions for ROI data
data[i, j, roi_idx]  # i = y coordinate (row), j = x coordinate (column)

# Variance maps from camera
variance_map[y, x]   # First index is y (row), second is x (column)

# When looping over pixels in kernel
for j in 1:box_size    # j is x (column)
    for i in 1:box_size  # i is y (row)
        pixel_value = data[i, j, roi_idx]
    end
end
```

This is standard Julia column-major indexing where the **first dimension varies fastest** in memory.

#### SMLMData Types

The package receives two key types from SMLMData.jl:

**1. Camera Models** (from SMLMData.jl):

```julia
# Ideal camera - Poisson noise only
SMLMData.IdealCamera
# No calibration fields needed

# sCMOS camera - Poisson + pixel-dependent readout noise
SMLMData.SCMOSCamera
    offset::Matrix{Float32}      # Per-pixel offset (ADU)
    gain::Matrix{Float32}        # Per-pixel gain (e⁻/ADU)
    readnoise::Matrix{Float32}   # Per-pixel readout noise σ (e⁻)
```

**2. ROI Batches** (from SMLMData.jl):

```julia
SMLMData.ROIBatch{T}
    data::Array{T,3}              # (box_size, box_size, n_rois) - ROI pixels
    x_corners::Vector{Int32}      # X (column) position of each ROI on full sensor
    y_corners::Vector{Int32}      # Y (row) position of each ROI on full sensor
    frame_indices::Vector{Int32}  # Frame number for each ROI
    camera::AbstractCamera        # Camera model (IdealCamera or SCMOSCamera)
    # x_corners[i], y_corners[i] = 1-based camera coordinates of ROI top-left
```

#### Variance Map Flow: Camera → ROIBatch → Kernel

**Complete data flow for sCMOS fitting:**

```julia
# 1. User creates camera with calibration
camera = SMLMData.SCMOSCamera(offset, gain, readnoise)
# readnoise is σ in electrons

# 2. ROIBatch created with camera and corner positions
batch = ROIBatch(data, x_corners, y_corners, frame_indices, camera)
# x_corners[i], y_corners[i] = camera position for each ROI (1-based)

# 3. fit() extracts variance map from camera
variance_map = camera.readnoise .^ 2  # Convert σ → σ² (variance in e⁻²)
# variance_map is same size as full camera sensor

# 4. Kernel receives variance_map and separate corner vectors
kernel(..., variance_map, x_corners, y_corners, ...)

# 5. Kernel indexes variance map at camera coordinates
for j in 1:box_size  # x within ROI
    for i in 1:box_size  # y within ROI
        # Map ROI pixel to camera coordinates (1-based)
        camera_y = i + y_corners[roi_idx] - 1
        camera_x = j + x_corners[roi_idx] - 1

        # Index variance map at camera position
        variance = variance_map[camera_y, camera_x]
    end
end
```

**Why corners are needed**: Each ROI is extracted from a specific position on the full camera sensor. The variance map is calibrated for the entire sensor, so we must index it at the correct absolute position, not the ROI-relative position.

#### IdealCamera vs SCMOSCamera

The two camera models have **fundamentally different data flows**:

| Aspect | IdealCamera | SCMOSCamera |
|--------|-------------|-------------|
| **Noise model** | Poisson only | Poisson + pixel readout noise |
| **Calibration** | None | offset, gain, readnoise maps |
| **Variance map** | Not used | Required (readnoise²) |
| **Corners** | Not used (can be `nothing`) | Required for indexing variance map |
| **Data units** | Photons (or ADU if gain=1) | Must convert ADU → electrons |
| **Likelihood** | `cf = data/model - 1`<br>`df = data/model²` | `cf = (data-model)/total_var`<br>`df = 1/total_var` |
| **Kernel indexing** | `data[i,j]` only | `data[i,j]` and `variance_map[cam_y, cam_x]` |

**Key implementation difference**:
- IdealCamera: Noise variance = model value (Poisson)
- SCMOSCamera: Noise variance = model + readnoise²[y,x] (pixel-dependent)

#### Kernel Indexing (1-Based Julia Convention)

**In unified_kernel.jl, all indexing is 1-based:**

```julia
# For IdealCamera (no corners needed)
for j in 1:box_size    # x: 1, 2, ..., box_size
    for i in 1:box_size  # y: 1, 2, ..., box_size
        pixel_data = data[i, j, roi_idx]
        # i and j are ROI-relative, 1-based
    end
end

# For SCMOSCamera (corners required)
corner_x = x_corners[roi_idx]  # 1-based camera X position of ROI top-left
corner_y = y_corners[roi_idx]  # 1-based camera Y position of ROI top-left

for j in 1:box_size    # x within ROI
    for i in 1:box_size  # y within ROI
        # ROI pixel data
        pixel_data = data[i, j, roi_idx]

        # Map to camera coordinates (1-based)
        camera_y = i + corner_y - 1  # -1 because corner is 1-based
        camera_x = j + corner_x - 1

        # Variance at this camera position
        variance = variance_map[camera_y, camera_x]
    end
end
```

**Why the -1**: `corners` are 1-based (Julia convention). ROI pixel (1,1) corresponds to camera pixel at `corner`, so we subtract 1 before adding the offset.

### Critical Implementation Details

#### Newton-Raphson vs Fisher Information
The kernel uses **different approaches** for optimization vs uncertainty estimation:

1. **Optimization (Newton-Raphson)**: Uses diagonal Hessian approximation
   - Element-wise updates: `θ[k] -= ∇L[k] / H_diag[k]`
   - More stable than full Newton method for convergence
   - Diagonal elements: `H_diag[k] += d2udt2[k,k] * cf - dudt[k] * dudt[k] * df`

2. **Uncertainty (CRLB)**: Requires full Fisher Information Matrix
   - Full symmetric matrix needed for proper uncertainties
   - Inverted via Cholesky decomposition (more robust than LU for positive definite matrices)
   - Off-diagonal correlations matter for CRLB

**CRITICAL SIGN CONVENTION** (unified_kernel.jl):

The diagonal Hessian and Fisher matrix have **different formulas with different signs**:

```julia
# Diagonal Hessian for Newton-Raphson (line ~331)
H_diag[k] += d2udt2_diag[k] * cf - dudt[k] * dudt[k] * df
#            ^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^
#            PLUS (2nd derivative)   MINUS (1st derivative²)

# Fisher Information Matrix for CRLB (line ~389)
F_kl = dudt[k] * dudt[l] / variance
H[k,l] += F_kl  # ONLY positive, no 2nd derivative term
```

**Why different**:
- Hessian = actual ∇²L (both 1st and 2nd derivatives, can be negative)
- Fisher = E[-∇²L] (only 1st derivatives, always positive definite)

Wrong signs cause convergence failure or incorrect CRLB.

#### Custom Linear Algebra

GPU kernels cannot use standard library matrix operations, so custom implementations are provided:

- `static_cholesky_decomposition!` - In-place Cholesky (preferred for symmetric positive definite)
- `static_cholesky_inverse!` - Invert via Cholesky factors
- `static_lu_decomposition!` - In-place LU without pivoting (fallback)
- `static_back_substitution!` - Solve linear systems

All use relative tolerances (`1e-10 * maximum(abs, A)`) matching SMITE.

#### Astigmatic Model Unit Convention

The astigmatic model uses **mixed units** (matching SMITE convention):
- **x, y**: pixels internally, converted to microns for output
- **z**: microns throughout (axial position is physical, not pixel-based)
- **σx₀, σy₀**: microns (converted to pixels internally)
- **γ, d**: microns (NOT converted - z-related parameters stay physical)

This means in `to_pixel_units(::AstigmaticXYZNB)`:
```julia
σx₀ / pixel_size  # Convert lateral width to pixels
σy₀ / pixel_size  # Convert lateral width to pixels
γ                  # Keep in microns (z is physical)
d                  # Keep in microns (z is physical)
```

The math is dimensionally consistent:
- `z - γ` → microns - microns = microns
- `(z - γ) / d` → dimensionless
- `dσ/dz` → pixels/micron
- `du/dz` → photons/micron
- CRLB[z] → microns

### Source Structure

- `src/unified_kernel.jl` - Main fitting kernel with custom linear algebra
- `src/interface.jl` - High-level `fit()` API and `GaussMLEFitter` type
- `src/psf_models.jl` - PSF model definitions (Gaussian, Astigmatic)
- `src/psf_derivatives.jl` - PSF derivative computations
- `src/camera_models.jl` - Camera noise models (IdealCamera, SCMOSCameraInternal)
- `src/devices.jl` - Device abstraction (CPU/GPU auto-detection)
- `src/emitters.jl` - Custom emitter types (Emitter2DFitSigma, Emitter2DFitSigmaXY)
- `src/results.jl` - Result conversion to SMLMData.BasicSMLD
- `src/roi_batch.jl` - ROI batch data structures
- `src/simulator.jl` - Data generation for testing
- `src/constraints.jl` - Parameter constraints system
- `src/gausslib/` - Legacy GaussLib reference implementation (CPU only)

### SMLMData Ecosystem Integration

**Returns SMLMData.BasicSMLD** with model-specific emitter types:
- `GaussianXYNB` → `SMLMData.Emitter2DFit`
- `GaussianXYNBS` → `GaussMLE.Emitter2DFitSigma` (custom, includes σ)
- `GaussianXYNBSXSY` → `GaussMLE.Emitter2DFitSigmaXY` (custom, includes σx/σy)
- `AstigmaticXYZNB` → `SMLMData.Emitter3DFit`

All emitter types subtype `SMLMData.AbstractEmitter` for ecosystem compatibility.

### Testing Philosophy

See `test/CLAUDE.md` for detailed testing guidelines. Key points:
- **No test gating**: All tests run on every invocation (except GPU hardware detection)
- **No @test_skip**: Failing tests indicate real problems
- Local performance benchmarks run automatically (skipped in CI via `ENV["CI"]` check)
- All `using` statements must be in `test/runtests.jl`

### Reference Implementation

Original MATLAB/CUDA implementation: https://github.com/LidkeLab/smite/tree/main/MATLAB/source/cuda

Design reference: `.claude/ref/package_structure.md`