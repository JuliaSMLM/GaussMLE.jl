# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
```bash
# Core group (what GitHub CI runs)
julia --project -e 'using Pkg; Pkg.test()'

# One or more groups: Core, QA, GPU, Long
GROUP=QA julia --project -e 'using Pkg; Pkg.test()'
GROUP=GPU,Long julia --project -e 'using Pkg; Pkg.test()'

# Every group this machine can run (GPU needs functional CUDA or AMDGPU in its group env)
GROUP=Everything julia --project -e 'using Pkg; Pkg.test()'
```

Groups are declared in `test/test_groups.toml` (lab standard, admiral decision 0008).

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

These four short scripts use the repository's active environment and the current API:
`smld, info = fit(data, fitter)`, with `backend` selecting CPU or GPU. The sCMOS
example fits an `ROIBatch` with a calibrated camera. The GPU example compares backends
when CUDA or AMDGPU is functional; otherwise it reports that the comparison is unavailable.

## Architecture

### Core Implementation

The package uses a unified kernel approach via KernelAbstractions.jl:
- Single implementation works on both CPU and GPU via `@kernel` macro
- Immutable StaticArrays (SVector/SMatrix) for allocation-free CPU and GPU kernels
- Custom GPU-compatible linear algebra (Cholesky decomposition and inverse)
- **Optimization (`_optimize`)**: Levenberg-Marquardt on the full Fisher matrix
  (`_lm_fit`) for every model; the Gaussian models stop once converged, `SplinePSFModel`
  runs several z starts
- **Full Fisher Information Matrix for CRLB** (symmetric, requires full matrix inverse)
- Spline pixel-integral tables are built on the host and adapted once per backend attempt;
  CPU (POCL under KernelAbstractions 0.10, native under 0.9), CUDA and AMDGPU use one
  kernel, with the point callable removed before launch
- Spline values and analytic first derivatives share one 4 × 4 × 4 gather (`compute_pixel_derivatives` returns `(model, dudt)` for every model: no fit uses second derivatives);
  rectangular pixels use each axis's spacing, and non-positive values have zero derivatives
- Spline initialization and z scanning use immutable `SVector` parameters; the scan gathers
  values only. Coefficients typically occupy 5 to 10 MiB for a PSFLearning export

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
SMLMData.SCMOSCamera{T}
    offset::Union{T, Matrix{T}}     # Offset (ADU)
    gain::Union{T, Matrix{T}}       # Gain (e⁻/ADU)
    readnoise::Union{T, Matrix{T}}  # Readout noise σ (e⁻)
```

Calibration values can be scalars or `(ny, nx)` matrices, with the same element type
as the pixel size.

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
nx, ny = 512, 512
camera = SMLMData.SCMOSCamera(nx, ny, 0.1f0, readnoise; offset=offset, gain=gain)
# readnoise is σ in electrons

# 2. ROIBatch created with camera and corner positions
batch = ROIBatch(data, x_corners, y_corners, frame_indices, camera)
# x_corners[i], y_corners[i] = camera position for each ROI (1-based)

# 3. fit() extracts variance map from camera
variance = camera.readnoise .^ 2  # Convert σ → σ² (variance in e⁻²)
variance_map = variance isa Real ? fill(variance, ny, nx) : variance
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
| **Corners** | Required for output coordinates | Required for output coordinates and variance-map lookup |
| **Data units** | Photons (or ADU if gain=1) | Must convert ADU → electrons |
| **Fit objective** | Poisson log-likelihood | Poisson-shifted: `(data+v) log(model+v) - (model+v)`, `v` = readnoise² |
| **Gradient weight** | `(data-model)/model` | `(data-model)/total_var` |
| **p-value statistic** | Pearson `Σ (data-model)²/model` against χ²(df) | Pearson `Σ (data-model)²/total_var` against χ²(df) (`-2×` the log-likelihood ratio) |
| **Kernel indexing** | `data[i,j]` only | `data[i,j]` and `variance_map[cam_y, cam_x]` |

**Key implementation difference**:
- IdealCamera: Noise variance = model value (Poisson)
- SCMOSCamera: Noise variance = model + readnoise²[y,x] (pixel-dependent)

#### Kernel Indexing (1-Based Julia Convention)

**In unified_kernel.jl, all indexing is 1-based:**

```julia
# For IdealCamera (corners used for output coordinates, not noise lookup)
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

#### Optimization and Fisher Information
The kernel uses the Fisher matrix for both optimization and uncertainty estimation:

1. **Optimization: Levenberg-Marquardt (`_lm_fit`) on the full Fisher matrix, every model**
   - One pixel pass per iteration at the trial point: objective (`_fit_objective`, read-noise
     variance from `_readout_variance`, zero for IdealCamera), gradient
     `g = Σ dudt (d - μ)/var` and Fisher `F = Σ dudt dudt'/var`; no second derivatives
   - Step: `(F_acc + λ diag F_acc) Δ = g_acc` (Fisher scoring: F is the expected curvature),
     held to `max_step` and the bounds; a higher objective is accepted, a lower one rejected
     (λ×10); factorizations in the `@noinline` `_lm_solve` and `_newton_gain` (POCL speed)
   - A parameter on a bound whose gradient points out of the box is held (identity row and
     column via `_hold_rows`, zero gradient) while the rest take the step
   - Default photon and background steps are uncapped (`Inf32` in `default_constraints`)
   - `iterations` caps the pixel passes, the last only evaluating, so a fit takes at most
     `iterations - 1` steps; default `_default_iterations`: 100 for the Gaussian models, 20 for
     SplinePSFModel; `_lm_fit` returns `(θ_acc, obj_acc, passes, stopped)`, and `_optimize`
     returns `(θ, converged)`
   - Gaussian models (`_optimize` for `PSFModel`, `gain_tol = _LM_GAIN_TOL = 1e-4`, adaptive):
     stop after an accepted pass whose predicted gain `g' F⁻¹ g / 2` over the free parameters
     (undamped `F_acc`) is below 1e-4, i.e. within 0.014 CRLB of the maximum of the quadratic
     model with curvature F (measured within 0.1 CRLB of the likelihood's own maximum on
     simulated ROIs); λ by Nielsen's rule from the gain ratio ρ on accepted passes
     (λ·max(1/3, 1 − (2ρ − 1)³), ν = 2) and λ·ν, ν·2 on rejected ones; a trial is kept on a
     positive gain, the trapezoid `(g_acc + g)' s / 2` when the objective change is within
     10× its Float32 rounding (its sign decides: added to `obj_acc` it rounds away for a
     mismatched bright fit, |obj| ~ 1e4 to 1e5); a non-finite objective is rejected; the whole step is scaled to `max_step` instead of
     clamping each component (`@noinline _adaptive_trial`, which skips non-finite
     components); a trial that rounds to `θ_acc` with λ at its 1e7 cap ends the fit (a
     fixed point: every later pass makes the same trial); a fit that ends without the stop test gets `converged = false`,
     which the kernel writes to its `converged` output (`UInt8`, after `chi2`) and
     `GaussMLEFitInfo.n_converged` counts; its uncertainties stay the CRLB
   - SplinePSFModel: `_optimize(::SplinePSFModel)` runs every pass (`gain_tol = 0`: λ/10 on
     accept, per-component clamp, no stop, bit for bit as before the Gaussian change), from
     the scan start and then from `starts - 1` fixed z starts (`_spline_start_z`; x, y, N,
     bg from the scan start; default count `_default_spline_starts`, one fixed start per
     0.4 µm over the central 80 % of `z_range`), each with the full `iterations`, keeping
     the strictly higher objective (the earlier start on a tie). `starts = 1` is the scan
     start alone

2. **Uncertainty (CRLB)**: Requires full Fisher Information Matrix
   - Full symmetric matrix needed for proper uncertainties
   - `_invert_fisher` inverts the Jacobi-scaled matrix (`G = D F D`, `d_k = 1/sqrt(F_kk)`)
     and returns `D G⁻¹ D`, with no added regularization: an absolute one swamps the
     photon entry `F_NN ~ 1/N` against `F_xx ~ N` and collapses `σ_photons` for bright
     emitters. G is factored by Cholesky only (`static_cholesky_decomposition` with a
     tolerance): a pivot at or below `n_pixels * eps(T)` (`n_pixels = box_size^2`, passed
     from the kernel) means F is singular to working precision. The Float32 accumulation
     over the box's pixels perturbs the entries of G by about that much, so a smaller
     pivot of a real matrix cannot be told from an exactly singular one (a tolerance of
     `N * eps(T)` passes a simulated ROI, which then reports σ_x = 72 nm). There is no LU
     fallback, because F is positive semidefinite by construction and LU there only turns
     rounding into a number. Failed inversion: a `F_kk` that is not positive and finite, a failed Cholesky, or an inverse
     diagonal entry that is not positive and finite (no `max(0, ...)` on a variance).
     Failed means every uncertainty is `Inf` and covariances are zero
   - After the kernel, on the host (`_mark_pinned_fits!`, CPU and GPU alike), a fit whose
     x or y (and z for AstigmaticXYZNB and SplinePSFModel) ended exactly on a lower or
     upper bound of the constraints the kernel used did not localize an emitter inside the
     allowed region: its uncertainties are `Inf` and covariances zero too; parameters,
     log-likelihood and p-value are unchanged. A parameter the caller holds fixed
     (`lower == upper`) is not a bound hit
   - Off-diagonal correlations matter for CRLB

**SIGN CONVENTION** (unified_kernel.jl): the gradient is of the log-likelihood (ascent), and
the step solves `(F + λ diag F) Δ = +g` with `θ_new = θ + Δ`:

```julia
g  += dudt * ((d - model) / var)          # ascent gradient
F  += (dudt * dudt') * (1 / var)          # Fisher = E[-∇²L], positive semidefinite
```

Poisson Fisher = E[-∇²L] (only 1st derivatives); sCMOS uses `var = model + readnoise²` in
the same outer-product accumulation, which is also the Poisson-shifted objective's Fisher.

#### Custom Linear Algebra

GPU kernels cannot use standard library matrix operations, so custom implementations are provided:

- `static_cholesky_decomposition(A, tol = zero(T))` - Cholesky (symmetric positive
  definite); a pivot at or below `tol` fails
- `static_cholesky_inverse` - Invert via Cholesky factors

All accept immutable `SMatrix` values and return `(ok::Bool, result)` without mutation.
The Levenberg-Marquardt step and its convergence check call the decomposition without a
tolerance (a positive pivot passes, a NaN pivot is not caught there); the inverse helper has no
tolerance check. The Fisher matrix goes through `_invert_fisher(F, n_pixels)`, which scales
it to a unit diagonal first and factors it with tolerance `n_pixels * eps(T)`; a NaN pivot
ends as a failed inversion through the check on the inverse diagonal.

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
- `src/interface.jl` - High-level `fit()` API and `GaussMLEConfig` type
- `src/psf_models.jl` - PSF model definitions (Gaussian, Astigmatic)
- `src/psf_derivatives.jl` - PSF derivative computations
- `src/camera_models.jl` - Camera noise models (IdealCamera, SCMOSCameraInternal)
- `src/devices.jl` - Device abstraction (CPU/GPU auto-detection), GPU vendor interface; `_gpu_workgroup_size` fixes AMD launches at 64 (an unsized AMDGPU launch of a large batch hung on a Radeon 780M), CUDA sizes its own
- `src/emitters.jl` - Custom emitter types (Emitter2DFitSigma, Emitter2DFitSigmaXY)
- `src/results.jl` - Fit metadata and legacy result types
- `src/precompile.jl` - PrecompileTools workload: a few CPU fits per model and camera (never a GPU context; skipped where KernelAbstractions 0.10's POCL is unavailable)
- `src/roi_batch.jl` - Localization results and emitter/SMLD conversion
- `src/simulator.jl` - Data generation for testing
- `src/constraints.jl` - Parameter constraints system
- `src/gausslib/` - Gaussian math helpers used by CPU and GPU kernels
- `src/spline_psf.jl` - `SplinePSFModel`: tabulated 3D PSF (moved from PSFLearning), host
  construction, backend adaptation, analytic derivatives, and the multi-start fit
  (`starts` field) on CPU and GPU
- `ext/GaussMLEMicroscopePSFsExt.jl` - `SplinePSFModel(::MicroscopePSFs.SplinePSF; ...)` constructor, with the `starts` keyword (weak dependency)

### SMLMData Ecosystem Integration

**Returns SMLMData.BasicSMLD** with model-specific emitter types:
- `GaussianXYNB` → `GaussMLE.Emitter2DFitGaussMLE`
- `GaussianXYNBS` → `GaussMLE.Emitter2DFitSigma` (custom, includes σ)
- `GaussianXYNBSXSY` → `GaussMLE.Emitter2DFitSigmaXY` (custom, includes σx/σy)
- `AstigmaticXYZNB` → `GaussMLE.Emitter3DFitGaussMLE`
- `SplinePSFModel` → `GaussMLE.Emitter3DFitGaussMLE`

All emitter types subtype `SMLMData.AbstractEmitter` for ecosystem compatibility.

### Testing Philosophy

See `test/CLAUDE.md` for detailed testing guidelines. Key points:
- Lab test layout: `test/runtests.jl` is the lab template (never edit it); groups are
  Core (`test/*.jl`), QA (`test/qa/`), GPU (`test/gpu/`), Long (`test/long/`)
- **No @test_skip**: Failing tests indicate real problems
- Each test file runs in its own module and carries its own `using` statements
- The performance benchmark is part of the GPU group; GitHub CI runs only Core and QA

### Reference Implementation

Original MATLAB/CUDA implementation: https://github.com/LidkeLab/smite/tree/main/MATLAB/source/cuda
