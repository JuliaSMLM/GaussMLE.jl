# Coordinate Convention Analysis

## Problem Statement

GaussMLE.jl currently uses inconsistent coordinate conventions with respect to Julia's standard array indexing. This analysis documents all violations and proposes systematic fixes.

## Julia's Standard Convention

For 2D arrays in Julia:
- **First index**: Row (y-coordinate, vertical)
- **Second index**: Column (x-coordinate, horizontal)
- Arrays are **column-major** (first index varies fastest in memory)

For an array `A[i, j]`:
- `i` = row index = y-coordinate
- `j` = column index = x-coordinate

## Current Implementation (INCORRECT)

The codebase systematically treats:
- `i` as x-coordinate (should be y)
- `j` as y-coordinate (should be x)

## Violations Found

### 1. PSF Evaluation Functions

**Location**: `src/psf_models.jl:166-171` (and similar for all PSF models)

**Current (WRONG)**:
```julia
@inline function evaluate_psf(psf::GaussianXYNB, i, j, θ::Params{4})
    x, y, N, bg = θ
    psf_x = integral_gaussian_1d(i, x, psf.σ)  # i with x - WRONG
    psf_y = integral_gaussian_1d(j, y, psf.σ)  # j with y - WRONG
    return bg + N * psf_x * psf_y
end
```

**Should be**:
```julia
@inline function evaluate_psf(psf::GaussianXYNB, i, j, θ::Params{4})
    x, y, N, bg = θ
    psf_x = integral_gaussian_1d(j, x, psf.σ)  # j with x - CORRECT
    psf_y = integral_gaussian_1d(i, y, psf.σ)  # i with y - CORRECT
    return bg + N * psf_x * psf_y
end
```

**Affected functions**:
- `evaluate_psf(::GaussianXYNB, ...)` (line 166)
- `evaluate_psf(::GaussianXYNBS, ...)` (line 174)
- `evaluate_psf(::GaussianXYNBSXSY, ...)` (line 182)
- `evaluate_psf(::AstigmaticXYZNB, ...)` (line 190)

### 2. PSF Derivative Computation

**Location**: `src/psf_derivatives.jl:6-36` (and similar for all models)

**Current (WRONG)**:
```julia
@inline function compute_pixel_derivatives(i, j, θ::Params{4}, psf::GaussianXYNB)
    x, y, N, bg = θ

    psf_x = integral_gaussian_1d(i, x, psf.σ)  # i with x - WRONG
    psf_y = integral_gaussian_1d(j, y, psf.σ)  # j with y - WRONG

    dudt_x, d2udt2_x = derivative_integral_gaussian_1d(i, x, psf.σ, N, psf_y)  # WRONG
    dudt_y, d2udt2_y = derivative_integral_gaussian_1d(j, y, psf.σ, N, psf_x)  # WRONG
    ...
end
```

**Should be**:
```julia
@inline function compute_pixel_derivatives(i, j, θ::Params{4}, psf::GaussianXYNB)
    x, y, N, bg = θ

    psf_x = integral_gaussian_1d(j, x, psf.σ)  # j with x - CORRECT
    psf_y = integral_gaussian_1d(i, y, psf.σ)  # i with y - CORRECT

    dudt_x, d2udt2_x = derivative_integral_gaussian_1d(j, x, psf.σ, N, psf_y)  # CORRECT
    dudt_y, d2udt2_y = derivative_integral_gaussian_1d(i, y, psf.σ, N, psf_x)  # CORRECT
    ...
end
```

**Affected functions**:
- `compute_pixel_derivatives(i, j, ::Params{4}, ::GaussianXYNB)` (line 6)
- `compute_pixel_derivatives(i, j, ::Params{5}, ::GaussianXYNBS)` (line 40)
- `compute_pixel_derivatives(i, j, ::Params{6}, ::GaussianXYNBSXSY)` (line 76)
- `compute_pixel_derivatives(i, j, ::Params{5}, ::AstigmaticXYZNB)` (line 115)

### 3. Parameter Initialization (Center of Mass)

**Location**: `src/psf_models.jl:216-217` (repeated in all `initialize_parameters` functions)

**Current (WRONG)**:
```julia
x = sum((1:box_size) .* sum(signal, dims=2)[:]) / sum(signal)  # dims=2 sums over cols, weights by rows → y
y = sum((1:box_size) .* sum(signal, dims=1)[:]) / sum(signal)  # dims=1 sums over rows, weights by cols → x
```

**Explanation**:
- `sum(signal, dims=2)` sums over columns (dimension 2), leaving row sums
- Weighting by `(1:box_size)` gives row-weighted sum = y-coordinate
- `sum(signal, dims=1)` sums over rows (dimension 1), leaving column sums
- Weighting by `(1:box_size)` gives column-weighted sum = x-coordinate

**Should be**:
```julia
y = sum((1:box_size) .* sum(signal, dims=2)[:]) / sum(signal)  # row-weighted = y
x = sum((1:box_size) .* sum(signal, dims=1)[:]) / sum(signal)  # column-weighted = x
```

**Affected functions** (all in `src/psf_models.jl`):
- `initialize_parameters(roi, ::GaussianXYNB)` (lines 216-217)
- `initialize_parameters(roi, ::GaussianXYNBS)` (lines 231-232)
- `initialize_parameters(roi, ::GaussianXYNBSXSY)` (lines 247-248)
- `initialize_parameters(roi, ::AstigmaticXYZNB)` (lines 264-265)

### 4. Kernel Initialization

**Location**: `src/unified_kernel.jl:175-179`

**Current (WRONG)**:
```julia
@inbounds for j in 1:box_size, i in 1:box_size
    signal = max(zero(T), roi[i,j] - bg)
    total_signal += signal
    x_weighted += signal * T(i)  # i → x - WRONG
    y_weighted += signal * T(j)  # j → y - WRONG
end
```

**Should be**:
```julia
@inbounds for j in 1:box_size, i in 1:box_size
    signal = max(zero(T), roi[i,j] - bg)
    total_signal += signal
    x_weighted += signal * T(j)  # j → x - CORRECT
    y_weighted += signal * T(i)  # i → y - CORRECT
end
```

### 5. Simulator PSF Evaluation

**Location**: `src/simulator.jl:292-296` (and similar functions)

**Current (WRONG)**:
```julia
function _evaluate_psf_pixel(psf::GaussianXYNB, i::Int, j::Int, params::AbstractVector)
    x, y, photons, bg = params
    psf_x = integral_gaussian_1d(i, x, psf.σ)  # i with x - WRONG
    psf_y = integral_gaussian_1d(j, y, psf.σ)  # j with y - WRONG
    return bg + photons * psf_x * psf_y
end
```

**Should be**:
```julia
function _evaluate_psf_pixel(psf::GaussianXYNB, i::Int, j::Int, params::AbstractVector)
    x, y, photons, bg = params
    psf_x = integral_gaussian_1d(j, x, psf.σ)  # j with x - CORRECT
    psf_y = integral_gaussian_1d(i, y, psf.σ)  # i with y - CORRECT
    return bg + photons * psf_x * psf_y
end
```

**Affected functions** (all in `src/simulator.jl`):
- `_evaluate_psf_pixel(::GaussianXYNB, ...)` (line 292)
- `_evaluate_psf_pixel(::GaussianXYNBS, ...)` (line 299)
- `_evaluate_psf_pixel(::GaussianXYNBSXSY, ...)` (line 306)
- `_evaluate_psf_pixel(::AstigmaticXYZNB, ...)` (line 313)

### 6. Development Scripts

**Location**: `dev/analyze_astigmatic_crlb_vs_z.jl:130-131`

**Current (WRONG)**:
```julia
psf_x = GaussMLE.GaussLib.integral_gaussian_1d(i, x_true, σx_z)  # i with x - WRONG
psf_y = GaussMLE.GaussLib.integral_gaussian_1d(j, y_true, σy_z)  # j with y - WRONG
```

**Should be**:
```julia
psf_x = GaussMLE.GaussLib.integral_gaussian_1d(j, x_true, σx_z)  # j with x - CORRECT
psf_y = GaussMLE.GaussLib.integral_gaussian_1d(i, y_true, σy_z)  # i with y - CORRECT
```

## Correct Usage (No Changes Needed)

### Loop Structure
The loop structure in simulator and kernel is **correct**:
```julia
for j in 1:roi_size, i in 1:roi_size
    roi[i, j] = ...  # Accesses row i, column j - CORRECT
end
```

### Corners Storage
The corners convention is **correct**:
```julia
corners[1, i] = x  # First dimension = x-coordinate
corners[2, i] = y  # Second dimension = y-coordinate
```

### Parameters Storage
The parameters convention is **correct**:
```julia
parameters[1, :] = x  # First parameter = x
parameters[2, :] = y  # Second parameter = y
```

## Why This Matters

1. **Consistency**: Julia arrays should follow Julia conventions
2. **Interoperability**: Other Julia packages expect standard conventions
3. **Debugging**: Unexpected behavior when visualizing/inspecting arrays
4. **Future maintenance**: Confusion for new contributors
5. **Correctness**: While the current code may "work" because violations cancel out in some cases, it's fundamentally incorrect

## Impact Assessment

### Low Impact on Results (But Still Wrong)
Because the violations are **systematic** and **symmetric** (x and y are swapped consistently throughout), the current code may produce correct *numerical* results for:
- Symmetric PSFs (circular Gaussian)
- Centered emitters
- Statistical properties (variance, CRLB)

### Potential Issues
The swapped convention could cause problems with:
- Astigmatic PSFs (asymmetric in x and y)
- Position-dependent camera calibration
- Integration with other Julia packages
- Visual debugging and array inspection
- Future features that break the symmetry

## Testing Strategy

After fixing, we should verify:
1. All existing tests still pass
2. Astigmatic PSF tests specifically (most sensitive to x/y swap)
3. Visual inspection of fitted positions on test images
4. Comparison with reference implementations (MATLAB SMITE)

## Implementation Plan

1. **Fix all PSF evaluation functions** (`psf_models.jl`)
2. **Fix all derivative functions** (`psf_derivatives.jl`)
3. **Fix all initialization functions** (`psf_models.jl`, `unified_kernel.jl`)
4. **Fix simulator functions** (`simulator.jl`)
5. **Fix development scripts** (`dev/`)
6. **Run comprehensive tests**
7. **Verify astigmatic PSF specifically**

## Files Requiring Changes

1. `src/psf_models.jl` - Multiple functions
2. `src/psf_derivatives.jl` - All derivative functions
3. `src/unified_kernel.jl` - Initialization
4. `src/simulator.jl` - All PSF evaluation functions
5. `dev/analyze_astigmatic_crlb_vs_z.jl` - PSF evaluation

## Summary

The issue is **systematic and consistent** throughout the codebase. All instances of:
- `integral_gaussian_1d(i, x, ...)` should be `integral_gaussian_1d(j, x, ...)`
- `integral_gaussian_1d(j, y, ...)` should be `integral_gaussian_1d(i, y, ...)`
- Similar swaps for all derivative functions
- Center of mass calculations should swap x and y assignments

This is a straightforward find-and-replace operation, but requires careful verification afterward.
