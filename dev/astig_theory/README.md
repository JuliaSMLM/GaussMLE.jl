# Astigmatic PSF Theory - Fisher Information and CRLB

This directory contains a complete theoretical derivation and direct calculation of the Fisher Information Matrix and Cramér-Rao Lower Bound (CRLB) for astigmatic PSF localization.

## Contents

### 1. `astigmatic_fisher_derivation.md`

Complete mathematical derivation including:
- **Astigmatic PSF model** with z-dependent widths
- **Derivative calculations** for all 5 parameters [x, y, z, N, bg]
- **Fisher Information Matrix** formulation
- **CRLB computation** via matrix inversion
- **Physical interpretation** of z-precision variation

**Key insight**: The z-precision depends on |dσ/dz| - where the PSF shape changes rapidly with z, localization is more precise.

### 2. `compute_theoretical_crlb.jl`

Direct implementation of the theoretical derivation:
- ✅ **No GaussLib dependencies** - everything derived from first principles
- Implements pixel-integrated Gaussian PSF
- Computes all derivatives analytically
- Calculates 5×5 Fisher Information Matrix
- Inverts to get CRLB for each parameter
- Generates plots showing CRLB vs z

### 3. Output: `../output/theoretical_crlb_vs_z.png`

Visualization showing:
1. **PSF widths** σx(z) and σy(z) - characteristic astigmatic crossing
2. **Width gradients** |dσx/dz| and |dσy/dz| - z-information sources
3. **Lateral CRLB** (x,y precision) - varies from 0.031 to 0.048 pixels
4. **Axial CRLB** (z precision) - varies from 15 to 37 nm across ±600nm range

## Key Results

### CRLB Variation with Z

For test conditions (2000 photons, bg=1, 15×15 pixel ROI):

| Position | CRLB_z | Physical Reason |
|----------|--------|-----------------|
| z ≈ -500nm | **15.1 nm** (best) | Large \|dσx/dz\|, σx changing rapidly |
| z ≈ 0nm | **36.9 nm** (worst) | dσx/dz and dσy/dz have opposite signs, partial cancellation |
| z ≈ +500nm | **15.2 nm** (best) | Large \|dσy/dz\|, σy changing rapidly |

**Variation**: 2.44× between best and worst positions

### Physical Interpretation

The bottom-right plot shows **Z-Information** via width gradients:
- When |dσ/dz| is **large** → PSF shape sensitive to z → **small CRLB** (good precision)
- When |dσ/dz| is **small** → PSF shape insensitive to z → **large CRLB** (poor precision)

At z=0, the two gradients have opposite effects:
- σx is narrowing (dσx/dz < 0)
- σy is widening (dσy/dz > 0)
- Partial cancellation → reduced total z-information

## Running the Code

```bash
julia --project=dev dev/astig_theory/compute_theoretical_crlb.jl
```

No dependencies on GaussLib or GaussMLE internals - completely standalone calculation.

## Validation

This theoretical calculation can be compared against:
1. Empirical fitting results (std of fitted - true across many noise realizations)
2. Production code Fisher Information (from unified_kernel.jl)

For correctly converged fits: **empirical std ≈ theoretical CRLB** (ratio ≈ 1.0)

## Mathematical Consistency

The theoretical CRLB shows:
- **Best precision at extreme z** (±500nm): 15nm
- **Worst precision near center** (z≈0): 37nm

This matches the physical behavior:
- At extreme z: wide PSF but rapidly changing shape → good z-information
- Near z=0: narrowest PSF but opposite gradient directions → reduced z-information

The empirical results showing better performance at extremes are **physically correct** - this is not a bug, but the actual behavior of astigmatic PSF localization.
