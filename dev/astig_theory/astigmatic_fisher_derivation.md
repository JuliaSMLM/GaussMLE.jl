# Astigmatic PSF Fisher Information Matrix - Complete Derivation

## 1. Astigmatic PSF Model

### 1.1 Model Parameters

The astigmatic PSF model has 5 parameters:
- **θ = [x, y, z, N, bg]**
  - x, y: lateral position (pixels)
  - z: axial position (nm)
  - N: total photon count
  - bg: background per pixel

### 1.2 Z-Dependent Width Model

The key feature of astigmatic PSFs is that the x and y widths vary differently with z:

**Width scaling factors:**
```
αx(z) = 1 + ((z-γ)/d)² + Ax((z-γ)/d)³ + Bx((z-γ)/d)⁴
αy(z) = 1 + ((z+γ)/d)² + Ay((z+γ)/d)³ + By((z+γ)/d)⁴
```

where:
- γ: focal plane offset (nm) - creates astigmatism
- d: depth scale parameter (nm)
- Ax, Ay, Bx, By: aberration coefficients

**Actual widths:**
```
σx(z) = σx₀ √αx(z)
σy(z) = σy₀ √αy(z)
```

### 1.3 Pixel-Integrated Gaussian PSF

For a 1D Gaussian with mean μ and width σ, the integral over pixel i (spanning [i-0.5, i+0.5]) is:

```
Φ(i | μ, σ) = (1/2)[erf((i - μ + 0.5)/(√2 σ)) - erf((i - μ - 0.5)/(√2 σ))]
```

### 1.4 Full Model

The expected photon count at pixel (i,j) is:

```
μ(i,j | x,y,z,N,bg) = bg + N · Φ(i | x, σx(z)) · Φ(j | y, σy(z))
```

## 2. Derivative Calculations

### 2.1 Derivatives of erf Integral

For the pixel-integrated Gaussian, we need:

```
∂Φ/∂μ = (1/√(2π)σ) [exp(-((i-μ+0.5)/σ)²/2) - exp(-((i-μ-0.5)/σ)²/2)]

∂Φ/∂σ = -(1/√(2π)σ²) [(i-μ+0.5)exp(-((i-μ+0.5)/σ)²/2)
                        - (i-μ-0.5)exp(-((i-μ-0.5)/σ)²/2)]
```

### 2.2 Position Derivatives

Let Φx = Φ(i | x, σx) and Φy = Φ(j | y, σy).

**X derivative:**
```
∂μ/∂x = N · (∂Φx/∂x) · Φy
      = N · (∂Φx/∂μ) · Φy
```

**Y derivative:**
```
∂μ/∂y = N · Φx · (∂Φy/∂y)
      = N · Φx · (∂Φy/∂μ)
```

### 2.3 Z Derivative (Chain Rule)

The z derivative requires the chain rule through both σx(z) and σy(z):

```
∂μ/∂z = N · [(∂Φx/∂σx)·(dσx/dz)·Φy + Φx·(∂Φy/∂σy)·(dσy/dz)]
```

**Computing dσ/dz:**

First, find dα/dz:
```
dαx/dz = d/dz[1 + ((z-γ)/d)² + Ax((z-γ)/d)³ + Bx((z-γ)/d)⁴]
       = (2/d²)(z-γ) + (3Ax/d³)(z-γ)² + (4Bx/d⁴)(z-γ)³

dαy/dz = (2/d²)(z+γ) + (3Ay/d³)(z+γ)² + (4By/d⁴)(z+γ)³
```

Then, using σ = σ₀√α:
```
dσx/dz = (σx₀/(2√αx)) · dαx/dz
dσy/dz = (σy₀/(2√αy)) · dαy/dz
```

### 2.4 Photon and Background Derivatives

**Photon derivative:**
```
∂μ/∂N = Φx · Φy
```

**Background derivative:**
```
∂μ/∂bg = 1
```

## 3. Fisher Information Matrix

### 3.1 Definition

For Poisson noise, the Fisher Information Matrix is:

```
FI[k,l] = Σᵢⱼ (1/μᵢⱼ) · (∂μᵢⱼ/∂θk) · (∂μᵢⱼ/∂θl)
```

This is a 5×5 symmetric matrix.

### 3.2 Individual Components

Let's denote partial derivatives as:
- gx = ∂μ/∂x
- gy = ∂μ/∂y
- gz = ∂μ/∂z
- gN = ∂μ/∂N
- gb = ∂μ/∂bg

**Diagonal elements:**
```
FI[1,1] = Σᵢⱼ gx²/μ     (x precision)
FI[2,2] = Σᵢⱼ gy²/μ     (y precision)
FI[3,3] = Σᵢⱼ gz²/μ     (z precision)
FI[4,4] = Σᵢⱼ gN²/μ     (photon precision)
FI[5,5] = Σᵢⱼ gb²/μ     (background precision)
```

**Off-diagonal elements** (symmetric):
```
FI[1,2] = Σᵢⱼ (gx·gy)/μ    (x-y correlation)
FI[1,3] = Σᵢⱼ (gx·gz)/μ    (x-z correlation)
FI[1,4] = Σᵢⱼ (gx·gN)/μ    (x-N correlation)
... etc for all pairs
```

### 3.3 Expected Structure

**Key insights:**
1. FI[1,2] ≈ 0: x and y are nearly independent
2. FI[1,3] ≈ 0: x position independent of z
3. FI[2,3] ≈ 0: y position independent of z
4. FI[1,4] ≠ 0: x correlated with photon count
5. FI[3,3] varies strongly with z: z precision depends on dσ/dz

## 4. Cramér-Rao Lower Bound (CRLB)

### 4.1 Definition

The CRLB is the inverse of the Fisher Information Matrix:

```
CRLB = FI⁻¹
```

The diagonal elements give the minimum achievable variance:

```
σ²min(θk) = CRLB[k,k] = (FI⁻¹)[k,k]
```

### 4.2 Parameter Uncertainties

```
σx_min = √(CRLB[1,1])    (minimum x uncertainty)
σy_min = √(CRLB[2,2])    (minimum y uncertainty)
σz_min = √(CRLB[3,3])    (minimum z uncertainty)
σN_min = √(CRLB[4,4])    (minimum photon uncertainty)
σbg_min = √(CRLB[5,5])   (minimum background uncertainty)
```

## 5. Z-Dependence of Localization Precision

### 5.1 Physical Interpretation

The z-precision depends on the magnitude of dσ/dz:

**When |dσx/dz| and |dσy/dz| are large:**
- Small change in z causes large change in PSF shape
- Easy to determine z from PSF shape
- Small CRLB_z (good precision)

**When |dσx/dz| and |dσy/dz| are small:**
- PSF shape changes slowly with z
- Hard to determine z from PSF shape
- Large CRLB_z (poor precision)

### 5.2 Critical Points

For the astigmatic model with γ offset:

**At z ≈ ±γ (focal crossings):**
- One width is at minimum, the other changes rapidly
- |dσ/dz| is moderate but non-zero
- CRLB_z has local maxima (worse precision)

**At z = 0 (center):**
- σx = σy (circular PSF)
- dσx/dz and dσy/dz have opposite signs
- Partial cancellation in total z-information
- CRLB_z is moderate

**At extreme z (±600nm):**
- Both widths large and changing rapidly
- |dσx/dz| and |dσy/dz| both large
- CRLB_z is small (best precision)
- BUT: wide PSF means low SNR, practical limitations

## 6. Implementation Notes

### 6.1 Numerical Stability

When computing FI:
1. Only sum over pixels where μ > 0 (avoid division by zero)
2. Use regularization on FI before inversion if needed
3. Check condition number of FI before inversion

### 6.2 Matrix Inversion

The Fisher Information matrix should be:
- Symmetric
- Positive definite (all eigenvalues > 0)

Use Cholesky decomposition for stable inversion.

### 6.3 Validation

To validate the CRLB calculation:
1. Generate synthetic data with known parameters
2. Fit with MLE many times (different noise realizations)
3. Compute empirical std(fitted - true)
4. Compare empirical std to CRLB

For a correct implementation: **empirical ≈ CRLB** (ratio ≈ 1.0)

## 7. Summary of Key Equations

**Model:**
```
μᵢⱼ = bg + N·Φ(i|x,σx(z))·Φ(j|y,σy(z))
σx(z) = σx₀√αx(z),  αx(z) = 1 + ((z-γ)/d)² + Ax((z-γ)/d)³ + Bx((z-γ)/d)⁴
```

**Fisher Information:**
```
FI[k,l] = Σᵢⱼ (1/μᵢⱼ)·(∂μᵢⱼ/∂θk)·(∂μᵢⱼ/∂θl)
```

**CRLB:**
```
σ²min = diag(FI⁻¹)
```

**Z-derivative:**
```
∂μ/∂z = N[(∂Φx/∂σx)·(dσx/dz)·Φy + Φx·(∂Φy/∂σy)·(dσy/dz)]
dσ/dz = (σ₀/(2√α))·dα/dz
```
