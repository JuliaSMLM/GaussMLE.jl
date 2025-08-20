# Astigmatic PSF Model Documentation

## Overview

The astigmatic PSF model (`AstigmaticXYZNB`) implements a 3D localization model using engineered astigmatism, commonly achieved with a cylindrical lens in the optical path. This creates elliptical PSFs whose orientation and eccentricity encode the axial (z) position of the emitter.

## Mathematical Model

### PSF Width Modulation

The PSF widths in x and y directions vary with z position according to:

```
σx(z) = σx₀ × √(αx(z))
σy(z) = σy₀ × √(αy(z))
```

where the width scaling factors are:

```
αx(z) = 1 + ((z - γ)/d)² + Ax × ((z - γ)/d)³ + Bx × ((z - γ)/d)⁴
αy(z) = 1 + ((z + γ)/d)² + Ay × ((z + γ)/d)³ + By × ((z + γ)/d)⁴
```

### Model Parameters

The `AstigmaticXYZNB` model has 8 calibration parameters:

1. **σx₀, σy₀** (pixels): Base PSF widths at the focal plane
   - Typical values: 1.2-1.5 pixels
   - These define the minimum PSF width

2. **Ax, Ay** (dimensionless): Cubic aberration coefficients
   - Control asymmetric PSF expansion
   - For astigmatism: Ax = -Ay (opposite signs)
   - Typical values: ±0.3 to ±0.8

3. **Bx, By** (dimensionless): Quartic aberration coefficients
   - Fine-tune the PSF shape at large defocus
   - For astigmatism: Bx = -By (opposite signs)
   - Typical values: ±0.05 to ±0.2

4. **γ** (nm): Astigmatism offset
   - Shifts the focal planes for x and y
   - γ = 0: Symmetric astigmatism
   - γ ≠ 0: Asymmetric focal plane positions

5. **d** (nm): Depth scaling parameter
   - Controls the z-range of the PSF model
   - Typical values: 400-600 nm
   - Larger d → wider z-range but less z-resolution

### Fitting Parameters

For each single-molecule spot, the model fits 5 parameters:

1. **x, y** (pixels): Lateral position
2. **z** (nm): Axial position
3. **N** (photons): Total photon count
4. **bg** (photons/pixel): Background level

## Physical Interpretation

### Astigmatism Behavior

- **z < 0**: PSF elongated in y-direction (σy > σx)
- **z = 0**: PSF approximately circular (σx ≈ σy)
- **z > 0**: PSF elongated in x-direction (σx > σy)

### Key Relationships

1. **PSF Eccentricity**: The ratio σx/σy encodes z position
2. **Total PSF Area**: Proportional to σx × σy
3. **Peak Intensity**: Inversely proportional to PSF area

## Calibration Requirements

### Experimental Calibration

1. **Bead Scan**: Image fluorescent beads at known z-positions
2. **Extract PSF Widths**: Measure σx and σy at each z
3. **Fit Model**: Determine Ax, Ay, Bx, By, γ, d parameters
4. **Validate**: Check residuals and z-localization precision

### Typical Parameter Values

For a standard SMLM setup with 100× objective and cylindrical lens:

```julia
psf_model = AstigmaticXYZNB{Float32}(
    1.3f0, 1.3f0,  # σx₀, σy₀: base widths
    0.5f0, -0.5f0,  # Ax, Ay: opposite cubic terms
    0.1f0, -0.1f0,  # Bx, By: opposite quartic terms
    0.0f0,          # γ: no offset
    500.0f0         # d: 500 nm depth scale
)
```

## Implementation Details

### Parameter Ordering

**Critical**: The internal parameter order differs from the display order:

- **Model expects**: [x, y, z, N, bg] (z is parameter 3)
- **GaussLib functions expect**: [x, y, N, bg, z] (z is parameter 5)

This requires careful parameter reordering in derivative calculations.

### Derivative Computation

The z-derivative involves the chain rule through the width functions:

```
∂μ/∂z = N × [PSFy × (∂PSFx/∂σx)(∂σx/∂z) + PSFx × (∂PSFy/∂σy)(∂σy/∂z)]
```

where:
- ∂σx/∂z involves derivatives of αx(z)
- ∂PSFx/∂σx is the derivative of the integrated Gaussian

### Newton-Raphson Updates

The fitting uses diagonal (scalar) Newton-Raphson:
- Each parameter updated independently
- Works well when parameters are weakly coupled
- Requires proper scaling of derivatives

## Common Issues and Solutions

### Issue 1: Z-fitting Stuck at Initial Value

**Symptoms**: Z parameter doesn't update from initial guess

**Possible Causes**:
1. Parameter ordering mismatch in derivatives
2. Incorrect derivative calculation
3. Too restrictive constraints
4. Poor initialization

**Solutions**:
1. Verify parameter order consistency
2. Check derivative calculations numerically
3. Ensure constraints allow z movement
4. Improve z initialization from PSF shape

### Issue 2: Infinite Uncertainties

**Symptoms**: All CRLB uncertainties are Inf

**Possible Causes**:
1. Singular Fisher Information matrix
2. Zero or negative model values
3. Incorrect derivative calculation

**Solutions**:
1. Add regularization to Fisher matrix
2. Ensure model > 0 everywhere
3. Validate derivatives numerically

### Issue 3: Poor Z-resolution

**Symptoms**: Large z-localization uncertainty

**Possible Causes**:
1. Insufficient astigmatism (Ax, Ay too small)
2. Too large depth scale (d)
3. Low photon count

**Solutions**:
1. Increase cylindrical lens strength
2. Optimize d parameter
3. Increase labeling density or exposure

## Best Practices

1. **Calibration**: Always calibrate with beads in same medium as samples
2. **Z-range**: Limit fitting to calibrated z-range (typically ±600 nm)
3. **Validation**: Check calibration with known 3D structures
4. **Photon threshold**: Require minimum photons for reliable z-fitting
5. **Drift correction**: Apply 3D drift correction for long acquisitions

## References

1. Huang et al., "Three-dimensional super-resolution imaging by stochastic optical reconstruction microscopy", Science (2008)
2. Babcock et al., "A high-density 3D localization algorithm for stochastic optical reconstruction microscopy", Optical Nanoscopy (2012)
3. Li et al., "Real-time 3D single-molecule localization using experimental point spread functions", Nature Methods (2018)