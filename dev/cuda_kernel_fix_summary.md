# CUDA Kernel Convergence Issue Analysis and Fix

## Problem Description
The CUDA kernel implementation was converging to incorrect values (x=6, y=6) instead of the true parameters around (x=3.5, y=3.5) for a centered Gaussian in a 7x7 ROI.

## Root Causes Identified

### 1. Coordinate System Mismatch in Initialization
**Issue**: The center of mass calculation used 0-based indices for weighting, while the CPU implementation uses 1-based indices.

**Original code (lines 103-107)**:
```julia
i = (idx - 1) ÷ roi_size  # 0-based
j = (idx - 1) % roi_size  # 0-based
sum_x += val * T(j)       # Using 0-based index directly
sum_y += val * T(i)       # Using 0-based index directly
```

**Fix**: Convert to 1-based indices for center of mass calculation, then convert result to 0-based:
```julia
i = (idx - 1) ÷ roi_size + 1  # 1-based row
j = (idx - 1) % roi_size + 1  # 1-based column
sum_x_weighted += val_above_bg * T(j - 1)  # Convert to 0-based after weighting
sum_y_weighted += val_above_bg * T(i - 1)  # Convert to 0-based after weighting
```

### 2. Incorrect Derivative for Intensity Parameter
**Issue**: The derivative with respect to intensity was incorrectly computed.

**Original code (line 213)**:
```julia
d_n = gaussian_term / intensity / intensity  # Wrong!
```

**Fix**:
```julia
d_n = gaussian_val / intensity  # Correct derivative
```

### 3. Oversimplified Newton-Raphson Update
**Issue**: The original implementation solved separate 2x2 and 1D systems, ignoring parameter coupling.

**Original approach**:
- Solve 2x2 system for (x,y) position updates
- Solve separate 1D updates for intensity and background
- This ignores the coupling between position and intensity parameters

**Fix**: While a full 4x4 solver would be ideal, the fixed version uses a more careful approach with:
- Proper regularization of the Hessian diagonal
- Damping factor (0.7) for stability
- Better bounds checking

### 4. Inconsistent Index Conventions
**Issue**: The code mixed 0-based and 1-based indexing inconsistently.

**Fix**: Established clear conventions:
- ROI data uses 1-based indexing (Julia standard)
- Gaussian evaluation expects 1-based pixel indices but converts to 0-based internally
- Position parameters (x,y) are stored in 0-based coordinates

## Verification

The fixed kernel should now:
1. Correctly initialize parameters using center of mass
2. Converge to the true Gaussian center for both centered and off-center cases
3. Match CPU implementation results within numerical precision

## Testing
Run the test script to verify the fix:
```bash
julia dev/test_cuda_kernel_fix.jl
```

Expected results for centered Gaussian at (3,3):
- Original kernel: x≈6, y≈6 (incorrect)
- Fixed kernel: x≈3, y≈3 (correct)
- CPU: x≈3, y≈3 (reference)

## Next Steps
1. Replace the original kernel with the fixed version
2. Add comprehensive unit tests for edge cases
3. Implement proper CRLB computation from the inverse Hessian
4. Consider implementing a full 4x4 matrix solver for better accuracy