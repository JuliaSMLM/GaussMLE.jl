# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development References

- Look in `.claude/ref/package_structure.md` for design reference

## Commands

### Testing
```bash
# Run all tests (GPU auto-detected)
julia --project -e 'using Pkg; Pkg.test()'
```

GPU tests run automatically when a CUDA GPU is detected.

### Documentation
```bash
# Build documentation
julia --project=docs docs/make.jl

# Serve documentation locally
julia --project=docs -e 'using LiveServer; serve(dir="docs/build")'
```

### Development
```bash
# Activate development environment
julia --project=dev

# Run example fitting scripts
julia --project=dev dev/basicfit.jl
julia --project=dev dev/sigmafit.jl
```

## Architecture

### Core Implementation

The package uses a unified kernel approach via KernelAbstractions.jl:
- Single implementation works on both CPU and GPU
- StaticArrays for stack allocation and performance
- Custom GPU-compatible linear algebra (LU decomposition)
- Scalar Newton-Raphson with diagonal Hessian for optimization
- Full Fisher Information Matrix for CRLB uncertainty estimation

### Source Structure

- `src/unified_kernel.jl` - Main fitting kernel (CPU/GPU)
- `src/api.jl` - High-level user API
- `src/psf_models.jl` - PSF model definitions
- `src/camera_models.jl` - Camera noise models
- `src/devices.jl` - Device abstraction (CPU/GPU)
- `src/results.jl` - Result types and conversions
- `src/roi_batch.jl` - ROI batch data structures
- `src/simulator.jl` - Data generation for testing

### Reference Implementation

Original MATLAB/CUDA implementation: https://github.com/LidkeLab/smite/tree/main/MATLAB/source/cuda