# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
```bash
# Run all tests
julia --project test/runtests.jl

# Using Pkg (from Julia REPL)
julia --project -e 'using Pkg; Pkg.test()'
```

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

GaussMLE.jl implements Maximum Likelihood Estimation for Gaussian blob parameters in 2D images under Poisson noise. The package follows a modular architecture:

### Core Components

1. **Models** (`src/models/`): Defines abstract `GaussModel` type and concrete implementations:
   - `GaussXyNb`: Basic 2D Gaussian with (x, y, intensity, background) parameters
   - `GaussXyNbS`: Extended model adding PSF width parameter
   
2. **Fitting Engine** (`src/fit/`): Implements Newton-Raphson optimization:
   - `fitstack`: Main entry point accepting 3D data arrays (x, y, frames)
   - Supports variance-weighted fitting for sCMOS cameras
   - Returns fitted parameters and Cramér-Rao Lower Bound uncertainties

3. **Mathematical Library** (`src/gausslib/`): Low-level Gaussian calculations:
   - Derivatives and Hessian computation
   - Specialized matrix operations optimized for small matrices

4. **Simulation** (`src/sim/`): Generates synthetic data for testing and validation

### Key Design Patterns

- Abstract type hierarchy allows extending with new Gaussian models
- Newton-Raphson solver works generically with any `GaussModel` subtype
- CUDA support integrated throughout for GPU acceleration
- Variance maps enable proper handling of sCMOS camera noise characteristics

### Performance Considerations

- Optimized for fitting many small ROIs in parallel (typical SMLM workflow)
- Matrix operations specialized for 2x2 to 5x5 sizes
- Pre-allocated arrays minimize garbage collection
- GPU dispatch available when CUDA.jl is loaded