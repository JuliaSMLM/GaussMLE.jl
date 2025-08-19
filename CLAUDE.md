# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development References

- Look in `.claude/ref/package_structure.md` for design reference

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

# Run GPU tests
GAUSSMLE_TEST_GPU=true julia --project test/gpu_tests.jl
```

## Architecture

[Rest of the existing content remains unchanged]
- smite ref code here: https://github.com/LidkeLab/smite/tree/main/MATLAB/source/cuda