# Development Guidelines for dev/

This directory is for informal development testing, experimentation, and concept exploration. It provides a sandbox for testing ideas and interfaces outside of formal testing.

## Purpose
- Informal testing and experimentation
- Concept and interface exploration
- Quick feedback during development
- Not part of formal test suite

## Environment Setup

### Dev Environment
- Each file starts with: `using Pkg; Pkg.activate("dev")`
- Has its own environment with development tools:
  - CairoMakie (for plotting - NOT Plots.jl)
  - BenchmarkTools
  - Revise
  - Other development-specific packages

### Running Dev Files (For Code Agents/Claude)

Since you run Julia in separate processes (not an interactive REPL), use this command:
```bash
julia --project=. dev/my_experiment.jl
```

**Important**: The dev files handle their own environment activation, so:
1. Run from the project root with `--project=.`
2. The file itself will activate the dev environment
3. Always capture stdout to see println output
4. Read generated PNG files after execution

## Creating New Dev Files

When creating new files in the dev/ directory, follow these conventions:

### File Structure
1. **Always start with environment activation**:
   ```julia
   using Pkg
   Pkg.activate("dev")
   ```

2. **Define output directory and adjustable parameters**:
   ```julia
   # Parameters (adjust these as needed)
   n_iterations = 1000
   plot_size = (800, 600)
   seed = 42
   
   # Output directory
   output_dir = joinpath(@__DIR__, "output")
   mkpath(output_dir)  # Create if it doesn't exist
   ```

3. Test a single concept or interface per file
4. Use descriptive filenames (e.g., `test_new_api.jl`, `benchmark_parser.jl`)
5. **Save all outputs using the `output_dir` variable**:
   ```julia
   save(joinpath(output_dir, "results.png"), fig)
   ```

## Output Conventions

### Output Types
- **Console output**: Use `println()` for text feedback
- **Figures**: Save as PNG files using CairoMakie
- **Text files**: Only if absolutely necessary (txt or md format)

### Output Location
- All outputs go to `dev/output/` or subfolders
- The `dev/output/` directory is gitignored
- Create subfolders for organization if needed

### For Code Agents/LLMs

#### Execution Workflow
1. Run the dev file from project root:
   ```bash
   julia --project=. dev/experiment.jl
   ```
2. Capture and analyze stdout output
3. Check `dev/output/` for generated PNG files
4. Read and interpret visual results

#### Example Figure Generation
```julia
# In dev file (after parameters and output_dir setup)
using CairoMakie
fig = Figure(size=plot_size)  # Use parameter
# ... plotting code ...
save(joinpath(output_dir, "results.png"), fig)
println("Figure saved to $(joinpath(output_dir, "results.png"))")
```

#### After Running
- Read the console output for results and messages
- Use Read tool on PNG files to view plots
- Interpret visual data to provide feedback

## Best Practices
- Keep files focused on single concepts
- Use clear, descriptive output messages
- Save all generated figures for later reference
- Document findings with println statements
- Clean up old outputs periodically

## Current Development Files

### Astigmatic PSF Validation (Validated Parameters)
The astigmatic PSF model uses these validated parameters:
- σx₀ = 1.3, σy₀ = 1.3 (base widths, pixels)
- Ax = 0.05, Ay = -0.05 (cubic aberrations)
- Bx = 0.01, By = -0.01 (quartic aberrations)
- γ = 200.0 nm (focal plane offset)
- d = 500.0 nm (depth scale)

These parameters provide:
- Flat CRLB across z-range (±600 nm)
- Perfect empirical/CRLB agreement (X: 0.985, Y: 0.999, Z: 1.000)
- All tests pass with strict 10% tolerance

### Key Files
- **astig_theory/** - Complete theoretical derivation and validation framework
  - `astigmatic_fisher_derivation.md` - Fisher Information derivation
  - `compute_theoretical_crlb.jl` - Standalone CRLB calculator
  - `compare_theory_vs_gausslib.jl` - Theory vs production validation
  - `README.md` - Documentation

- **analyze_astigmatic_crlb_vs_z.jl** - Primary CRLB validation tool (1000 fits/z)
- **plot_astigmatic_psf_widths.jl** - PSF width visualization across z-range