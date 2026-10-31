# Examples Guidelines for examples/

This directory contains user-facing examples demonstrating how to use the package. These examples serve as both documentation and practical guides for users after development is complete.

## Purpose
- Demonstrate user interfaces and APIs
- Provide practical usage examples
- Serve as living documentation
- Help users understand package capabilities

## Environment Setup

### Examples Environment
- The four short scripts (`basic_fitting.jl`, `scmos_camera.jl`, `gpu_acceleration.jl`,
  `astigmatic_3d.jl`) use the repository's active project without environment activation.
- Extended examples activate `examples/` with `using Pkg; Pkg.activate(@__DIR__)`.
  That environment has example-specific packages:
  - CairoMakie (for plotting - NOT Plots.jl)
  - Any packages users might commonly use alongside this package
  - Visualization and output tools

### Running Example Files (For Code Agents/Claude)

Since you run Julia in separate processes (not an interactive REPL), use this command:
```bash
julia --project=. examples/my_example.jl
```

**Important**:
1. Run from the project root with `--project` (or `--project=.`)
2. The four short scripts keep that project active; extended examples activate `examples/`
3. Always capture stdout to see output
4. Read generated PNG files after execution

## Creating New Example Files

When creating new files in the examples/ directory, follow these conventions:

### File Structure
1. **Use the active project for short examples; activate for extra dependencies**:
   ```julia
   using Pkg
   Pkg.activate(@__DIR__)
   ```

2. **Define output directory and adjustable parameters**:
   ```julia
   # Example parameters (users can adjust these)
   data_size = 100
   output_format = "png"
   verbose = true
   
   # Output directory
   output_dir = joinpath(@__DIR__, "output")
   mkpath(output_dir)  # Create if it doesn't exist
   ```

3. Demonstrate a specific use case or feature
4. Include explanatory comments for users
5. Use descriptive filenames (e.g., `basic_usage.jl`, `advanced_features.jl`)
6. **Save all outputs using the `output_dir` variable**:
   ```julia
   save(joinpath(output_dir, "example_plot.png"), fig)
   ```

## Output Conventions

### Output Types
- **Console output**: Use `println()` to explain what's happening
- **Figures**: Save as PNG files using CairoMakie
- **Text files**: Only if demonstrating file I/O features

### Output Location
- All outputs go to `examples/output/` or subfolders
- The `examples/output/` directory is gitignored
- Create subfolders for different example categories

### For Code Agents/LLMs

#### Execution Workflow
1. Run the example file from project root:
   ```bash
   julia --project=. examples/example_name.jl
   ```
2. Capture and analyze stdout output
3. Check `examples/output/` for generated files
4. Verify the example runs successfully

#### Example Structure
```julia
# In example file (after parameters and output_dir setup)
println("=== Example: Basic Usage ===")
println("This example demonstrates...")
println("Parameters: data_size=$data_size, verbose=$verbose")

# Actual example code
result = some_function(n=data_size)
verbose && println("Result: ", result)

# If generating figures
using CairoMakie
fig = Figure()
# ... plotting code ...
output_file = joinpath(output_dir, "example_plot.$output_format")
save(output_file, fig)
println("Plot saved to $output_file")
```

## Best Practices
- Write examples as if teaching a new user
- Include comments explaining each step
- Show both simple and advanced usage
- Demonstrate error handling where appropriate
- Keep examples self-contained and runnable
- Test that examples work with a fresh environment

## Current Examples
The examples/ directory currently contains:
- `basic_fitting.jl` - Basic fitting with `smld, info = fit(data, fitter)`
- `scmos_camera.jl` - Calibrated sCMOS `ROIBatch` with full-sensor readnoise
- `gpu_acceleration.jl` - CPU/GPU comparison with `backend` (CUDA or AMDGPU required)
- `astigmatic_3d.jl` - Astigmatic 3D fitting
