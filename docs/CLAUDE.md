# Documentation Guidelines for docs/

This directory contains the package documentation using Documenter.jl. Follow these conventions when creating or updating documentation.

## Directory Structure

```
docs/
├── Project.toml      # Documentation-specific dependencies
├── make.jl          # Build configuration
├── src/             # Markdown source files
│   ├── index.md     # Home page
│   ├── api.md       # API reference
│   └── *.md         # Other topic pages
└── build/           # Generated documentation (gitignored)
```

## Setting Up Documentation

### docs/Project.toml
Create a separate environment for documentation with these dependencies:
```toml
[deps]
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
GaussMLE = "..."  # Your package UUID
```

Optional dependencies for enhanced features:
- `DocumenterInterLinks` - Cross-reference other packages
- `DocumenterTools` - Deployment utilities

### docs/make.jl Structure

Basic template:
```julia
using Documenter
using GaussMLE

# Optional: Set up doctests
DocMeta.setdocmeta!(GaussMLE, :DocTestSetup, :(using GaussMLE); recursive=true)

makedocs(
    sitename = "GaussMLE.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://JuliaSMLM.github.io/GaussMLE.jl/stable/",
    ),
    modules = [GaussMLE],
    pages = [
        "Home" => "index.md",
        "User Guide" => "guide.md",
        "API Reference" => "api.md",
    ],
)

# Deploy docs (usually from CI)
deploydocs(
    repo = "github.com/JuliaSMLM/GaussMLE.jl.git",
    devbranch = "main",
)
```

## Creating Documentation Pages

### Home Page (index.md)
```markdown
# GaussMLE.jl

Brief description of your package.

## Installation

```julia
using Pkg
Pkg.add("GaussMLE")
```

## Quick Start

```@example
using GaussMLE
# Simple example
result = main_function()
```
```

### API Reference (api.md)

For simple packages:
```markdown
# API Reference

```@index
```

```@autodocs
Modules = [GaussMLE]
```
```

For complex packages with public/internal separation:
```markdown
# API Reference

## Public API

```@docs
GaussMLE.fitstack
GaussMLE.GaussXyNb
GaussMLE.GaussXyNbS
```

## Internal API

```@autodocs
Modules = [GaussMLE]
Public = false
```
```

## Writing Documentation

### Code Examples

#### Use @example blocks for demonstrations:
```markdown
```@example
using GaussMLE
data = generate_data()
result = process(data)
println("Result: ", result)
```
```

#### Use jldoctest for verified examples:
```markdown
```jldoctest
julia> using GaussMLE

julia> add(2, 3)
5
```
```

### Best Practices

1. **Doctests**:
   - Place in docstrings for verification
   - Avoid RNG without explicit seeds
   - Use filters for variable output

2. **Examples**:
   - Use @example for complex demonstrations
   - Name blocks to share state: `@example myexample`
   - Hide setup code with `# hide` comments

3. **Cross-references**:
   - Link to functions: `[`GaussMLE.function`](@ref)`
   - Link to sections: `[Installation](@ref)`
   - External links with DocumenterInterLinks

4. **Math and Code**:
   - Use single backticks for inline code: `function_name`
   - Use double backticks for LaTeX: ``α = 1``
   - Prefer Unicode over LaTeX escapes

## Building Documentation

### Local Build
```bash
cd docs
julia --project=. make.jl
```

The documentation will be in `docs/build/`.

### Local Development
For live-reload during development:
```julia
using LiveServer
cd("docs")
servedocs()
```

### CI Integration
Documentation typically builds and deploys via GitHub Actions:
1. Build on pull requests (without deployment)
2. Deploy to gh-pages branch when merging to main

## Common Patterns

### Organizing Complex Documentation
For larger packages, organize pages hierarchically:
```julia
pages = [
    "Home" => "index.md",
    "User Guide" => [
        "Getting Started" => "guide/start.md",
        "Advanced Usage" => "guide/advanced.md",
    ],
    "Examples" => [
        "Basic Examples" => "examples/basic.md",
        "Advanced Examples" => "examples/advanced.md",
    ],
    "API Reference" => "api.md",
]
```

### Including External Files
To include Julia files from examples/:
```markdown
```@example
include("../../examples/demo.jl")
```
```

### Custom Styling
Add custom CSS/JS in make.jl:
```julia
format = Documenter.HTML(
    assets = ["assets/custom.css"],
)
```

## Troubleshooting

- **Missing docstrings**: Set `checkdocs = :none` in makedocs()
- **Broken doctests**: Update examples or use `doctest = false`
- **Build failures**: Check docs/Project.toml dependencies
- **Cross-references not working**: Ensure proper @ref syntax

## GaussMLE-Specific Documentation

The GaussMLE.jl documentation should cover:
1. **Theory**: Mathematical background on Gaussian MLE fitting
2. **Models**: Explanation of GaussXyNb and GaussXyNbS models
3. **GPU Support**: How to enable and use GPU acceleration
4. **sCMOS Support**: Using variance maps for sCMOS cameras
5. **Performance**: Benchmarking results and optimization tips