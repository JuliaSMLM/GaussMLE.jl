using Documenter
using GaussMLE

# Set up doctests
DocMeta.setdocmeta!(GaussMLE, :DocTestSetup, :(using GaussMLE); recursive=true)

makedocs(
    sitename = "GaussMLE.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://JuliaSMLM.github.io/GaussMLE.jl/stable/",
        edit_link = "main",
        assets = String[],
    ),
    modules = [GaussMLE],
    authors = "klidke@unm.edu",
    repo = "https://github.com/JuliaSMLM/GaussMLE.jl/blob/{commit}{path}#{line}",
    warnonly = [:missing_docs, :cross_references],  # Don't fail on internal docstrings
    pages = [
        "Home" => "index.md",
        "User Guide" => [
            "Getting Started" => "guide/getting_started.md",
            "Models" => "guide/models.md",
            "GPU Support" => "guide/gpu.md",
        ],
        "Examples" => [
            "Basic Fitting" => "examples/basic.md",
            "PSF Width Fitting" => "examples/sigma.md",
            "3D Astigmatic" => "examples/astigmatic.md",
        ],
        "API Reference" => "api.md",
        "Theory" => "theory.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaSMLM/GaussMLE.jl",
    devbranch="main",
)