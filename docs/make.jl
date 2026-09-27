using Documenter
using GaussMLE

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
    repo = Remotes.GitHub("JuliaSMLM", "GaussMLE.jl"),
    doctest = false,  # QA runs every docstring jldoctest (admiral decision 0018); pages use @example
    checkdocs = :exports,  # every exported docstring is in the manual (admiral decision 0018)
    # Opt-out: GaussLib is the internal legacy reference implementation. It exports its helpers
    # to GaussMLE only, and GaussMLE does not re-export them, so they are not public API.
    checkdocs_ignored_modules = [GaussMLE.GaussLib],
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
    repo = "github.com/JuliaSMLM/GaussMLE.jl",
    devbranch = "main",
)
