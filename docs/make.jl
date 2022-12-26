using GaussMLE
using Documenter

DocMeta.setdocmeta!(GaussMLE, :DocTestSetup, :(using GaussMLE); recursive=true)

makedocs(;
    modules=[GaussMLE],
    authors="klidke@unm.edu",
    repo="https://github.com/JuliaSMLM/GaussMLE.jl/blob/{commit}{path}#{line}",
    sitename="GaussMLE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaSMLM.github.io/GaussMLE.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaSMLM/GaussMLE.jl",
    devbranch="main",
)
