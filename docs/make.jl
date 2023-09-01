using GaussMLE
using Documenter

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
        "GaussMLE" => "index.md",
        "GaussFit" => "GaussFit.md",
        "GaussModel" => "GaussModel.md",
        "GaussLib" => "GaussLib.md",
        "GaussSim" => "GaussSim.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaSMLM/GaussMLE.jl",
    devbranch="main",
)