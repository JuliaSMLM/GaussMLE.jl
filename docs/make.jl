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
        "GaussLib Module" => "GaussLib.md",
        "GaussSim Module" => "GaussSim.md",
        "GaussFit Module" => "GaussFit.md",
        "GaussModel Module" => "GaussModel.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaSMLM/GaussMLE.jl",
    devbranch="main",
)