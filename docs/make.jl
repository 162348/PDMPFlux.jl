using PDMPFlux
using Documenter

DocMeta.setdocmeta!(PDMPFlux, :DocTestSetup, :(using PDMPFlux); recursive=true)

makedocs(;
    modules=[PDMPFlux],
    authors="Hirofumi Shiba",
    sitename="PDMPFlux.jl",
    format=Documenter.HTML(;
        canonical="https://162348.github.io/PDMPFlux.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/162348/PDMPFlux.jl",
    devbranch="main",
)
