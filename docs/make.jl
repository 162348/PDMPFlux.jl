# docs/make.jl
push!(LOAD_PATH, "../src")  # パッケージ本体を見えるように
using Documenter, PDMPFlux

# doctest で必要な using をページ全体に仕込む
DocMeta.setdocmeta!(PDMPFlux, :DocTestSetup, :(using PDMPFlux); recursive=true)

makedocs(
    sitename = "PDMPFlux.jl",
    modules  = [PDMPFlux],
    format   = Documenter.HTML(),
    pages = [
        "Home"            => "index.md",
        "Tutorials"       => ["tutorials/quickstart.md"],
    ],
    checkdocs = :none,  # ドキュメント化されていない関数の警告を抑制
)

deploydocs(
    repo         = "github.com/162348/PDMPFlux.jl.git",
    devbranch    = "main",
    push_preview = true,
)