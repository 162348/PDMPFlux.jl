# docs/make.jl
# パッケージルート（Project.toml があるディレクトリ）を LOAD_PATH に追加する。
# `../src` だけを足すと Project.toml が見えず、依存関係チェックで落ちる。
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Documenter, PDMPFlux

# doctest で必要な using をページ全体に仕込む
DocMeta.setdocmeta!(PDMPFlux, :DocTestSetup, :(using PDMPFlux); recursive=true)

makedocs(
    sitename = "PDMPFlux.jl",
    modules  = [PDMPFlux],
    format   = Documenter.HTML(),
    pages = [
        "Home"            => "index.md",
        "Tutorials"       => ["tutorials/quickstart.md", "tutorials/samplers.md", "tutorials/ad-backends.md"],
    ],
    checkdocs = :none,  # ドキュメント化されていない関数の警告を抑制
)

deploydocs(
    repo         = "github.com/162348/PDMPFlux.jl.git",
    devbranch    = "main",
    push_preview = true,
)