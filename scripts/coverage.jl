"""
Generate a local coverage report (lcov.info) for PDMPFlux.

Usage:
  julia --project=test scripts/coverage.jl extended
  julia --project=test scripts/coverage.jl coverage

Notes:
  - This script runs the test suite with `coverage=true`, then writes `lcov.info`
    at the repository root.
  - For an HTML report, you can use `genhtml lcov.info -o coverage-html`
    (requires lcov installed on your system).
"""

using Pkg

repo_root = normpath(joinpath(@__DIR__, ".."))
test_env  = joinpath(repo_root, "test")

suite = isempty(ARGS) ? "extended" : ARGS[1]
suite ∈ ("quick", "extended", "coverage") || error("Unknown suite: $suite (expected quick|extended|coverage)")

Pkg.activate(test_env)
Pkg.develop(PackageSpec(path=repo_root))
Pkg.instantiate()

Pkg.test("PDMPFlux"; coverage=true, test_args=[suite])

using Coverage

cov = process_folder(joinpath(repo_root, "src"))
LCOV.writefile(joinpath(repo_root, "lcov.info"), cov)

tot = sum(c -> count(!isnothing, c.coverage), cov)
hit = sum(c -> count(x -> (x isa Integer) && (x > 0), c.coverage), cov)
pct = tot == 0 ? 0.0 : 100 * hit / tot

println("Wrote lcov.info")
println("Line coverage: $(round(pct; digits=2))% ($hit / $tot)")


