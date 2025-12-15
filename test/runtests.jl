"""
`Pkg.test()` entrypoint.

This file acts as a thin dispatcher so CI can select suites via `test_args`,
e.g. `Pkg.test(test_args=["quick"])` / `Pkg.test(test_args=["extended"])`.
"""

if "quick" in ARGS || "--quick" in ARGS
    include("runtests_quick.jl")
elseif "extended" in ARGS
    include("runtests_extended.jl")
elseif "coverage" in ARGS
    # Coverage-oriented suite: run the extended suite plus extra tests aimed at
    # exercising otherwise-unhit branches/paths.
    include("runtests_extended.jl")
    include("test_coverage.jl")
else
    # Default to the extended suite to keep `Pkg.test()` comprehensive.
    include("runtests_extended.jl")
end