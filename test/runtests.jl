"""
`Pkg.test()` entrypoint.

This file acts as a thin dispatcher so CI can select suites via `test_args`,
e.g. `Pkg.test(test_args=["quick"])` / `Pkg.test(test_args=["extended"])`.
"""

if "quick" in ARGS || "--quick" in ARGS
    include("runtests_quick.jl")
elseif "extended" in ARGS
    include("runtests_extended.jl")
else
    # Default to the extended suite to keep `Pkg.test()` comprehensive.
    include("runtests_extended.jl")
end