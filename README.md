# PDMPFlux

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://162348.github.io/PDMPFlux.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://162348.github.io/PDMPFlux.jl/dev/)
[![Build Status](https://github.com/162348/PDMPFlux.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/162348/PDMPFlux.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/162348/PDMPFlux.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/162348/PDMPFlux.jl)

This repository contains a [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) implementation of the PDMP samplers.

Currently, only Zig-Zag samplers are implemented.

## Examples


```julia
using PDMPFlux

N_sk = 1_000_000 # number of skeleton points
N = 1_000_000 # number of samples

function U_banana(x::Vector{Float64})
    mean_x2 = (x[1]^2 - 1)
    return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
end

dim = 50
xinit = ones(dim)
vinit = ones(dim)
grid_size = 0  # use constant bounds

sampler = ZigZag(dim, grad_U, grid_size=grid_size)  # initialize your Zig-Zag sampler
output = sample_skeleton(sampler, N_sk, xinit, vinit, verbose = true)  # simulate skeleton points
samples = sample_from_skeleton(sampler, N, output)  # get samples from the skeleton points

jointplot(samples, coordinate_numbers=[2,1])
plot_traj(output, 10000)
diagnostic(output)
```

## Gallery

![](assets/banana_density.svg)

![](assets/banana_jointplot.svg)

![](assets/Cauchy1D.gif)

## Remarks

- The implementation of the PDMP samplers is based on the paper [Andral and Kamatani (2024) Automated Techniques for Efficient Sampling of Piecewise-Deterministic Markov Processes](https://arxiv.org/abs/2408.03682) and its implementation in [`pdmp_jax`](https://github.com/charlyandral/pdmp_jax).
- `pdmp_jax` has 

## References

* [`pdmp_jax.jl`](https://github.com/charlyandral/pdmp_jax): This repository is based on this repository.
* [Andral and Kamatani (2024) Automated Techniques for Efficient Sampling of Piecewise-Deterministic Markov Processes](https://arxiv.org/abs/2408.03682)
* [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) is used for automatic differentiation.
