# PDMPFlux

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://162348.github.io/PDMPFlux.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://162348.github.io/PDMPFlux.jl/dev/)
[![Build Status](https://github.com/162348/PDMPFlux.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/162348/PDMPFlux.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/162348/PDMPFlux.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/162348/PDMPFlux.jl)

This repository contains a [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) implementation of the PDMP samplers.

## Examples

## Remarks

- The implementation of the PDMP samplers is based on the paper [Andral and Kamatani (2024) Automated Techniques for Efficient Sampling of Piecewise-Deterministic Markov Processes](https://arxiv.org/abs/2408.03682) and its implementation in [`pdmp_jax`](https://github.com/charlyandral/pdmp_jax).
- `pdmp_jax` has 

## References

* [`pdmp_jax.jl`](https://github.com/charlyandral/pdmp_jax): This repository is based on this repository.
* [Andral and Kamatani (2024) Automated Techniques for Efficient Sampling of Piecewise-Deterministic Markov Processes](https://arxiv.org/abs/2408.03682)
* [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) is used for automatic differentiation.
