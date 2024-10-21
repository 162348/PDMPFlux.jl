# PDMPFlux

| Documentation | Workflows | Code Coverage | Quality Assurance |
|:-------------:|:---------:|:-------------:|:-----------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://162348.github.io/PDMPFlux.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://162348.github.io/PDMPFlux.jl/dev/) | [![Build Status](https://github.com/162348/PDMPFlux.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/162348/PDMPFlux.jl/actions/workflows/CI.yml?query=branch%3Amain) | [![Coverage](https://codecov.io/gh/162348/PDMPFlux.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/162348/PDMPFlux.jl) | [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) |

## Overview

`PDMPFlux.jl` provides a fast and efficient implementation of **Piecewise Deterministic Markov Process (PDMP)** samplers using a grid-based Poisson thinning approach.

In this version `v0.2.0`, only Zig-Zag samplers are implemented. We will extend the functionality to include other PDMP samplers in the future.

### Key Features

* To sample from a distribution $p(x)$, the *only* required inputs are its dimension $d$ and the negative log density $U(x)=-\log p(x)$ (up to constant).


## Motivation

Markov Chain Monte Carlo (MCMC) methods are standard in sampling from distributions with unknown normalizing constants.

However, PDMPs offer a promising alternative due to their continuous and non-reversible dynamics, particularly in high-dimensional and big data contexts, as discussed in [Bouchard-Côté et. al. (2018)](https://arxiv.org/abs/1510.02451) and [Bierkens et. al. (2019)](https://arxiv.org/abs/1607.03188).

Despite their potential, practical applications of PDMPs remain limited by a lack of efficient and flexible implementations.

`PDMPFlux.jl` is my attempt to fill this gap, with the aid of the existing automatic differentiation engines.

## Installation

Currently, `julia >= 1.11` is required for compatibility.

To install the package, use Julia's package manager:

```julia-repl
(@v1.11) pkg> add PDMPFlux
```

## Usage

### Basic

The following example demonstrates how to sample from a standard Gaussian distribution using a Zig-Zag sampler.

```julia
using PDMPFlux

function U_Gauss(x::Vector)
    return sum(x.^2) / 2
end

dim = 10
sampler = ZigZagAD(dim, U_Gauss)

N_sk, N, xinit, vinit = 1_000_000, 1_000_000, zeros(dim), ones(dim)
samples = sample(sampler, N_sk, N, xinit, vinit, seed=2024)

jointplot(samples)
```

### Advanced

For more control, you can manually provide the gradient.

Also, by breaking down the `sample()` function into two steps: `sample_skeleton()` and `sample_from_skeleton()`, you can use `plot_traj()` and `diagnostic()` functions to diagnose the sampler:

```julia
using PDMPFlux
using Zygote

N_sk = 1_000_000 # number of skeleton points
N = 1_000_000 # number of samples

function ∇U_banana(x::Vector)
    mean_x2 = (x[1]^2 - 1)
    return - (- x[1] + -(x[2] - mean_x2) - sum(x[3:end]))  # don't forget the minus sign!
end

dim = 50
xinit = ones(dim)
vinit = ones(dim)
grid_size = 0  # use constant bounds

sampler = ZigZag(dim, ∇U_banana, grid_size=grid_size)  # manually providing the gradient
output = sample_skeleton(sampler, N_sk, xinit, vinit, verbose = true)  # simulate skeleton points
samples = sample_from_skeleton(sampler, N, output)  # get samples from the skeleton points

plot_traj(output, 10000)
diagnostic(output)

jointplot(samples)
```

## Gallery

<table>
    <tbody>
        <tr>
            <td style="width: 25%;"><img src="examples/Funnel/Funnel_GroundTruthSamples.svg"></td>
            <td style="width: 25%;"><img src="examples/Funnel/ZigZag_Funnel2D_trajectory.svg"></td>
            <td style="width: 25%;"><img src="examples/Funnel/ZigZag_Funnel2D.gif"></td>
            <td style="width: 25%;"><img src="examples/Funnel/ZigZag_Funnel3D.gif"></td>
        </tr>
        <tr>
            <td align="center"><a href="examples/ZigZag_Funnel3D.jl"><sup>2D</sup> Funnel Distribution (Ground Truth)</a></td>
            <td align="center"><a href="examples/ZigZag_Funnel3D.jl"><sup>2D</sup> Zig-Zag Trajectory (T<sub>max</sub>=10000)</a></td>
            <td align="center"><a href="examples/ZigZag_Funnel2D.jl"><sup>2D</sup> Zig-Zag on Funnel</a></td>
            <td align="center"><a href="examples/ZigZag_Funnel3D.jl"><sup>3D</sup> Zig-Zag on Funnel</a></td>
        </tr>
        <tr>
            <td style="width: 25%;"><img src="assets/banana_density.svg"></td>
            <td style="width: 25%;"><img src="assets/banana_jointplot.svg"></td>
            <td style="width: 25%;"><img src="examples/Banana/ZigZag_Banana2D.gif"></td>
            <td style="width: 25%;"><img src="examples/Banana/ZigZag_Banana3D.gif"></td>
        </tr>
        <tr>
            <td align="center"><a href="test/runtests.jl"><sup>2D</sup> Banana Density Contour (Ground Truth)</a></td>
            <td align="center"><a href="test/runtests.jl"><sup>2D</sup> Zig-Zag Sample Jointplot</a></td>
            <td align="center"><a href="examples/Banana/ZigZag_Banana2D.jl"><sup>2D</sup> Zig-Zag on Banana</a></td>
            <td align="center"><a href="examples/Banana/ZigZag_Banana3D.jl"><sup>3D</sup> Zig-Zag on Banana</a></td>
        </tr>
    </tbody>
</table>

<table>
    <tbody>
        <tr>
            <td style="width: 33%;"><img src="assets/Cauchy1D.gif"></td>
            <td style="width: 33%;"><img src="assets/Gauss1D.gif"></td>
            <td style="width: 33%;"><img src="assets/densities.svg"></td>
        </tr>
        <tr>
            <td align="center"><a href="test/1d_test.jl"><sup>1D</sup> Zig-Zag on Cauchy</a></td>
            <td align="center"><a href="test/1d_test.jl"><sup>1D</sup> Zig-Zag on Gaussian</a></td>
            <td align="center"><a href="test/1d_test.jl">Cauchy vs. Gaussian Density Plot</a></td>
        </tr>
    </tbody>
</table>

## Remarks

- The automatic Poisson thinning implementation in `PDMPFlux.jl` is based on the paper [Andral and Kamatani (2024) Automated Techniques for Efficient Sampling of Piecewise-Deterministic Markov Processes](https://arxiv.org/abs/2408.03682) and its accompanying Python package [`pdmp_jax`](https://github.com/charlyandral/pdmp_jax).
- [`pdmp_jax`](https://github.com/charlyandral/pdmp_jax) has a [`jax`](https://github.com/jax-ml/jax) based implementation, and typically about four times faster than current `PDMPFlux.jl`.
- Both [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) and [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) are used for automatic differentiation, each with their own trade-offs.

## References

* [`pdmp_jax`](https://github.com/charlyandral/pdmp_jax) by [Charly Andral](https://github.com/charlyandral), on which this repository is strongly based on.
  * [Andral and Kamatani (2024) Automated Techniques for Efficient Sampling of Piecewise-Deterministic Markov Processes](https://arxiv.org/abs/2408.03682)
* [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) and [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) are used for automatic differentiation.
  * [Revels, Lubin, and Papamarkou (2016) Forward-Mode Automatic Differentiation rin Julia](https://arxiv.org/abs/1607.07892)
  * [Innes et. al. (2018) Fashionable Modelling with Flux](https://arxiv.org/abs/1811.01457)
* Other PDMP packages:
  * Julia package [`ZigZagBoomerang.jl`](https://github.com/mschauer/ZigZagBoomerang.jl) by [Marcel Schauer](https://github.com/mschauer)
  * R package [`rjpdmp`](https://github.com/matt-sutton/rjpdmp) by [Matthew Sutton](https://github.com/matt-sutton)
