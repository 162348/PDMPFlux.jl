# PDMPFlux

| Documentation | Workflows | Code Coverage | Quality Assurance |
|:-------------:|:---------:|:-------------:|:-----------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://162348.github.io/PDMPFlux.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://162348.github.io/PDMPFlux.jl/dev/) | [![Build Status](https://github.com/162348/PDMPFlux.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/162348/PDMPFlux.jl/actions/workflows/CI.yml?query=branch%3Amain) | [![Coverage](https://codecov.io/gh/162348/PDMPFlux.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/162348/PDMPFlux.jl) | [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) |

## Overview

`PDMPFlux.jl` provides a fast and efficient implementation of **Piecewise Deterministic Markov Process (PDMP)** samplers, using a grid-based Poisson thinning approach proposed in [Andral and Kamatani (2024)](https://arxiv.org/abs/2408.03682).

By the means of the automatic differentiation engines, `PDMPFlux.jl` only requires `dim` and `U`, which is the negative log density of the target distribution (e.g., posterior).

$$
U(x) = -\log p(x) + \text{const}.
$$

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
output = sample_skeleton(sampler, N_sk, xinit, vinit)  # simulate skeleton points
samples = sample_from_skeleton(sampler, N, output)  # get samples from the skeleton points

plot_traj(output, 10000)
diagnostic(output)

jointplot(samples)
```
<!-- 
## Motivation

Markov Chain Monte Carlo (MCMC) methods are standard in sampling from distributions with unknown normalizing constants.

However, PDMPs (also known as Event Chain Monte Carlo) offer a promising alternative due to their 

1. rejection-free simulation strategy,
2. continuous and non-reversible dynamics,

particularly in high-dimensional and big data contexts, as discussed in [Bouchard-Côté et. al. (2018)](https://arxiv.org/abs/1510.02451) and [Bierkens et. al. (2019)](https://arxiv.org/abs/1607.03188).

Despite their potential, practical applications of PDMPs remained limited by the lack of efficient and flexible implementations.

Inspired by [Andral and Kamatani (2024)](https://arxiv.org/abs/2408.03682) and their `jax` based implementation in [`pdmp_jax`](https://github.com/charlyandral/pdmp_jax),`PDMPFlux.jl` is my attempt to fill this gap, with the aid of the existing automatic differentiation engines.
-->

## Implemented PDMP Samplers

Click a thumbnail to open the full GIF.

<table>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/ZigZag_SlantedGauss2D.gif?raw=true">
        <img alt="Zig-Zag demo" src="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/ZigZag_SlantedGauss2D.gif?raw=true" width="260" />
      </a><br/>
      <b>Zig-Zag Sampler</b><br/>
      <a href="https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-3/The-Zig-Zag-process-and-super-efficient-sampling-for-Bayesian/10.1214/18-AOS1715.full">Bierkens, Fearnhead &amp; Roberts (2019)</a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/BPS_SlantedGauss2D.gif?raw=true">
        <img alt="BPS demo" src="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/BPS_SlantedGauss2D.gif?raw=true" width="260" />
      </a><br/>
      <b>Bouncy Particle Sampler (BPS)</b><br/>
      <a href="https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1294075">Bouchard-Côte et. al. (2018)</a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/ForwardECMC_SlantedGauss2D.gif?raw=true">
        <img alt="Forward ECMC demo" src="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/ForwardECMC_SlantedGauss2D.gif?raw=true" width="260" />
      </a><br/>
      <b>Forward ECMC</b><br/>
      <a href="https://www.tandfonline.com/doi/full/10.1080/10618600.2020.1750417">Michel, Durmus &amp; Sénécal (2020)</a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/Boomerang_SlantedGauss2D.gif?raw=true">
        <img alt="Boomerang demo" src="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/Boomerang_SlantedGauss2D.gif?raw=true" width="260" />
      </a><br/>
      <b>Boomerang Sampler</b><br/>
      <a href="https://proceedings.mlr.press/v119/bierkens20a.html">Bierkens et. al. (2020)</a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/RHMC_SlantedGauss2D.gif?raw=true">
        <img alt="RHMC demo" src="https://github.com/162348/162348.github.io/blob/main/posts/2026/Slides/ISM/RHMC_white.gif?raw=true" width="260" />
      </a><br/>
      <b>Randomized Hamiltonian Monte Carlo (RHMC)</b><br/>
      <a href="https://projecteuclid.org/journals/annals-of-applied-probability/volume-27/issue-4/Randomized-Hamiltonian-Monte-Carlo/10.1214/16-AAP1255.full">Bou-Rabee & Sanz-Serna (2017)</a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/SUZZ_SlantedGauss2D.gif?raw=true">
        <img alt="SUZZ demo" src="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/SUZZ_SlantedGauss2D.gif?raw=true" width="260" />
      </a><br/>
      <b>Speed Up Zig-Zag (SUZZ)</b><br/>
      <a href="https://projecteuclid.org/journals/annals-of-applied-probability/volume-33/issue-6A/Speed-up-Zig-Zag/10.1214/23-AAP1930.full">Vasdekis &amp; Roberts (2023)</a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/StickyZigZag_SlantedGauss2D.gif?raw=true">
        <img alt="Sticky Zig-Zag demo" src="https://github.com/162348/162348.github.io/blob/main/posts/2024/Julia/assets/SlantedGauss/StickyZigZag_SlantedGauss2D.gif?raw=true" width="260" />
      </a><br/>
      <b>Sticky Zig-Zag Sampler</b><br/>
      <a href="https://link.springer.com/article/10.1007/s11222-022-10180-5">Bierkens et. al. (2023)</a>
    </td>
    <td width="33%"></td>
    <td width="33%"></td>
  </tr>
</table>


<!-- 


## Remarks

- The automatic Poisson thinning implementation in `PDMPFlux.jl` is based on the paper [Andral and Kamatani (2024) Automated Techniques for Efficient Sampling of Piecewise-Deterministic Markov Processes](https://arxiv.org/abs/2408.03682) and its accompanying Python package [`pdmp_jax`](https://github.com/charlyandral/pdmp_jax).
- [`pdmp_jax`](https://github.com/charlyandral/pdmp_jax) has a [`jax`](https://github.com/jax-ml/jax) based implementation, and typically about four times faster than current `PDMPFlux.jl`.
- Both [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) and [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) are used for automatic differentiation, each with their own trade-offs.
-->

## References

* [`pdmp_jax`](https://github.com/charlyandral/pdmp_jax) by [Charly Andral](https://github.com/charlyandral), on which this repository is strongly based on and indebted to.
  * [Andral and Kamatani (2024) Automated Techniques for Efficient Sampling of Piecewise-Deterministic Markov Processes](https://arxiv.org/abs/2408.03682)
<!-- * [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) and [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) are used for automatic differentiation.
  * [Revels, Lubin, and Papamarkou (2016) Forward-Mode Automatic Differentiation in Julia](https://arxiv.org/abs/1607.07892)
  * [Innes et. al. (2018) Fashionable Modelling with Flux](https://arxiv.org/abs/1811.01457) -->
* Other PDMP packages:
  * Julia
    * [`ZigZagBoomerang.jl`](https://github.com/mschauer/ZigZagBoomerang.jl) by [Marcel Schauer](https://github.com/mschauer)
    * [`ZZDiffusionBridge`](https://github.com/SebaGraz/ZZDiffusionBridge) by [Sebastiano Grazzi](https://github.com/SebaGraz)
    * [`PDSampler.jl`](https://github.com/alan-turing-institute/PDSampler.jl) by [Alan Turing Institute](https://github.com/alan-turing-institute)
  * R
    * [`rjpdmp`](https://github.com/matt-sutton/rjpdmp) by [Matthew Sutton](https://github.com/matt-sutton)
    * [`RZigZag`](https://github.com/jbierkens/RZigZag) by [Joris Bierkens](https://github.com/jbierkens)

Please kindly notify me if I missed any references. Thank you!