# Quickstart Guide

PDMPFlux.jl provides a fast and efficient implementation of **Piecewise Deterministic Markov Process (PDMP)** samplers. This guide will walk you through the basic usage of PDMPFlux.jl.

## Installation

Julia 1.11 or higher is required. Install using Julia's package manager:

```julia
using Pkg
Pkg.add("PDMPFlux")
```

## Basic Usage Examples

### 1. Sampling from a Standard Gaussian Distribution

Let's start with the simplest example - sampling from a standard Gaussian distribution using the Zig-Zag sampler:

```julia
using PDMPFlux

# Define the negative log density function
function U_Gauss(x::Vector)
    return sum(x.^2) / 2
end

# Set up the sampler
dim = 10
sampler = ZigZagAD(dim, U_Gauss)

# Run sampling
N_sk, N, xinit, vinit = 1_000_000, 1_000_000, zeros(dim), ones(dim)
samples = sample(sampler, N_sk, N, xinit, vinit, seed=2024)

# Visualize results
jointplot(samples)
```

### 2. Sampling from a Banana Distribution

For a more complex example, let's try sampling from a Banana distribution (a non-linear distribution):

```julia
using PDMPFlux
using Zygote

# Banana distribution negative log density function
function U_banana(x::Vector)
    mean_x2 = (x[1]^2 - 1)
    return -(- x[1]^2 + -(x[2] - mean_x2)^2) / 2
end

# Manually provide gradient
function ∇U_banana(x::Vector)
    mean_x2 = (x[1]^2 - 1)
    return -(- x[1] + -(x[2] - mean_x2) - sum(x[3:end]))
end

dim = 50
xinit = ones(dim)
vinit = ones(dim)
grid_size = 0  # use constant bounds

# Create and run sampler
sampler = ZigZag(dim, ∇U_banana, grid_size=grid_size)
output = sample_skeleton(sampler, 1_000_000, xinit, vinit)
samples = sample_from_skeleton(sampler, 1_000_000, output)

# Visualize trajectory
plot_traj(output, 10000)
jointplot(samples)
```

## Available Samplers

PDMPFlux.jl provides the following samplers:

### Zig-Zag Sampler
- `ZigZagAD`: Uses automatic differentiation
- `ZigZag`: Manual gradient provision

### Bouncy Particle Sampler (BPS)
- `BPSAD`: Uses automatic differentiation
- `BPS`: Manual gradient provision

### Forward Event Chain Monte Carlo (Forward ECMC)
- `ForwardECMCAD`: Uses automatic differentiation
- `ForwardECMC`: Manual gradient provision

### Boomerang Sampler
- `BoomerangAD`: Uses automatic differentiation
- `Boomerang`: Manual gradient provision

### Speed Up Zig-Zag (SUZZ)
- `SpeedUpZigZagAD`: Uses automatic differentiation
- `SpeedUpZigZag`: Manual gradient provision

### Sticky Zig-Zag Sampler
- `StickyZigZagAD`: Uses automatic differentiation
- `StickyZigZag`: Manual gradient provision

## Visualization and Diagnostics

### Trajectory Visualization
```julia
# Plot 2D trajectory
plot_traj(output, 10000)

# Plot 3D trajectory
plot_traj(output, 1000, plot_type="3D")

# Create animation
anim_traj(output, 1000; filename="trajectory.gif")
```

### Diagnostic Functions
```julia
# Sampler diagnostics
diagnostic(output)
```

### Sample Visualization
```julia
# Joint distribution plot
jointplot(samples)

# Marginal distribution plot
marginalplot(samples)
```

## Advanced Usage

### Custom Gradient Provision
```julia
using ForwardDiff

# Gradient calculation using ForwardDiff
∇U(x::Vector) = ForwardDiff.gradient(U, x)
sampler = ZigZag(dim, ∇U, grid_size=grid_size)
```

### Using Different AD Backends
```julia
using Zygote

# Gradient calculation using Zygote
∇U(x::Vector) = gradient(U, x)[1]
sampler = ZigZag(dim, ∇U, grid_size=grid_size)
```

## Next Steps

- Check the [full documentation](../index.md) for more detailed information
- Explore more complex examples in the [examples](https://github.com/162348/162348.github.io/tree/main/posts/2024/Julia/examples) directory
- Understand the characteristics of each sampler and choose the appropriate one for your problem

PDMPFlux.jl is a powerful tool that enables efficient sampling in high dimensions. Start with simple examples and gradually work your way up to more complex problems.
