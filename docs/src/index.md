```@meta
CurrentModule = PDMPFlux
```

# PDMPFlux

Documentation for [PDMPFlux](https://github.com/162348/PDMPFlux.jl).

```@index
```

```@example
using PDMPFlux

function U_Gauss(x::AbstractVector)
    return sum(x.^2) / 2
end

dim = 10
sampler = BPSAD(dim, U_Gauss)

N_sk, N, xinit, vinit = 20_000, 5_000, zeros(dim), ones(dim)
samples = sample(sampler, N_sk, N, xinit, vinit, seed=2025)

jointplot(samples)
```

```@example
using PDMPFlux

function U_banana(x::AbstractVector)
  mean_x2 = (x[1]^2 - 1)
  return -(- x[1]^2 + -(x[2] - mean_x2)^2 - x[3]^2) / 2
end

dim = 3
sampler = ForwardECMCAD(dim, U_banana)

N_sk, xinit, vinit = 20_000, zeros(dim), ones(dim)
output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2025)

plot_traj(output, 300, plot_type="3D")
```

```@autodocs
Modules = [PDMPFlux]
Private = false
Order   = [:module, :function, :type, :macro]
```
