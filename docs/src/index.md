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

N_sk, N, xinit, vinit = 1_000_000, 1_000_000, zeros(dim), ones(dim)
samples = sample(sampler, N_sk, N, xinit, vinit, seed=2025)

jointplot(samples)
```

```@example
using PDMPFlux

function U_banana(x::Vector)
  mean_x2 = (x[1]^2 - 1)
  return -(- x[1]^2 + -(x[2] - mean_x2)^2 - x[3]^2) / 2
end

dim = 3
sampler = ForwardECMCAD(dim, U_Gauss)

N_sk, N, xinit, vinit = 1_000_000, 1_000_000, zeros(dim), ones(dim)
output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2025)

anim_traj(output_BPS, 100)
```

```@autodocs
Modules = [PDMPFlux]
Private = false
Order   = [:module, :function, :type, :macro]
```
