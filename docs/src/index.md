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
sampler = ForwardECMCAD(dim, U_Gauss)

N_sk, N, xinit, vinit = 1_000_000, 1_000_000, zeros(dim), ones(dim)
samples = sample(sampler, N_sk, N, xinit, vinit, seed=2025)

jointplot(samples)
```

```@autodocs
Modules = [PDMPFlux]
Private = false
Order   = [:module, :type, :function, :macro]
```
