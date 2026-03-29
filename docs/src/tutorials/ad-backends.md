# AD Backends

## Goal of this page

This page explains how automatic differentiation (AD) is handled in `PDMPFlux.jl`, how the `AD_backend` keyword works in the `*AD` constructors, and what happens when a requested backend is not compatible with the user-provided potential.

## Which constructors use `AD_backend`?

The following constructors build a gradient `∇U(x)` from a potential `U(x)`:

- `ZigZagAD`
- `BPSAD`
- `ForwardECMCAD`
- `BoomerangAD`
- `SpeedUpZigZagAD`
- `StickyZigZagAD`
- `RHMCAD`

All of them accept the keyword

```julia
AD_backend="ForwardDiff"
```

and all of them now use the same backend-resolution logic implemented in `src/ADBackend.jl`.

## Default behavior

The default requested backend is now `ForwardDiff` for all `*AD` constructors.

For example:

```julia
using PDMPFlux

U(x) = sum(abs2, x) / 2

sampler = ZigZagAD(5, U)
@show sampler.AD_backend
```

In the common case, this prints

```julia
sampler.AD_backend = "ForwardDiff"
```

## Initialization-time backend resolution

The important point is that `AD_backend` is now resolved **once, during initialization**.

When you call a constructor such as `ZigZagAD(dim, U; AD_backend="ForwardDiff")`, PDMPFlux:

1. inspects how `U` is meant to be called,
2. checks whether the requested backend is compatible,
3. builds the gradient closure `∇U`,
4. stores the **resolved** backend in the sampler.

This means that PDMPFlux does **not** try one backend during sampling, throw exceptions repeatedly, and then fall back at runtime. The backend is fixed before sampling starts.

## Why this matters

This change is especially important for performance-sensitive tests and long sampling runs.

A common Julia pattern is to write a scalar potential as

```julia
U(x::Float64) = x^2 / 2
```

This is valid Julia code, but it is often **not compatible with `ForwardDiff`**, because `ForwardDiff` passes dual numbers internally, not plain `Float64` values.

Previously, a runtime fallback strategy could repeatedly hit exceptions inside gradient evaluations. That made long tests dramatically slower.

Now the compatibility check happens only once at construction time.

## Example: scalar `Float64` signature

Consider the following 1D example:

```julia
using PDMPFlux

U_scalar_typed(x::Float64) = x^2 / 2

sampler = ZigZagAD(1, U_scalar_typed; AD_backend="ForwardDiff", grid_size=0)
@show sampler.AD_backend
```

Because `U_scalar_typed` is not dual-number compatible, PDMPFlux resolves the backend at initialization time and stores

```julia
sampler.AD_backend = "Zygote"
```

After that, sampling uses the resolved backend directly:

```julia
output = sample_skeleton(sampler, 1000, 0.0, 1.0, seed=42)
```

No runtime AD fallback is needed.

## Recommended coding style for `ForwardDiff`

If you want to keep using `ForwardDiff`, prefer generic method signatures instead of `Float64`-only ones.

Good:

```julia
U(x) = x^2 / 2
```

or

```julia
U(x::Real) = x^2 / 2
```

or, in the multivariate case,

```julia
U(x::AbstractVector) = sum(abs2, x) / 2
```

Less robust for `ForwardDiff`:

```julia
U(x::Float64) = x^2 / 2
```

The last form may still work with `Zygote`, but it prevents `ForwardDiff` from being selected.

## Manual backend selection

You can still request a backend explicitly:

```julia
sampler1 = ZigZagAD(5, U; AD_backend="ForwardDiff")
sampler2 = ZigZagAD(5, U; AD_backend="Zygote")
sampler3 = ZigZagAD(5, U; AD_backend="ReverseDiff")
```

If `Enzyme` is installed, you can also use

```julia
sampler4 = ZigZagAD(5, U; AD_backend="Enzyme")
```

If a requested backend is unsupported, PDMPFlux throws an error immediately.

## What does `sampler.AD_backend` mean?

The field `sampler.AD_backend` stores the **effective backend actually used by the sampler**.

So if you request

```julia
sampler = ZigZagAD(1, U_scalar_typed; AD_backend="ForwardDiff")
```

and `ForwardDiff` is not compatible, then

```julia
sampler.AD_backend == "Zygote"
```

This is intentional: the field reflects the resolved backend, not just the original user request.

## Upper-bound construction and `AD_backend`

The same backend field is also used when PDMPFlux builds derivatives needed for thinning bounds such as `upper_bound_grid` and `upper_bound_grid_vect`.

So once a sampler is initialized, the resolved `AD_backend` is used consistently both for:

- building `∇U(x)`, and
- building upper bounds during sampling.

## Practical advice

- If you do not care about the backend, just use the default and let PDMPFlux resolve it.
- If you want `ForwardDiff`, write backend-friendly potential functions with generic argument types.
- If you need to confirm what was selected, inspect `sampler.AD_backend`.
- If you already have a hand-written gradient, use the non-`AD` constructor directly.

## Summary

- All `*AD` constructors now share a unified `AD_backend` implementation.
- The default requested backend is `ForwardDiff`.
- Backend compatibility is checked once at initialization time.
- The resolved backend is fixed before sampling starts.
- `sampler.AD_backend` reports the backend that is actually used.
