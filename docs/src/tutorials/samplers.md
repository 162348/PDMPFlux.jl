# Samplers

## Goal of this page

This page gives a **code-oriented** overview of the **seven samplers** implemented in PDMPFlux. The focus is not on theory, but on where the key building blocks live in the codebase and how they are wired together: **`flow` / `rate` / upper bounds / thinning / `velocity_jump`**.

Samplers covered here (exported):
- `ZigZag`
- `BPS` (Bouncy Particle Sampler)
- `ForwardECMC` (Forward Event Chain Monte Carlo)
- `Boomerang`
- `SpeedUpZigZag`
- `StickyZigZag`
- `RHMC` (Randomized Hamiltonian Monte Carlo)

## Common implementation: the minimal `AbstractPDMP` interface

Where things live:
- Sampler types: `src/Samplers/*.jl`
- Common initialization: `src/Samplers/AbstractPDMP.jl`
- Upper bounds (`BoundBox`): `src/UpperBound.jl` / `src/Composites.jl`
- Sampling loop: `src/SamplingLoopInplace.jl` (non-`!` wrappers in `src/SamplingLoop.jl`)
- Sticky-specific loop: `src/StickySamplingLoop.jl`

Each sampler is a subtype of `AbstractPDMP` and is mostly defined by closures assembled in its constructor:
- **`flow(x, v, t)`**: deterministic dynamics between events (possibly a numerical integrator)
- **`rate(x, v, t)` / `rate_vect(x, v, t)`**: event rate (hazard)
- **`signed_rate*`**: for signed-bound strategies (when enabled)
- **`velocity_jump(x, v, rng)`**: velocity update at accepted events (reflection / flip / refresh, etc.)
- **`grid_size, tmax, adaptive, (vectorized_bound, signed_bound)`**: how the proxy upper bound for thinning is constructed

### Thinning (proposal → accept/reject) at a glance

The core loop in `src/SamplingLoopInplace.jl` works like this:
- Build a **proxy upper bound** `BoundBox` via `upper_bound_func(x, v, horizon)` (if `grid_size == 0`, a constant bound is used)
- Sample a **proposed event time `tp`** and the corresponding **bound value `λ̄`** via `next_event(boundbox, Exp(1))`
- Evaluate the true rate `λ(tp) = rate(x, v, tp)` and accept with probability \(ar = λ(tp)/λ̄\)
- **If accepted**: advance with `flow` by `tp`, then apply `velocity_jump` to form the next event state
- **If rejected**: accumulate exponential waiting time and re-propose within the same `BoundBox` (optionally shrinking `horizon` when `adaptive=true`)
- **If the proposal crosses the horizon**: advance to the horizon via `flow` and continue (with mild horizon adaptation if enabled)

Sticky samplers branch to `src/StickySamplingLoop.jl`, where axis-crossings and thawing are treated as additional “events” (via `is_active`).

## Implementation notes for each sampler (7)

### Zig-Zag (`ZigZag`) — `src/Samplers/ZigZagSamplers.jl`

- **flow**: linear motion `x(t) = x + v t` (typically with component-wise velocities \(v_i \in \{\pm 1\}\))
- **rate**:
  - Global: `sum(max.(0, ∇U(x_t) .* v_t))`
  - Vectorized: `max.(0, ∇U(x_t) .* v_t)` (per-coordinate rates)
  - With `signed_bound=true`, `signed_rate_vect = ∇U(x_t) .* v_t` is used for bound construction (note: signed bound is not compatible with `vectorized_bound=false` here, so it is auto-disabled)
- **jump (`velocity_jump`)**: sample an index from probabilities proportional to `max.(0, ∇U(x) .* v)` and flip one component `v[m] *= -1`
- **note**: with `grid_size > 0` and `vectorized_bound=true`, bounds are typically built via `upper_bound_grid_vect`.

### Speed Up Zig-Zag (`SpeedUpZigZag`) — `src/Samplers/SpeedUpZigZagSamplers.jl`

- **intent**: a “sped-up” ZigZag variant via a position-dependent time/velocity transformation (implemented via a nonlinear flow)
- **flow**: closed-form nonlinear update in `flow(x, v, t)` (not straight-line)
- **rate**: same ZigZag structure but using an effective gradient
  - `∇U_effective(x) = speed(x) * ∇U(x) - ∇speed(x)`
  - `speed(x) = sqrt(1 + x⋅x)`, `∇speed(x) = x / speed(x)`
- **jump**: same as ZigZag (flip a single coordinate chosen by the rates)

### Bouncy Particle Sampler (`BPS`) — `src/Samplers/BouncyParticleSamplers.jl`

- **flow**: `x(t)=x+v t`
- **rate**: `max(0, ∇U(x_t)⋅v_t) + refresh_rate`
  - This implementation does not use vectorized bounds (it forces `vectorized_bound=false`).
- **jump**:
  - With probability `bounce_prob = bounce_rate/(bounce_rate+refresh_rate)`: **reflect**
  - Otherwise: **refresh** (draw new `v` via `randn!`; if `Gaussian_velocity=false`, normalize to unit length)
  - Reflection is implemented as `v - 2 * (v⋅g)/||g||^2 * g` with `g = ∇U(x)`.

### Forward Event Chain Monte Carlo (`ForwardECMC`) — `src/Samplers/ForwardEventChainMonteCarlo.jl`

- **flow**: `x(t)=x+v t`
- **rate**: `max(0, ∇U(x_t)⋅v_t)` (a signed variant is also provided)
- **jump**: relative to the gradient direction \(n = ∇U(x)/||∇U(x)||\)
  - decompose `v = v_parallel + v_orthogonal` and sample a new radial/parallel component magnitude `ρ`
  - with probability `mix_p`, **refresh the orthogonal component** (orthogonal switch if `switch=true`, full refresh otherwise)
  - options include `ran_p` (random angle), `positive` (avoid backtracking), `speed_factor` (speed scaling), `normal` (alternative `ρ` generation)
- **note**: requires `dim >= 3` (validated in the constructor).

### Boomerang (`Boomerang`) — `src/Samplers/BoomerangSamplers.jl`

- **flow**: harmonic-oscillator rotation of `(x, v)` by angle `t`
  - `x(t) = x cos t + v sin t`, `v(t) = -x sin t + v cos t`
- **rate**: `max(0, ∇U_eff(x_t)⋅v_t) + refresh_rate` where `∇U_eff(x) = ∇U(x) - x`
- **jump**:
  - Like BPS: **reflect** or **refresh**
  - Refresh draws `v ~ N(0, I)` (this implementation does not normalize).

### Sticky Zig-Zag (`StickyZigZag`) — `src/Samplers/StickyZigZagSamplers.jl` + `src/StickySamplingLoop.jl`

- **extension point**: `StickyPDMP <: AbstractPDMP` adds an **`is_active`** mask (coordinates can be active or frozen/sticky).
- **flow / rate / jump**: same functional form as ZigZag, but the loop passes a masked velocity (inactive coordinates treated as `v_i = 0`) via `_active_velocity`.
- **extra events (Sticky loop)**:
  - **Axis crossing**: if an active coordinate crosses 0, the loop advances to that axis and sets `is_active[i]=false` (stick/freeze)
  - **Thawing**: sample a thaw time using the total thaw rate `sum(κ[i])` over inactive coordinates, then reactivate one coordinate proportional to `κ`
- **parameter**: `κ::Vector{Float64}` controls thawing rates (often tied to prior inclusion probabilities).

### Randomized Hamiltonian Monte Carlo (`RHMC`) — `src/Samplers/RandomizedHamiltonianMonteCarlo.jl`

- **flow**: approximate Hamiltonian dynamics `ẋ=v, v̇=-∇U(x)` using **velocity-Verlet** with step size `step_size`
- **rate**: **constant** Poisson refresh clock only, i.e. `rate = refresh_rate`
  - therefore `init_state(::RHMC, ...)` builds a cheap 2-point `BoundBox` specialized for constant rates
- **jump**: Horowitz (partial) momentum refresh
  - `v ← cos(phi) v + sin(phi) ξ`, with `ξ ~ N(0, I)`
- **note**: if `mean_duration` is provided, it is converted to `refresh_rate = 1/mean_duration`.

## About AD constructors

Each sampler has an `*AD` constructor (e.g. `ZigZagAD`, `BPSAD`, `ForwardECMCAD`, `BoomerangAD`, `SpeedUpZigZagAD`, `StickyZigZagAD`, `RHMCAD`) that builds `∇U(x)` from a potential `U(x)` and passes it to the sampler.

Additionally, bound construction for thinning (`upper_bound_grid*`) may need derivatives with respect to time, so `AD_backend` is also consulted on the **upper-bound side** (see `_pdmp_ad_backend` in `src/Samplers/AbstractPDMP.jl`).

