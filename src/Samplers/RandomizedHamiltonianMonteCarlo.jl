using Random

"""
    RHMC(dim::Int, ∇U;
         mean_duration=nothing,
         refresh_rate=1.0,
         phi=pi/2,
         step_size=0.05,
         tmax=10.0,
         adaptive=false,
         AD_backend="FiniteDiff")

Randomized Hamiltonian Monte Carlo (RHMC) sampler (Bou-Rabee & Sanz-Serna, 2017) as a PDMP.

Between events it follows Hamiltonian dynamics

    ẋ = v,    v̇ = -∇U(x)

and at Poisson event times (rate `refresh_rate = 1/mean_duration`) it refreshes momentum via

    v ← cos(phi) v + sin(phi) ξ,   ξ ~ N(0, I)

Notes:
- This implementation uses a velocity-Verlet integrator with step size `step_size` for the flow.
- It is rejection-free in the *exact-flow* limit; with a finite `step_size` this is an approximation.
"""
mutable struct RHMC{G,KF,KR,KRV,KSR,KSRV,KVJ} <: AbstractPDMP
    dim::Int
    ∇U::G
    grid_size::Int
    tmax::Float64
    refresh_rate::Float64
    vectorized_bound::Bool
    signed_bound::Bool
    adaptive::Bool
    flow::KF
    rate::KR
    rate_vect::KRV
    signed_rate::KSR
    signed_rate_vect::KSRV
    velocity_jump::KVJ
    rng::AbstractRNG
    state::Any
    AD_backend::String
    step_size::Float64
    phi::Float64

    function RHMC(
        dim::Int,
        ∇U;
        mean_duration::Union{Nothing,Real}=nothing,
        refresh_rate::Real=1.0,
        phi::Real=pi/2,
        step_size::Real=0.05,
        tmax::Union{Float64,Int}=10.0,
        adaptive::Bool=false,
        AD_backend::String="FiniteDiff",
    )
        if dim <= 0
            throw(ArgumentError("dimension dim must be positive. Current value: $dim"))
        end

        if mean_duration !== nothing
            md = Float64(mean_duration)
            if !(isfinite(md)) || md <= 0
                throw(ArgumentError("mean_duration must be finite and positive. Current value: $mean_duration"))
            end
            refresh_rate = 1.0 / md
        end

        refresh_rate = Float64(refresh_rate)
        if !(isfinite(refresh_rate)) || refresh_rate <= 0
            throw(ArgumentError("refresh_rate must be finite and positive. Current value: $refresh_rate"))
        end

        phi = Float64(phi)
        if !(0.0 < phi <= (pi/2))
            throw(ArgumentError("phi must satisfy 0 < phi ≤ π/2. Current value: $phi"))
        end

        step_size = Float64(step_size)
        if !(isfinite(step_size)) || step_size <= 0
            throw(ArgumentError("step_size must be finite and positive. Current value: $step_size"))
        end

        tmax = Float64(tmax)
        if !(isfinite(tmax)) || tmax < 0
            throw(ArgumentError("tmax must be finite and non-negative. Current value: $tmax"))
        end

        # RHMC uses only a constant Poisson refresh clock.
        grid_size = 0
        vectorized_bound = false
        signed_bound = false

        # Hamiltonian flow approximated by velocity-Verlet.
        function flow(x0, v0, t)
            t = Float64(t)
            if t == 0.0
                return copy(x0), copy(v0)
            end
            if t < 0 || !isfinite(t)
                throw(ArgumentError("flow time t must be finite and non-negative. Current value: $t"))
            end

            x = copy(x0)
            v = copy(v0)

            h = step_size
            n = Int(floor(t / h))
            r = t - n * h

            @inbounds for _ in 1:n
                g = ∇U(x)
                @. v = v - 0.5 * h * g
                @. x = x + h * v
                g2 = ∇U(x)
                @. v = v - 0.5 * h * g2
            end

            if r > 0
                g = ∇U(x)
                @. v = v - 0.5 * r * g
                @. x = x + r * v
                g2 = ∇U(x)
                @. v = v - 0.5 * r * g2
            end

            return x, v
        end

        # Constant event rate (Poisson refresh clock).
        rate = (_x0, _v0, _t) -> refresh_rate

        rate_vect = nothing
        signed_rate = nothing
        signed_rate_vect = nothing

        cφ = cos(phi)
        sφ = sin(phi)

        # Horowitz momentum refreshment (in-place on v).
        function velocity_jump(_x, v, rng)
            @inbounds for i in eachindex(v)
                v[i] = cφ * v[i] + sφ * randn(rng)
            end
            return v
        end

        rng = Random.default_rng()
        state = nothing

        return new{typeof(∇U), typeof(flow), typeof(rate), typeof(rate_vect), typeof(signed_rate), typeof(signed_rate_vect), typeof(velocity_jump)}(
            dim,
            ∇U,
            grid_size,
            tmax,
            refresh_rate,
            vectorized_bound,
            signed_bound,
            adaptive,
            flow,
            rate,
            rate_vect,
            signed_rate,
            signed_rate_vect,
            velocity_jump,
            rng,
            state,
            AD_backend,
            step_size,
            phi,
        )
    end
end

"""
    RHMCAD(dim::Int, U::Function; kwargs...)

Convenience constructor that builds `∇U` via the selected AD backend, like `ZigZagAD` etc.
"""
function RHMCAD(dim::Int, U::Function; AD_backend::String="Zygote", kwargs...)
    ∇U = set_AD_backend(AD_backend, U, dim)
    return RHMC(dim, ∇U; AD_backend=AD_backend, kwargs...)
end

"""
    init_state(pdmp::RHMC, xinit, vinit, seed=nothing)

Specialized initializer for RHMC: the event time is driven by a *constant* rate,
so we can use a cheap 2-point `BoundBox` without optimization / gridding.
"""
function init_state(
    pdmp::RHMC,
    xinit::AbstractVector{Float64},
    vinit::AbstractVector{Float64},
    seed::Union{Int,Nothing}=nothing,
)
    if length(xinit) != pdmp.dim || length(vinit) != pdmp.dim
        throw(DimensionMismatch("xinit と vinit の次元は pdmp.dim ($(pdmp.dim)) と一致する必要があります。現在の次元: xinit ($(length(xinit))), vinit ($(length(vinit)))"))
    end

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    pdmp.rng = rng

    λ = pdmp.refresh_rate
    upper_bound_func = function (_x, _v, horizon)
        horizon = Float64(horizon)
        if horizon < 0 || !isfinite(horizon)
            throw(ArgumentError("horizon must be finite and non-negative. Current value: $horizon"))
        end
        # piecewise-constant bound over [0, horizon]
        grid = [0.0, horizon]
        box_max = [λ]
        cum_sum = [0.0, λ * horizon]
        return BoundBox(grid, box_max, cum_sum, horizon)
    end

    boundox = upper_bound_func(xinit, vinit, pdmp.tmax)
    state = PDMPState(copy(xinit), copy(vinit), 0.0, pdmp.tmax, upper_bound_func, boundox, pdmp.adaptive)
    pdmp.state = state
    return state
end

