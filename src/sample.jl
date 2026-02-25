"""
    Core sampling functions for PDMPs, such as

    - sample(): sequential application of `sample_skeleton()` and `sample_from_skeleton()`
    - sample_skeleton()
    - sample_from_skeleton()
"""

using ProgressBars

"""
    sample(): draw samples from a `AbstractPDMP` sampler.

    This is a thin wrapper that calls `sample_skeleton()` and then `sample_from_skeleton()`.

    Args:
        N_sk (Int): Number of skeleton points to generate.
        N_samples (Int): Number of final samples to generate from the skeleton.
        xinit (Array{Float64, 1}): Initial position.
        vinit (Array{Float64, 1}): Initial velocity.
        seed (Int): Seed for random number generation.
        verbose (Bool, optional): Whether to print progress information. Defaults to true.

    Returns:
        Array{Float64, 2}: Array of samples generated from the PDMP model.
"""
function sample(
    sampler::AbstractPDMP,
    N_sk::Int,
    N_samples::Int,
    xinit::Vector{Float64},
    vinit::Vector{Float64};
    seed::Union{Int, Nothing}=nothing,
    verbose::Bool = true,
    discard_vt::Bool = true
)::Matrix{Float64}

    history = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed, verbose=verbose)
    return sample_from_skeleton(sampler, N_samples, history, discard_vt=discard_vt)

end

## Dispatch method for 1d
function sample(
    sampler::AbstractPDMP,
    N_sk::Int,
    N_samples::Int,
    xinit::Union{Float64, Int},
    vinit::Union{Float64, Int};
    seed::Union{Int, Nothing}=nothing,
    verbose::Bool = true
)::Matrix{Float64}

    xinit = [Float64(xinit)]
    vinit = [Float64(vinit)]
    return sample(sampler, N_sk, N_samples, xinit, vinit, seed=seed, verbose=verbose)

end

"""
    sample_skeleton(): generate a PDMP skeleton (event times/states) from a PDMP sampler.

    Parameters:
    - n_sk (Int): The number of skeleton samples to generate.
    - xinit (Array{Float64, 1}): The initial position of the particles.
    - vinit (Array{Float64, 1}): The initial velocity of the particles.
    - seed (Int): The seed value for random number generation.
    - verbose (Bool): Whether to display progress bar during sampling. Default is true.

    Returns:
    - output: The output state of the sampling process.
"""
function sample_skeleton(
    sampler::AbstractPDMP,
    n_sk::Int,
    xinit::Vector{Float64},
    vinit::Vector{Float64};
    seed::Union{Int, Nothing}=nothing,
    verbose::Bool=true
)::PDMPHistory

    if n_sk <= 0
        throw(ArgumentError("n_sk must be positive. Current value: $n_sk"))
    end

    d = length(xinit)
    # In CI/test environments the progress bar can emit warnings, so only enable it in interactive sessions.
    iter = (verbose && isinteractive()) ? ProgressBar(1:n_sk, unit="B", unit_scale=true) : 1:n_sk

    state = init_state(sampler, xinit, vinit, seed)  # initializing sampler
    history = PDMPHistory(d, n_sk)  # initialize history
    record!(history, 1, state, d)
    
    # Column 1 is already filled with the initial state, so record starting from 2.
    for k in Base.Iterators.drop(iter, 1)
        # StickyPDMP has a specialized `get_event_state(::PDMPState, ::StickyPDMP)`,
        # so call the public non-`!` API to trigger multiple dispatch.
        state = get_event_state(state, sampler)  # go to SamplingLoop.jl or StickySamplingLoop.jl
        record!(history, k, state, d)
    end
    sampler.state = state

    return history
end

"""
    failsafe dispatch of sample_skeleton(), admitting scalar initial values, used mainly for 1d case.
"""
function sample_skeleton(
    sampler::AbstractPDMP,
    N::Int,
    xinit::Union{Float64, Int},
    vinit::Union{Float64, Int};
    seed::Union{Int, Nothing}=nothing,
    verbose::Bool=true
)::PDMPHistory

    if N <= 0
        throw(ArgumentError("n_sk must be positive. Current value: $N"))
    end

    # Validate NaN/Inf in initial values
    if !isfinite(xinit) || !isfinite(vinit)
        throw(ArgumentError("initial values contain NaN, Inf, or -Inf."))
    end

    xinit = [Float64(xinit)]
    vinit = [Float64(vinit)]
    return sample_skeleton(sampler, N, xinit, vinit, seed=seed, verbose=verbose)

end

"""
    sample_skeleton(sampler::AbstractPDMP, T::Float64, xinit, vinit; seed, verbose, init_capacity)

Time-horizon variant: advance the PDMP up to time `T` (instead of generating `n_sk` events) and return the skeleton.

- Since the number of events is not known a priori, `PDMPHistory` is allocated with `init_capacity`
  and grown by doubling as needed (with copying).
- The return value is forced to satisfy `t[end] == T` by adding one final point at `t=T`
  via deterministic flow (because `sample_from_skeleton` uses `t[end]` as the time scale).
"""
function sample_skeleton(
    sampler::AbstractPDMP,
    T::Float64,
    xinit::Vector{Float64},
    vinit::Vector{Float64};
    seed::Union{Int, Nothing}=nothing,
    verbose::Bool=true,
    init_capacity::Int=1024,
)::PDMPHistory

    if !(isfinite(T)) || T < 0
        throw(ArgumentError("T must be finite and non-negative. Current value: $T"))
    end

    d = length(xinit)
    state = init_state(sampler, xinit, vinit, seed)

    cap = max(1, init_capacity)
    history = PDMPHistory(d, cap)
    k = 1
    record!(history, k, state, d)

    show_progress = verbose && isinteractive()
    last_pct = -1
    if show_progress && T > 0
        # print 0% at the beginning
        print(stderr, "\r[sample_skeleton] 0%  (t=0/$(T), events=$(k))")
        flush(stderr)
        last_pct = 0
    end

    # If T==0, the initial point alone is sufficient.
    if T == 0.0
        sampler.state = state
        return _trim_history(history, d, k)
    end

    # Save the previous state (used to create the final t=T point via flow if we overshoot).
    x_prev = similar(state.x)
    v_prev = similar(state.v)
    active_prev = BitVector(undef, d)
    v_used = similar(state.v)
    t_prev = 0.0

    while state.t < T
        copyto!(x_prev, state.x)
        copyto!(v_prev, state.v)
        @inbounds for i in 1:d
            active_prev[i] = state.is_active[i]
        end
        t_prev = state.t

        state = get_event_state(state, sampler)

        if state.t <= T
            k += 1
            if k > size(history.X, 2)
                newcap = max(k, 2 * size(history.X, 2))
                history = _grow_history(history, d, newcap, k - 1)
            end
            record!(history, k, state, d)
        else
            # Overshoot: create and append the t=T point via flow from the previous state, then stop.
            τ = T - t_prev
            if sampler isa StickyPDMP
                @inbounds for i in 1:d
                    v_used[i] = active_prev[i] ? v_prev[i] : 0.0
                end
            else
                copyto!(v_used, v_prev)
            end

            xT, vT = sampler.flow(x_prev, v_used, τ)
            state.x = xT
            state.v = vT
            state.t = T
            @inbounds for i in 1:d
                state.is_active[i] = active_prev[i]
            end
            # `record!` expects these statistics at "non-event" points, so fill with zeros.
            state.ar = 0.0
            state.errored_bound = 0
            fill!(state.error_value_ar, 0.0)
            state.rejected = 0
            state.hitting_horizon = 0

            # Keep v_active consistent too.
            @inbounds for i in 1:d
                state.v_active[i] = state.is_active[i] ? state.v[i] : 0.0
            end

            k += 1
            if k > size(history.X, 2)
                newcap = max(k, 2 * size(history.X, 2))
                history = _grow_history(history, d, newcap, k - 1)
            end
            record!(history, k, state, d)
            break
        end

        if show_progress
            pct = Int(floor(100 * min(state.t / T, 1.0)))
            if pct != last_pct
                print(stderr, "\r[sample_skeleton] $(pct)%  (t=$(round(state.t; digits=4))/$(T), events=$(k))")
                flush(stderr)
                last_pct = pct
            end
        end
    end

    sampler.state = state
    if show_progress
        print(stderr, "\r[sample_skeleton] 100% (t=$(T)/$(T), events=$(k))\n")
        flush(stderr)
    end
    return _trim_history(history, d, k)
end

## Dispatch method for 1d (time horizon)
function sample_skeleton(
    sampler::AbstractPDMP,
    T::Float64,
    xinit::Union{Float64, Int},
    vinit::Union{Float64, Int};
    seed::Union{Int, Nothing}=nothing,
    verbose::Bool=true,
    init_capacity::Int=1024,
)::PDMPHistory
    if !isfinite(xinit) || !isfinite(vinit)
        throw(ArgumentError("initial values contain NaN, Inf, or -Inf."))
    end
    return sample_skeleton(
        sampler,
        T,
        [Float64(xinit)],
        [Float64(vinit)];
        seed=seed,
        verbose=verbose,
        init_capacity=init_capacity,
    )
end

"""
    Sample along a skeleton and return a `Matrix{Float64}` whose rows store per-dimension time series.

    Args:
        N (Int): The number of samples to generate.
        output (PdmpOutput): The PDMP output containing the trajectory information.

    Returns:
        Array{Float64, 2}: The sampled points from the PDMP trajectory skeleton.
    """
function sample_from_skeleton(sampler::AbstractPDMP, N::Int, history::PDMPHistory; discard_vt = true)::Matrix{Float64}
    if N <= 0
        throw(ArgumentError("N must be positive. Current value: $N"))
    end

    X = history.X
    V = history.V
    t = history.t
    t_end = t[end]
    dt = t_end / N

    d = size(X, 1)
    out = discard_vt ? Matrix{Float64}(undef, d, N) : Matrix{Float64}(undef, 2d + 1, N)

    i = 1
    Nh = length(t)
    @inbounds for j in 1:N
        tm = j * dt
        while i < Nh && t[i+1] <= tm
            i += 1
        end

        τ = tm - t[i]
        x0 = @view X[:, i]
        v0 = @view V[:, i]

        if discard_vt
            x_new, _v_new = sampler.flow(x0, v0, τ)
            out[1:d, j] = x_new
        else
            x_new, v_new = sampler.flow(x0, v0, τ)
            out[1:d, j] = x_new
            out[d+1:2d, j] = v_new
            out[2d+1, j] = tm
        end
    end

    return out
end

# Sticky samplers need to respect `is_active` in the reconstructed velocity.
function sample_from_skeleton(sampler::StickyPDMP, N::Int, history::PDMPHistory; discard_vt = true)::Matrix{Float64}
    if N <= 0
        throw(ArgumentError("N must be positive. Current value: $N"))
    end

    X = history.X
    V = history.V
    t = history.t
    active = history.is_active
    t_end = t[end]
    dt = t_end / N

    d = size(X, 1)
    out = discard_vt ? Matrix{Float64}(undef, d, N) : Matrix{Float64}(undef, 2d + 1, N)

    v_used = Vector{Float64}(undef, d)
    i = 1
    Nh = length(t)

    @inbounds for j in 1:N
        tm = j * dt
        while i < Nh && t[i+1] <= tm
            i += 1
        end

        τ = tm - t[i]
        x0 = @view X[:, i]
        v0 = @view V[:, i]

        @inbounds @simd for k in 1:d
            v_used[k] = active[k, i] ? v0[k] : 0.0
        end

        if discard_vt
            x_new, _v_new = sampler.flow(x0, v_used, τ)
            out[1:d, j] = x_new
        else
            x_new, v_new = sampler.flow(x0, v_used, τ)
            out[1:d, j] = x_new
            out[d+1:2d, j] = v_new
            out[2d+1, j] = tm
        end
    end

    return out
end

"""
    Sample along a skeleton and return a `Matrix{Float64}` whose rows store per-dimension time series.

    Args:
        dt (Float64): The time step.
        output (PdmpOutput): The PDMP output containing the trajectory information.

    Returns:
        Array{Float64, 2}: The sampled points from the PDMP trajectory skeleton.
"""
function sample_from_skeleton(
    sampler::AbstractPDMP,
    dt::Float64,
    history::PDMPHistory;
    discard_vt::Bool = true,
)
    return _sample_from_skeleton_impl(sampler, dt, history, discard_vt, false)
end

# Sticky version that respects `is_active`.
function sample_from_skeleton(
    sampler::StickyPDMP,
    dt::Float64,
    history::PDMPHistory;
    discard_vt::Bool = true,
)
    return _sample_from_skeleton_impl(sampler, dt, history, discard_vt, true)
end

function _sample_from_skeleton_impl(
    sampler,
    dt::Float64,
    history::PDMPHistory,
    discard_vt::Bool,
    use_active::Bool,
)
    xhist = history.X
    vhist = history.V
    thist = history.t
    active = history.is_active

    T_end = thist[end]
    n_skel = Int(floor(T_end / dt))

    d = size(xhist, 1)

    if discard_vt
        out = Matrix{Float64}(undef, d, n_skel)
    else
        out = Matrix{Float64}(undef, 2d + 1, n_skel)
    end

    i = 1
    N = length(thist)
    v_used = use_active ? Vector{Float64}(undef, d) : Vector{Float64}(undef, 0)

    @inbounds for j in 1:n_skel
        t = j * dt
        while i < N && thist[i+1] <= t
            i += 1
        end

        τ  = t - thist[i]
        @views x0 = xhist[:, i]
        @views v0 = vhist[:, i]

        if use_active
            @inbounds @simd for k in 1:d
                v_used[k] = active[k, i] ? v0[k] : 0.0
            end
            x_new, v_new = sampler.flow(x0, v_used, τ)
        else
            x_new, v_new = sampler.flow(x0, v0, τ)
        end

        out[1:d, j] = x_new
        if !discard_vt
            out[d+1:2d, j] = v_new
            out[2d+1, j]   = t
        end
    end

    return out
end


function sample_from_skeleton(sampler::AbstractPDMP, N::Int, dt::Float64, history::PDMPHistory; discard_vt = true)::Matrix{Float64}
    X = @view history.X[:, 1:N]
    V = @view history.V[:, 1:N]
    t = @view history.t[1:N]
    active = @view history.is_active[:, 1:N]

    tm = collect(dt:dt:t[end])  # equidistant time points
    M = length(tm)
    indices = searchsortedfirst.(Ref(t), tm) .- 1  # previous index

    d = size(X, 1)
    out = discard_vt ? Matrix{Float64}(undef, d, M) : Matrix{Float64}(undef, 2d + 1, M)

    @inbounds for j in 1:M
        i = indices[j]
        i = i < 1 ? 1 : i
        τ = tm[j] - t[i]

        @views x0 = X[:, i]
        @views v0 = V[:, i]

        if discard_vt
            x_new = sampler.flow(x0, v0 .* active[:, i], τ)[1]
            out[1:d, j] = x_new
        else
            x_new, v_new = sampler.flow(x0, v0 .* active[:, i], τ)
            out[1:d, j] = x_new
            out[d+1:2d, j] = v_new
            out[2d+1, j] = tm[j]
        end
    end

    return out
end

function previous_indices(thist::Vector{Float64}, tm::Vector{Float64})
    N = length(thist)
    M = length(tm)
    idx = Vector{Int}(undef, M)

    i = 1
    @inbounds for j in 1:M
        t = tm[j]
        while i < N && thist[i+1] <= t
            i += 1
        end
        idx[j] = i
    end
    return idx
end