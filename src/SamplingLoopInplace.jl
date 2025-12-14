"""
    In-place (mutating) sampling loop primitives.

    This file provides `!`-functions starting from `one_step_of_thinning!`.
    The non-`!` wrappers are defined in `SamplingLoop.jl` for backward compatibility.

    Main goal: avoid allocations inside the tight loops, especially those coming from
    `v .* is_active` and similar broadcasts.
"""

using Random

@inline function _active_velocity(state::PDMPState)
    if all(state.is_active)
        return state.v
    end

    va = state.v_active
    v = state.v
    a = state.is_active
    @inbounds @simd for i in eachindex(v)
        va[i] = a[i] ? v[i] : zero(eltype(v))
    end
    return va
end

function get_event_state!(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    state.errored_bound = 0
    state.rejected = 0
    state.hitting_horizon = 0
    fill!(state.error_value_ar, zero(eltype(state.error_value_ar)))

    while !state.accept
        one_step_of_thinning!(state, sampler)
    end

    state.accept = false
    return state
end

function one_step_of_thinning!(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    # build a (possibly) masked velocity without allocating
    v_used = _active_velocity(state)

    upper_bound::BoundBox = state.upper_bound_func(state.x, v_used, state.horizon)
    exp_rv = randexp(sampler.rng)  # Exp(1)
    tp, lambda_bar = next_event(upper_bound, exp_rv)

    state.tp = tp
    state.exp_rv = exp_rv
    state.lambda_bar = lambda_bar
    state.upper_bound = upper_bound

    if tp > state.horizon
        move_to_horizon!(state, sampler)
    else
        moves_until_horizon!(state, sampler)
    end

    return state
end

function move_to_horizon!(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    if all(state.is_active)
        state.x, state.v = sampler.flow(state.x, state.v, state.horizon)
    else
        v_used = _active_velocity(state)
        state.x, _v_discard = sampler.flow(state.x, v_used, state.horizon)
        # state.v doesn't change since no event has happened
    end

    state.ts += state.horizon
    state.hitting_horizon += 1
    state.horizon = state.adaptive ? state.horizon * 1.01 : state.horizon

    return state
end

function moves_until_horizon!(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    state.accept = false

    while state.tp < state.horizon && !state.accept
        ac_step!(state, sampler)
    end

    return state
end

function ac_step!(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    v_used = _active_velocity(state)

    lambda_t = sampler.rate(state.x, v_used, state.tp)
    ar = lambda_t / state.lambda_bar

    state.lambda_t = lambda_t
    state.ar = ar

    if ar > 1.0
        erroneous_acceptance_rate!(state, sampler)
    else
        ac_step_with_proxy!(state, sampler)
    end

    return state
end

function erroneous_acceptance_rate!(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    horizon = state.horizon / 2

    v_used = _active_velocity(state)
    upper_bound = state.upper_bound_func(state.x, v_used, horizon)
    exp_rv = randexp(sampler.rng)
    tp, lambda_bar = next_event(upper_bound, exp_rv)

    # shrink state.horizon if state.adaptive = true
    state.horizon = state.adaptive ? horizon : state.horizon

    state.tp = tp
    state.exp_rv = exp_rv
    state.lambda_bar = lambda_bar
    state.upper_bound = upper_bound

    state.errored_bound += 1
    state.error_value_ar[state.errored_bound % 5 + 1] = state.ar

    return state
end

function ac_step_with_proxy!(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    accept = rand(sampler.rng) < state.ar
    state.accept = accept

    if accept
        if_accept!(state, sampler)
    else
        if_reject!(state, sampler)
    end

    if (!state.accept) && (min(state.tp, state.tt) > state.horizon)
        move_to_horizon2!(state, sampler)
    end

    return state
end

function if_accept!(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    if all(state.is_active)
        state.x, state.v = sampler.flow(state.x, state.v, state.tp)
    else
        v_used = _active_velocity(state)
        state.x, _v_discard = sampler.flow(state.x, v_used, state.tp)
    end

    state.v = sampler.velocity_jump(state.x, state.v, sampler.rng)
    state.t = state.t + state.tp + state.ts

    state.ts = 0.0
    state.tp = 0.0
    state.accept = true

    return state
end

function if_reject!(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    # propose tp again (cumulative exp rv)
    exp_rv = state.exp_rv + randexp(sampler.rng)
    tp, lambda_bar = next_event(state.upper_bound, exp_rv)

    # adaptive = true の場合は horizon を縮める
    state.horizon = state.adaptive ? state.horizon / 1.04 : state.horizon

    state.tp = tp
    state.exp_rv = exp_rv
    state.lambda_bar = lambda_bar

    state.rejected += 1

    return state
end

function move_to_horizon2!(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    if all(state.is_active)
        state.x, state.v = sampler.flow(state.x, state.v, state.horizon)
    else
        v_used = _active_velocity(state)
        state.x, _v_discard = sampler.flow(state.x, v_used, state.horizon)
    end

    state.ts += state.horizon
    state.hitting_horizon += 1

    return state
end
