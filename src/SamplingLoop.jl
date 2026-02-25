"""
    PDMP sampling loop (non-`!` wrappers).

    The mutating implementations live in `SamplingLoopInplace.jl`.
    This file keeps the original public API (non-`!`) as thin wrappers.
"""

"""
    get_event_state(state::PDMPState, sampler::AbstractPDMP)::PDMPState

    the argument `sampler` is only used to implement multiple dispatch.

    Repeats `one_step_of_thinning()` while `state.accept` is `false`.

    For `state.accept` to become `true`, the call chain must reach
    `ac_step_with_proxy()` → `if_accept()`.
"""
function get_event_state(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return get_event_state!(state, sampler)
end

"""
    Execute one outer thinning step while `state.accept` is `false`.

    Internally this repeats the chain `moves_until_horizon()` → `ac_step_with_proxy()` → `if_accept()`
    until an event is accepted.
"""
function one_step_of_thinning(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return one_step_of_thinning!(state, sampler)
end

"""
    Case `tp > state.horizon`: advance the flow to the horizon (no event within the horizon).
"""
function move_to_horizon(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return move_to_horizon!(state, sampler)
end

"""
    Case `tp <= state.horizon`.

    Implemented as repeated calls to `ac_step()` until `state.accept` becomes `true`.
"""
function moves_until_horizon(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return moves_until_horizon!(state, sampler)
end

"""
    acceptance-rejection step

    `lambda_bar` is an *approximate* upper bound (e.g. when the grid is coarse it may be too small).
    In that case we correct via `erroneous_acceptance_rate()`.
"""
function ac_step(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return ac_step!(state, sampler)
end

"""
    Handle the case where the proxy bound `lambda_bar` was too small.

    Shrinks the horizon and retries `ac_step()` more conservatively. If `state.adaptive == true`,
    the horizon is permanently reduced here.
"""
function erroneous_acceptance_rate(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return erroneous_acceptance_rate!(state, sampler)
end

"""
    Proxy-bound thinning step using `lambda_bar`.

    If Poisson thinning crosses the horizon, calls `move_to_horizon2()`.
"""
function ac_step_with_proxy(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return ac_step_with_proxy!(state, sampler)
end

"""
    Action taken when a proposal is accepted under proxy-bound thinning.

    Sets `accept = true` to terminate `one_step_of_thinning()`.
"""
function if_accept(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return if_accept!(state, sampler)
end

"""
    Action taken when a proposal is rejected under proxy-bound thinning.

    Repeats Poisson thinning until the horizon is exceeded.
"""
function if_reject(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return if_reject!(state, sampler)
end

"""
    Action taken when Poisson thinning with proxy bound `lambda_bar` exceeds the horizon.
"""
function move_to_horizon2(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return move_to_horizon2!(state, sampler)
end
