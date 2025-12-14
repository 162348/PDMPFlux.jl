"""
    Sampling loop tailored to Sticky PDMP Samplers.
"""

"""
    multiple dispatch for get_event_state() function for StickyZigZag
    
    Here, not only an accepted move, but also a crossing of axes and thawing a coordinate constitutes an event,
    leading out of the loop.

    When the new types of events are identified, it is informed via the `indicator` flag.
"""
function get_event_state(state::PDMPState, sampler::StickyPDMP)::PDMPState

    # initialize the counters. The argument sampler::StickyPDMP is there for multiple dispatch.
    state.errored_bound = 0
    state.rejected = 0
    state.hitting_horizon = 0
    fill!(state.error_value_ar, zero(eltype(state.error_value_ar)))

    while !state.accept && !state.stick_or_thaw_event  # get out of the loop if state.accept = true or state.sticking_event = true
        state = one_step_of_thinning_or_sticking_or_thawing(state, sampler)
    end
    
    state.accept, state.stick_or_thaw_event = false, false  # prepare for the next loop

    return state
end

function one_step_of_thinning_or_sticking_or_thawing(state::PDMPState, sampler::StickyPDMP)::PDMPState

    # Note v .* is_active is used, instead of v.
    v_used = _active_velocity(state)
    upper_bound::BoundBox = state.upper_bound_func(state.x, v_used, state.horizon)
    exp_rv = randexp(sampler.rng)
    tp, lambda_bar = next_event(upper_bound, exp_rv)  # tp: proposed time and lambda_bar: upper bound value just before tp.

    rate_thawing = 0.0
    @inbounds for i in eachindex(state.is_active)
        if !state.is_active[i]
            rate_thawing += sampler.κ[i]
        end
    end
    tt = rate_thawing == 0 ? Inf : randexp(sampler.rng) / rate_thawing  # time to thaw

    state.tp, state.exp_rv, state.lambda_bar, state.upper_bound, state.tt = tp, exp_rv, lambda_bar, upper_bound, tt

    event_time = min(tp, state.horizon, tt)  # The earliest time among {proposed time, horizon, time to thaw}
    xₑ, v_discard = sampler.flow(state.x, v_used, event_time)  # calculate the earliest event location
    crossed = false
    @inbounds for i in eachindex(state.x)
        if state.x[i] * xₑ[i] < 0
            crossed = true
            break
        end
    end
    if crossed  # if it crosses axes
        state = move_to_axes_and_stick(state, sampler)  # end the loop
    elseif min(tp, tt) > state.horizon  # else if no event happens before horizon
        state = move_to_horizon(state, sampler)  # continue
    else  # if proposed time or thawing time is before horizon
        state = moves_until_horizon_or_axes(state, sampler)  # go into the next loop (acceptance-rejection / thawing loop)
    end

    return state
end

"""
    movement special to `StickyPDMP`` samplers.
    They move to one of the axes, and start to stick to it.
"""
function move_to_axes_and_stick(state::PDMPState, sampler::StickyPDMP)::PDMPState

    ## Fetch the index of the axis to stick to
    t_togo = Inf
    i = 0
    @inbounds for j in eachindex(state.x)
        if state.is_active[j]
            dj = state.x[j] * state.v[j]  # remaining signed distance (|v|=1 assumption)
            if dj < 0
                tj = -dj
                if tj < t_togo
                    t_togo = tj
                    i = j
                end
            end
        end
    end

    if t_togo == Inf
        error("erronous t_togo, although no axis is crossed")
        # no chance to happen but just to be safe
    end

    # move to the axis & stick to it
    v_used = _active_velocity(state)
    state.x, v_discard = sampler.flow(state.x, v_used, t_togo)
    state.is_active[i] = false  # froze the coordinate
    state.t += t_togo + state.ts
    state.ts = 0.0

    state.stick_or_thaw_event = true  # get out of the loop

    return state  # end of the loop
end

"""
    called while 
        1. no axis is crossed
        and
        2. min(tp, tt) <= state.horizon
    This function calls ac_step() until
        1. state.accept = true
        or
        2. tp > state.horizon
        or
        3. an axis is crossed
"""
function moves_until_horizon_or_axes(state::PDMPState, sampler::StickyPDMP)::PDMPState

    while min(state.tp, state.tt) < state.horizon && !state.accept && !state.stick_or_thaw_event
        if state.tp < state.tt
            state = ac_step(state, sampler)
        else
            state = thaw_one_coordinate(state, sampler)
        end
    end

    return state
end

"""
    Now `tt` is reached, time to thaw one coordinate,
    which is determined according to the proportion of `state.κ[state.is_active]`.
"""
function thaw_one_coordinate(state::PDMPState, sampler::StickyPDMP)::PDMPState
    v_used = _active_velocity(state)
    state.x, v_discard = sampler.flow(state.x, v_used, state.tt)  # move to the thawing time

    total = 0.0
    @inbounds for j in eachindex(state.is_active)
        if !state.is_active[j]
            total += sampler.κ[j]
        end
    end
    u = rand(sampler.rng) * total
    acc = 0.0
    i = 0
    @inbounds for j in eachindex(state.is_active)
        if !state.is_active[j]
            acc += sampler.κ[j]
            if acc >= u
                i = j
                break
            end
        end
    end
    state.is_active[i] = true  # thaw the coordinate

    state.t += state.tt
    state.ts = 0.0
    state.stick_or_thaw_event = true  # get out of the loop
    return state
end
