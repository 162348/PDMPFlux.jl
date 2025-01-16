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
    state.errored_bound, state.rejected, state.hitting_horizon, state.error_value_ar = 0, 0, 0, zeros(5)

    while !state.accept && !state.stick_or_thaw_event  # get out of the loop if state.accept = true or state.sticking_event = true
        state = one_step_of_thinning_or_sticking_or_thawing(state, sampler)
    end
    
    state.accept, state.stick_or_thaw_event = false, false  # prepare for the next loop

    return state
end

function one_step_of_thinning_or_sticking_or_thawing(state::PDMPState, sampler::StickyPDMP)::PDMPState

    # Note v .* is_active is used, instead of v.
    upper_bound::BoundBox = state.upper_bound_func(state.x, state.v .* state.is_active, state.horizon)
    exp_rv = rand(state.key, Exponential(1.0))
    tp, lambda_bar = next_event(upper_bound, exp_rv)  # tp: proposed time and lambda_bar: upper bound value just before tp.

    is_freezing = map(!, state.is_active)
    rate_thawing = sum(sampler.κ[is_freezing])
    tt = rate_thawing == 0 ? Inf : rand(state.key, Exponential(rate_thawing))  # time to thaw

    state.tp, state.exp_rv, state.lambda_bar, state.upper_bound, state.tt = tp, exp_rv, lambda_bar, upper_bound, tt

    event_time = min(tp, state.horizon, tt)  # The earliest time among {proposed time, horizon, time to thaw}
    xₑ, v_discard = state.flow(state.x, state.v .* state.is_active, event_time)  # calculate the earliest event location
    if any(state.x .* xₑ .< 0)  # if it crosses axes
        state = move_to_axes_and_stick(state)  # end the loop
    elseif min(tp, tt) > state.horizon  # else if no event happens before horizon
        state = move_to_horizon(state)  # continue
    else  # if proposed time or thawing time is before horizon
        state = moves_until_horizon_or_axes(state, sampler)  # go into the next loop (acceptance-rejection / thawing loop)
    end

    return state
end

"""
    movement special to `StickyPDMP`` samplers.
    They move to one of the axes, and start to stick to it.
"""
function move_to_axes_and_stick(state::PDMPState)::PDMPState

    ## Fetch the index of the axis to stick to
    d = state.x .* (state.v .* state.is_active)  # remaining distance to the axis (only works for |v|=1 samplers)
    t_togo, i = findmin(map(x -> x < 0 ? -x : Inf, d))

    if t_togo == Inf
        error("erronous t_togo, although no axis is crossed")
        # no chance to happen but just to be safe
    end

    # move to the axis & stick to it
    state.x, v_discard = state.flow(state.x, state.v .* state.is_active, t_togo)
    state.is_active[i] = false  # froze the coordinate
    state.t += t_togo
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
            state = ac_step(state)
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
    state.x, v_discard = state.flow(state.x, state.v .* state.is_active, state.tt)  # move to the thawing time
    is_freezing = map(!, state.is_active)
    p = sampler.κ .* is_freezing ./ sum(sampler.κ[is_freezing])
    i = rand(state.key, Categorical(p))
    state.is_active[i] = true  # thaw the coordinate

    state.t += state.tt
    state.ts = 0.0
    state.stick_or_thaw_event = true  # get out of the loop
    return state
end
