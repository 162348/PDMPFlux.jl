"""
    ここでは PDMP のシミュレーションに必要な処理を get_event_state() として実装する．
"""

using Random
using Distributions

"""
    get_event_state(state::PDMPState, sampler::AbstractPDMP)::PDMPState

    the argument `sampler` is only used to implement multiple dispatch.

    state.accept が false である限り，one_step_of_thinning() を繰り返す．
    state.accept が true になるには ac_step_with_proxy() → if_accept() が呼ばれる必要がある．
"""
function get_event_state(state::PDMPState, sampler::AbstractPDMP)::PDMPState

    state.errored_bound, state.rejected, state.hitting_horizon = 0, 0, 0
    state.error_value_ar = zeros(5)

    while !state.accept
        state = one_step_of_thinning(state)
    end
    
    state.accept = false

    return state
end

"""
    state.accept が false である限り実行する処理
    moves_until_horizon() → ac_step_with_proxy() → if_accept() が呼ばれるまで繰り返す．
"""
function one_step_of_thinning(state::PDMPState)::PDMPState
    upper_bound::BoundBox = state.upper_bound_func(state.x, state.v, state.horizon)
    exp_rv = rand(state.key, Exponential(1.0))
    tp, lambda_bar = next_event(upper_bound, exp_rv)  # tp: proposed time and lambda_bar: upper bound value just before tp.
    # lambda_bar can be improper as an upper bound. In that case, erroneous_acceptance_rate() is called in ac_step().
    
    state.tp, state.exp_rv, state.lambda_bar, state.upper_bound = tp, exp_rv, lambda_bar, upper_bound

    if tp > state.horizon
        state = move_to_horizon(state)
    else
        state = moves_until_horizon(state)
    end

    return state
end

"""
    tp > state.horizon の場合，もう一度 Poisson simulation を行う．
"""
function move_to_horizon(state::PDMPState)::PDMPState

    if all(state.is_active)
        state.x, state.v = state.flow(state.x, state.v, state.horizon)
    else  # called only when sampler <: StickyPDMP. The second return value has to be discarded.
        state.x, v_discard = state.flow(state.x, state.v .* state.is_active, state.horizon)
        # state.v doesn't change since no event has happened
    end
    
    state.ts += state.horizon
    state.hitting_horizon = state.hitting_horizon + 1
    state.horizon = state.adaptive ? state.horizon * 1.01 : state.horizon

    return state
end

"""
    tp <= state.horizon の場合の処理
    state.accept = true になるまでの ac_step() の繰り返しとして実装される．
"""
function moves_until_horizon(state::PDMPState)::PDMPState
    state.accept = false  # just to make sure

    while state.tp < state.horizon && !state.accept
        state = ac_step(state)
    end

    return state
end

"""
    acceptance-rejection step
    lambda_bar は正確な上界ではなく，grid が粗い場合に足りない可能性がある．
    その場合は erroneous_acceptance_rate() で補正する．
"""
function ac_step(state::PDMPState)::PDMPState
    lambda_t = state.rate(state.x, state.v .* state.is_active, state.tp)
    ar = lambda_t / state.lambda_bar  # acceptance rate

    state.lambda_t = lambda_t
    state.ar = ar
    
    state = ar > 1.0 ? erroneous_acceptance_rate(state) : ac_step_with_proxy(state)

    return state
end

"""
    代理上界 lambda_bar で足りなかった場合は horizon を縮めてより慎重に ac_step() を繰り返す．
    state.adaptive = true の場合は，ここで horizon を恒久的に縮めておく．
"""
function erroneous_acceptance_rate(state::PDMPState)::PDMPState

    horizon = state.horizon / 2
    upper_bound = state.upper_bound_func(state.x, state.v .* state.is_active, horizon)
    exp_rv = rand(state.key, Exponential(1.0))
    tp, lambda_bar = next_event(upper_bound, exp_rv)

    # shrink state.horizon if state.adaptive = true
    state.horizon = state.adaptive ? horizon : state.horizon
    state.tp, state.exp_rv, state.lambda_bar = tp, exp_rv, lambda_bar
    state.errored_bound += 1
    state.error_value_ar[state.errored_bound % 5 + 1] = state.ar

    return state
end

"""
    代理上界 lambda_bar で足りた場合は，ここで簡単に Poisson 剪定を行う．
    Poisson 剪定中に horizon を超えた場合は move_to_horizon2() を呼び出す．
"""
function ac_step_with_proxy(state::PDMPState)::PDMPState

    accept = rand(Bernoulli(state.ar))
    state.accept = accept
    
    state = accept ? if_accept(state) : if_reject(state)

    if (!state.accept) && (min(state.tp, state.tt) > state.horizon)
        state = move_to_horizon2(state)
    end
    
    return state
    # if tt < tp, thaw_one_coordinate() will be called in moves_until_horizon_or_axes()
    # if tp < tt, ac_step() will be called again in moves_until_horizon()
end

"""
    代理上界 lambda_bar を用いた剪定で accept された場合の処置
    ここで one_step_of_thinning() を終了するために accept = true とされる．
"""
function if_accept(state::PDMPState)::PDMPState
    if all(state.is_active)
        state.x, state.v = state.flow(state.x, state.v, state.tp)
    else  # called only when sampler <: StickyPDMP. The second return value has to be discarded.
        state.x, v_discard = state.flow(state.x, state.v .* state.is_active, state.tp)
    end
    state.v = state.velocity_jump(state.x, state.v, state.key)
    state.t = state.t + state.tp + state.ts

    state.ts, state.tp = 0.0, 0.0
    state.accept = true
    
    return state
end

"""
    代理上界 lambda_bar を用いた剪定で accept されなかった場合の処置
    horizon を超えるまで Poisson 剪定を繰り返す．
"""
function if_reject(state::PDMPState)::PDMPState

    # propose tp again
    exp_rv = state.exp_rv + rand(state.key, Exponential(1.0))
    tp, lambda_bar = next_event(state.upper_bound, exp_rv)

    # adaptive = true の場合は horizon を縮める
    state.horizon = state.adaptive ? state.horizon / 1.04 : state.horizon

    state.tp, state.exp_rv, state.lambda_bar = tp, exp_rv, lambda_bar
    state.rejected += 1
    
    return state  # back to ac_step_with_proxy() to see whether new tp < horizon
end

"""
    代理上界 lambda_bar を使った Poisson 剪定中に horizon を超えた場合の動き
"""
function move_to_horizon2(state::PDMPState)::PDMPState

    if all(state.is_active)
        state.x, state.v = state.flow(state.x, state.v, state.horizon)
    else  # called only when sampler <: StickyPDMP. The second return value has to be discarded.
        state.x, v_discard = state.flow(state.x, state.v .* state.is_active, state.horizon)
    end
    state.ts += state.horizon
    state.hitting_horizon = state.hitting_horizon + 1
    
    return state
end