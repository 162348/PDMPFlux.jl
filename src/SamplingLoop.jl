"""
    ここでは PDMP のシミュレーションに必要な処理を one_step() として実装する．
"""

using Random
using Distributions

"""
    one_step(state::PDMPState)::PDMPState

    state.indicator が false である限り，one_step_while() を繰り返す．
    state.indicator が true になるには ok_acceptance() → if_accept() が呼ばれる必要がある．
"""
function one_step(state::PDMPState)::PDMPState

    state.error_bound = 0
    state.rejected = 0
    state.hitting_horizon = 0
    state.error_value_ar = zeros(5)

    while !state.indicator
        state = one_step_while(state)
    end
    
    state.indicator = false

    return state
end

"""
    state.indicator が false である限り実行する処理
    move_before_horizon() → ok_acceptance() → if_accept() が呼ばれるまで繰り返す．
"""
function one_step_while(state::PDMPState)::PDMPState
    upper_bound::BoundBox = state.upper_bound_func(state.x, state.v, state.horizon)
    
    exp_rv = rand(state.key, Exponential(1.0))
    
    tp, lambda_bar = next_event(upper_bound, exp_rv)  # tp: proposed time and lambda_bar: upper bound value just before tp.
    # lambda_bar can be insufficient as an upper bound. In that case, error_acceptance() is called in inner_while().
    
    state.tp = tp
    state.exp_rv = exp_rv
    state.lambda_bar = lambda_bar
    state.upper_bound = upper_bound

    if tp > state.horizon
        state = move_to_horizon(state)
    else
        state = move_before_horizon(state)
    end

    return state
end

"""
    tp <= state.horizon の場合の処理
    state.accept = true になるまでの inner_while() の繰り返しとして実装される．
"""
function move_before_horizon(state::PDMPState)::PDMPState
    state.accept = false

    while state.tp < state.horizon && !state.accept
        state = inner_while(state)
    end

    return state
end

"""
    lambda_bar は正確な上界ではなく，grid が粗い場合に足りない可能性がある．
    その場合は error_acceptance() で補正する．
"""
function inner_while(state::PDMPState)::PDMPState
    lambda_t = state.rate(state.x, state.v, state.tp)
    ar = lambda_t / state.lambda_bar

    state.lambda_t = lambda_t
    state.ar = ar
    
    state = ar > 1.0 ? error_acceptance(state) : ok_acceptance(state)

    return state
end

"""
    代理上界 lambda_bar で足りなかった場合は horizon を縮めてより慎重に inner_while() を繰り返す．
    state.adaptive = true の場合は，ここで horizon を恒久的に縮めておく．
"""
function error_acceptance(state::PDMPState)::PDMPState
    horizon = state.horizon / 2
    upper_bound = state.upper_bound_func(state.x, state.v, horizon)
    exp_rv = rand(state.key, Exponential(1.0))
    tp, lambda_bar = next_event(upper_bound, exp_rv)

    # adaptive = true の場合は state.horizon を恒久的に縮める
    state.horizon = state.adaptive ? horizon : state.horizon
    state.tp = tp
    state.exp_rv = exp_rv
    state.lambda_bar = lambda_bar
    state.error_bound += 1
    state.error_value_ar[state.error_bound % 5 + 1] = state.ar

    return state
end

"""
    代理上界 lambda_bar で足りた場合は，ここで簡単に Poisson 剪定を行う．
    Poisson 剪定中に horizon を超えた場合は move_to_horizon2() を呼び出す．
"""
function ok_acceptance(state::PDMPState)::PDMPState
    accept = rand(Bernoulli(state.ar))
    state.accept = accept
    
    state = accept ? if_accept(state) : if_reject(state)

    if (state.tp > state.horizon) && (!state.accept)
        state = move_to_horizon2(state)
    end
    
    return state
end

"""
    代理上界 lambda_bar を用いた剪定で accept された場合の処置
    ここで one_step_while() を終了するために indicator = true とされる．
"""
function if_accept(state::PDMPState)::PDMPState
    x, v = state.integrator(state.x, state.v, state.tp)
    v = state.velocity_jump(x, v, state.key)
    t = state.t + state.tp + state.ts

    state.x = x
    state.v = v
    state.t = t
    state.indicator = true
    state.ts = 0.0
    state.tp = 0.0
    state.accept = true
    
    return state
end

"""
    代理上界 lambda_bar を用いた剪定で accept されなかった場合の処置
    horizon を超えるまで Poisson 剪定を繰り返す．
"""
function if_reject(state::PDMPState)::PDMPState
    exp_rv = state.exp_rv + rand(state.key, Exponential(1.0))
    tp, lambda_bar = next_event(state.upper_bound, exp_rv)

    # adaptive = true の場合は horizon を縮める
    horizon = state.adaptive ? state.horizon / 1.04 : state.horizon

    state.tp = tp
    state.exp_rv = exp_rv
    state.lambda_bar = lambda_bar
    state.rejected += 1
    state.horizon = horizon
    
    return state
end

"""
    tp > state.horizon の場合，もう一度 Poisson simulation を行う．
"""
function move_to_horizon(state::PDMPState)::PDMPState
    ts = state.ts + state.horizon
    xi, vi = state.integrator(state.x, state.v, state.horizon)

    horizon = state.adaptive ? state.horizon * 1.01 : state.horizon
    
    state.x = xi
    state.v = vi
    state.ts = ts
    state.hitting_horizon = state.hitting_horizon + 1
    state.horizon = horizon

    return state
end

"""
    代理上界 lambda_bar を使った Poisson 剪定中に horizon を超えた場合の動き
"""
function move_to_horizon2(state::PDMPState)::PDMPState
    ts = state.ts + state.horizon
    xi, vi = state.integrator(state.x, state.v, state.horizon)

    state.x = xi
    state.v = vi
    state.ts = ts
    state.hitting_horizon = state.hitting_horizon + 1
    
    return state
end