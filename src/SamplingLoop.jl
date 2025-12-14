"""
    PDMP sampling loop (non-`!` wrappers).

    The mutating implementations live in `SamplingLoopInplace.jl`.
    This file keeps the original public API (non-`!`) as thin wrappers.
"""

"""
    get_event_state(state::PDMPState, sampler::AbstractPDMP)::PDMPState

    the argument `sampler` is only used to implement multiple dispatch.

    state.accept が false である限り，one_step_of_thinning() を繰り返す．
    state.accept が true になるには ac_step_with_proxy() → if_accept() が呼ばれる必要がある．
"""
function get_event_state(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return get_event_state!(state, sampler)
end

"""
    state.accept が false である限り実行する処理
    moves_until_horizon() → ac_step_with_proxy() → if_accept() が呼ばれるまで繰り返す．
"""
function one_step_of_thinning(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return one_step_of_thinning!(state, sampler)
end

"""
    tp > state.horizon の場合，もう一度 Poisson simulation を行う．
"""
function move_to_horizon(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return move_to_horizon!(state, sampler)
end

"""
    tp <= state.horizon の場合の処理
    state.accept = true になるまでの ac_step() の繰り返しとして実装される．
"""
function moves_until_horizon(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return moves_until_horizon!(state, sampler)
end

"""
    acceptance-rejection step
    lambda_bar は正確な上界ではなく，grid が粗い場合に足りない可能性がある．
    その場合は erroneous_acceptance_rate() で補正する．
"""
function ac_step(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return ac_step!(state, sampler)
end

"""
    代理上界 lambda_bar で足りなかった場合は horizon を縮めてより慎重に ac_step() を繰り返す．
    state.adaptive = true の場合は，ここで horizon を恒久的に縮めておく．
"""
function erroneous_acceptance_rate(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return erroneous_acceptance_rate!(state, sampler)
end

"""
    代理上界 lambda_bar で足りた場合は，ここで簡単に Poisson 剪定を行う．
    Poisson 剪定中に horizon を超えた場合は move_to_horizon2() を呼び出す．
"""
function ac_step_with_proxy(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return ac_step_with_proxy!(state, sampler)
end

"""
    代理上界 lambda_bar を用いた剪定で accept された場合の処置
    ここで one_step_of_thinning() を終了するために accept = true とされる．
"""
function if_accept(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return if_accept!(state, sampler)
end

"""
    代理上界 lambda_bar を用いた剪定で accept されなかった場合の処置
    horizon を超えるまで Poisson 剪定を繰り返す．
"""
function if_reject(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return if_reject!(state, sampler)
end

"""
    代理上界 lambda_bar を使った Poisson 剪定中に horizon を超えた場合の動き
"""
function move_to_horizon2(state::PDMPState, sampler::AbstractPDMP)::PDMPState
    return move_to_horizon2!(state, sampler)
end
