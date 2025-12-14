"""
    Core sampling functions for PDMPs, such as

    - sample(): sample_skeleton() と sample_from_skeleton() の連続適用
    - sample_skeleton()
    - sample_from_skeleton()
"""

using ProgressBars

"""
    sample()：PDMPSampler からサンプルをするための関数．
    sample_skeleton() と sample_from_skeleton() の wrapper．

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
    seed::Int=nothing,
    verbose::Bool = true
)::Matrix{Float64}

    history = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed, verbose=verbose)
    return sample_from_skeleton(sampler, N_samples, history)

end

## Dispatch method for 1d
function sample(
    sampler::AbstractPDMP,
    N_sk::Int,
    N_samples::Int,
    xinit::Union{Float64, Int},
    vinit::Union{Float64, Int};
    seed::Int=nothing,
    verbose::Bool = true
)::Matrix{Float64}

    xinit = [Float64(xinit)]
    vinit = [Float64(vinit)]
    return sample(sampler, N_sk, N_samples, xinit, vinit, seed=seed, verbose=verbose)

end

"""
    sample_skeleton(): PDMP Samplers からスケルトンを抽出する．

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
    # CI/テスト環境では進捗バーが warning を出すことがあるので、対話環境のみ有効化する
    iter = (verbose && isinteractive()) ? ProgressBar(1:n_sk, unit="B", unit_scale=true) : 1:n_sk

    state = init_state(sampler, xinit, vinit, seed)  # initializing sampler
    history = PDMPHistory(d, n_sk)  # initialize history
    record!(history, 1, state, d)
    
    # 1列目は初期状態で埋めたので 2 から記録する
    for k in Base.Iterators.drop(iter, 1)
        state = get_event_state!(state, sampler)  # go to SamplingLoop.jl or StickySamplingLoop.jl
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

    # NaN/Inf値の検証を追加
    if !isfinite(xinit) || !isfinite(vinit)
        throw(ArgumentError("initial values contain NaN, Inf, or -Inf."))
    end

    xinit = [Float64(xinit)]
    vinit = [Float64(vinit)]
    return sample_skeleton(sampler, N, xinit, vinit, seed=seed, verbose=verbose)

end

"""
    スケルトンからサンプリングをし，各行ベクトルに次元毎の時系列が格納された Matrix{Float64} を返す．

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
    スケルトンからサンプリングをし，各行ベクトルに次元毎の時系列が格納された Matrix{Float64} を返す．

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

# sticky だけ active を使う版
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