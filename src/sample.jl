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

using Debugger

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

    iter = verbose ? ProgressBar(1:n_sk, unit="B", unit_scale=true) : 1:n_sk

    state = init_state(sampler, xinit, vinit, seed)  # initializing sampler
    history = PDMPHistory(state)  # initialize history
    
    for _ in iter
        state = get_event_state(state, sampler)  # go to SamplingLoop.jl or StickySamplingLoop.jl
        push!(history, state)
    end
    sampler.state = state

    return history
end

"""
    failsafe dispatch of sample_skeleton(), admitting scalar initial values, used mainly for 1d case.
"""
function sample_skeleton(
    sampler::AbstractPDMP,
    n_sk::Int,
    xinit::Union{Float64, Int},
    vinit::Union{Float64, Int};
    seed::Int=nothing,
    verbose::Bool=true
)::PDMPHistory

    xinit = [Float64(xinit)]
    vinit = [Float64(vinit)]
    return sample_skeleton(sampler, n_sk, xinit, vinit, seed=seed, verbose=verbose)

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
    x, v, t = history.x, history.v, history.t
    tm = (t[end] / N) .* collect(1:N)  # equidistant time points
    indeces = searchsortedfirst.(Ref(t), tm) .- 1  # previous index
    samples = map(tuple -> sampler.flow(x[tuple[1]], v[tuple[1]] .* history.is_active[tuple[1]], tm[tuple[2]] - t[tuple[1]])[1], zip(indeces, 1:N))  # flow を通じて位置を取得
    if discard_vt
        return hcat(samples...)
    else 
        samples_v = map(tuple -> sampler.flow(x[tuple[1]], v[tuple[1]] .* history.is_active[tuple[1]], tm[tuple[2]] - t[tuple[1]])[2], zip(indeces, 1:N))  # flow を通じて位置を取得
        return vcat(hcat(samples...), hcat(samples_v...), hcat(tm)')
    end
end