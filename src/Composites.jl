using Random

"""
    BoundBox <: Any

    上界関数の出力を格納するための構造体．

    Attributes:
        grid (Float64 Array{T,1}): The grid values.
        box_max (Float64 Array{T,1}): The maximum values on each segment of the grid.
        cum_sum (Float64 Array{T,1}): The cumulative sum of box_max.
        step_size (Float64): The step size of the grid.
"""
struct BoundBox <: Any
    grid::Vector{Float64}  #  grid points, with "n_grid" dimension
    box_max::Vector{Float64}  #  box_max[i] is the maximum value on the interval [grid[i], grid[i+1]]. Therefore, it has "n_grid - 1" dimension.
    cum_sum::Vector{Float64}  #  cumulative sum of box_max, with "n_grid" dimension. The first element has to be 0.
    step_size::Float64  #  step size of the grid
end


"""
    PDMPState <: Any

    AbstractPDMP の状態空間の元．次のフィールドを持つ構造体として実装：

    Attributes:
        x (Array{Float64, 1}): position
        v (Array{Float64, 1}): velocity
        t (Float64): time
        is_active (Array{Bool, 1}): indicator for the freezing state. Used in the sampling loop for Sticky samplers.

        ∇U (Function): gradient of the potential function
        rate (Function): rate function
        flow (Function): flow function
        velocity_jump (Function): velocity jump function
        upper_bound_func (Function): upper bound function
        upper_bound (Union{Nothing, NamedTuple}): upper bound box

        lambda_bar (Float64): upper bound for the Poisson process
        exp_rv (Float64): exponential random variable for the Poisson process
        
        lambda_t (Float64): rate at the current time
        horizon (Float64): horizon
        tp (Float64): time to the next event
        ts (Float64): time spent
        tt (Vector{Float64}): remaining frozen time for each coordinate, with dimension `dim`
        ar (Float64): acceptance rate for the thinning

        adaptive (Bool): adaptive indicator
        accept (Bool): accept indicator for the thinning
        stick_or_thaw_event (Bool): indicator for the sticking or thawing event
        
        errored_bound (Int): count of the number of errors in the upper bound
        rejected (Int): count of the number of rejections in the thinning
        hitting_horizon (Int): count of the number of hits of the horizon
        
        key (Any): random key
"""
mutable struct PDMPState <: Any
    x::Array{Float64}
    v::Array{Float64}
    t::Float64
    is_active::Array{Bool}
    horizon::Float64
    key::AbstractRNG
    flow::Function
    ∇U::Function
    rate::Function
    velocity_jump::Function
    upper_bound_func::Function
    accept::Bool
    upper_bound::Union{Nothing, BoundBox}
    tp::Float64  # time proposed
    ts::Float64  # time spent
    tt::Float64  # time proposed to thaw one coordinate
    # They are used in `SamplingLoop.jl` to conduct poisson thinning, where `tp` is the proposed jump time, and is added to `ts` when rejected.
    exp_rv::Float64  # to store an exponential random variable
    lambda_bar::Float64  # upper bound for the rate function, calculated from `BoundBox`
    lambda_t::Float64
    ar::Float64  # acceptance rate
    errored_bound::Int  # 代理上界で足りなかった回数
    error_value_ar::Vector{Float64}  #? jax の実装に引っ張られすぎ？
    rejected::Int
    hitting_horizon::Int  # the total times of hitting the horizon
    adaptive::Bool
    stick_or_thaw_event::Bool  # indicator for the sticking or thawing event
end

function PDMPState(x::Vector{Float64}, v::Vector{Float64}, t::Float64, horizon::Float64, key::AbstractRNG,
    flow::Function, ∇U::Function, rate::Function, velocity_jump::Function, upper_bound_func::Function,
    upper_bound::Union{Nothing, BoundBox}, adaptive::Bool)
    is_active = fill(true, length(x))
    accept = false
    tp = 0.0
    ts = 0.0
    tt = Inf
    exp_rv = 0.0
    lambda_bar = 0.0
    lambda_t = 0.0
    ar = 0.0
    errored_bound = 0
    error_value_ar = zeros(5)
    rejected = 0
    hitting_horizon = 0
    stick_or_thaw_event = false
    return PDMPState(x, v, t, is_active, horizon, key, flow, ∇U, rate, velocity_jump, upper_bound_func, accept, upper_bound, tp, ts, tt, exp_rv, lambda_bar, lambda_t, ar, errored_bound, error_value_ar, rejected, hitting_horizon, adaptive, stick_or_thaw_event)
end




# """
#     PDMPOutput <: Any

#     PDMP 実行後の出力を格納するための構造体．

#     Attributes:
#         x (Array{Float64, 1}): The state trajectory.
#         v (Array{Float64, 1}): The velocity trajectory.
#         t (Array{Float64, 1}): The time points at which the state and velocity are recorded.
#         errored_bound (Array{Int64, 1}): The error bound at each time point.
#         error_value_ar (Array{Float64, 1}): The error values at each time point.
#         rejected (Array{Int64, 1}): The indicator of whether a jump was rejected at each time point.
#         hitting_horizon (Array{Int64, 1}): The indicator of whether the process hit the horizon at each time point.
#         ar (Array{Float64, 1}): Acceptance rate at each time point.
#         horizon (Array{Float64, 1}): Horizon values.
# """
# struct PDMPOutput <: Any
#     x::Vector{Float64}
#     v::Vector{Float64}
#     t::Float64
    
#     horizon::Float64
#     ar::Float64
#     errored_bound::Float64
#     error_value_ar::Float64
#     rejected::Float64
#     hitting_horizon::Float64
# end

# """
#     Converts the given PDMPState object into a PDMPOutput object by selecting the relevant fields.

#     Args:
#         state (NamedTuple): The PdmpState object to convert.

#     Returns:
#         NamedTuple: The converted PDMPOutput object.
# """
# function PDMPOutput(state::PDMPState)::PDMPOutput
#     keys = fieldnames(PDMPOutput)
#     values = [getfield(state, key) for key in keys]
#     return PDMPOutput(values...)
# end

mutable struct PDMPHistory <: Any
    x::Vector{Vector{Float64}}
    v::Vector{Vector{Float64}}
    t::Vector{Float64}

    horizon::Vector{Float64}
    ar::Vector{Float64}
    errored_bound::Vector{Int}
    error_value_ar::Vector{Vector{Float64}}
    rejected::Vector{Int}
    hitting_horizon::Vector{Int}
end

"""
    最初の PDMPState オブジェクトから，PDMPHistory オブジェクトを生成するコンストラクタのディスパッチ
"""

function PDMPHistory(init_state::PDMPState)::PDMPHistory
    keys = fieldnames(PDMPHistory)
    values = [[getfield(init_state, key)] for key in keys]
    return PDMPHistory(values...)
end

"""
    PDMPHistory オブジェクトに PDMPState オブジェクトから必要なフィールドを追記するメソッド
"""
# function push!(history::PDMPHistory, output::PDMPOutput)::PDMPHistory
#     keys = fieldnames(PDMPHistory)
#     for key in keys
#         push!(getfield(history, key), getfield(output, key))
#     end
#     return history
# end
function push!(history::PDMPHistory, state::PDMPState)::PDMPHistory
    keys = fieldnames(PDMPHistory)
    for key in keys
        Base.push!(getfield(history, key), getfield(state, key))  # なぜか Base. が必要．
    end
    return history
end
