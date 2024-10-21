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
    grid::Vector{Float64}  # "n_grid" dimension
    box_max::Vector{Float64}  # "n_grid - 1" dimension
    cum_sum::Vector{Float64}  # "n_grid" dimension
    step_size::Float64
end


"""
    PDMPState <: Any

    AbstractPDMP の状態空間の元．次のフィールドを持つ構造体として実装：

    Attributes:
        x (Array{Float64, 1}): position
        v (Array{Float64, 1}): velocity
        t (Float64): time
        horizon (Float64): horizon
        key (Any): random key
        integrator (Function): integrator function
        ∇U (Function): gradient of the potential function
        rate (Function): rate function
        velocity_jump (Function): velocity jump function
        upper_bound_func (Function): upper bound function
        accept (Bool): accept indicator for the thinning
        upper_bound (Union{Nothing, NamedTuple}): upper bound box
        indicator (Bool): indicator for jumping
        tp (Float64): time to the next event
        ts (Float64): time spent
        exp_rv (Float64): exponential random variable for the Poisson process
        lambda_bar (Float64): upper bound for the Poisson process
        lambda_t (Float64): rate at the current time
        ar (Float64): acceptance rate for the thinning
        error_bound (Int): count of the number of errors in the upper bound
        rejected (Int): count of the number of rejections in the thinning
        hitting_horizon (Int): count of the number of hits of the horizon
        adaptive (Bool): adaptive indicator
"""
mutable struct PDMPState <: Any
    x::Array{Float64}
    v::Array{Float64}
    t::Float64
    horizon::Float64
    key::AbstractRNG
    integrator::Function
    ∇U::Function
    rate::Function
    velocity_jump::Function
    upper_bound_func::Function
    accept::Bool
    upper_bound::Union{Nothing, BoundBox}
    indicator::Bool
    tp::Float64  # time proposed
    ts::Float64  # time spent
    exp_rv::Float64
    lambda_bar::Float64
    lambda_t::Float64
    ar::Float64  # acceptance rate
    error_bound::Int  # 代理上界で足りなかった回数
    error_value_ar::Vector{Float64}  #? jax の実装に引っ張られすぎ？
    rejected::Int
    hitting_horizon::Int  # the total times of hitting the horizon
    adaptive::Bool
end

function PDMPState(x::Vector{Float64}, v::Vector{Float64}, t::Float64, horizon::Float64, key::AbstractRNG, integrator::Function, ∇U::Function, rate::Function, velocity_jump::Function, upper_bound_func::Function, upper_bound::Union{Nothing, BoundBox}, adaptive::Bool)
    accept = false
    indicator = false
    tp = 0.0
    ts = 0.0
    exp_rv = 0.0
    lambda_bar = 0.0
    lambda_t = 0.0
    ar = 0.0
    error_bound = 0
    error_value_ar = zeros(5)
    rejected = 0
    hitting_horizon = 0
    return PDMPState(x, v, t, horizon, key, integrator, ∇U, rate, velocity_jump, upper_bound_func, accept, upper_bound, indicator, tp, ts, exp_rv, lambda_bar, lambda_t, ar, error_bound, error_value_ar, rejected, hitting_horizon, adaptive)
end


# """
#     PDMPOutput <: Any

#     PDMP 実行後の出力を格納するための構造体．

#     Attributes:
#         x (Array{Float64, 1}): The state trajectory.
#         v (Array{Float64, 1}): The velocity trajectory.
#         t (Array{Float64, 1}): The time points at which the state and velocity are recorded.
#         error_bound (Array{Int64, 1}): The error bound at each time point.
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
#     error_bound::Float64
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
    error_bound::Vector{Int}
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
