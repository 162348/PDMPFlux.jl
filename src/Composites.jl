using Random
using StaticArrays

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
        x (AbstractVector{T}): position
        v (AbstractVector{T}): velocity
        t (Float64): time
        is_active (AbstractVector{Bool}): indicator for the freezing state. Used in the sampling loop for Sticky samplers.

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
        
    Note:
        `flow/∇U/rate/velocity_jump/rng` は sampler 側に持たせ，state は「状態」(x,v,t,...) に集中させる．
"""
mutable struct PDMPState{T,TX<:AbstractVector{T},TV<:AbstractVector{T},TA<:AbstractVector{Bool},TF} <: Any
    x::TX
    v::TV
    v_active::TV   # scratch buffer: v .* is_active (no allocations)
    t::T
    is_active::TA
    horizon::T
    upper_bound_func::TF
    accept::Bool
    upper_bound::Union{Nothing, BoundBox}
    tp::T  # time proposed
    ts::T  # time spent
    tt::T  # time proposed to thaw one coordinate
    # They are used in `SamplingLoop.jl` to conduct poisson thinning, where `tp` is the proposed jump time, and is added to `ts` when rejected.
    exp_rv::T  # to store an exponential random variable
    lambda_bar::T  # upper bound for the rate function, calculated from `BoundBox`
    lambda_t::T
    ar::T  # acceptance rate
    errored_bound::Int  # 代理上界で足りなかった回数
    error_value_ar::MVector{5,T}  # fixed-length ring buffer of recent erroneous ARs
    rejected::Int
    hitting_horizon::Int  # the total times of hitting the horizon
    adaptive::Bool
    stick_or_thaw_event::Bool  # indicator for the sticking or thawing event
end


function PDMPState(
    x::AbstractVector{T},
    v::AbstractVector{T},
    t::T,
    horizon::T,
    upper_bound_func,
    upper_bound::Union{Nothing, BoundBox},
    adaptive::Bool,
) where {T<:AbstractFloat}
    is_active = trues(length(x))          # BitVector
    v_active = similar(v)                # same concrete array type as v
    accept = false
    tp = zero(T)
    ts = zero(T)
    tt = T(Inf)
    exp_rv = zero(T)
    lambda_bar = zero(T)
    lambda_t = zero(T)
    ar = zero(T)
    errored_bound = 0
    error_value_ar = MVector{5,T}(undef)
    fill!(error_value_ar, zero(T))
    rejected = 0
    hitting_horizon = 0
    stick_or_thaw_event = false
    return PDMPState(
        x,
        v,
        v_active,
        t,
        is_active,
        horizon,
        upper_bound_func,
        accept,
        upper_bound,
        tp,
        ts,
        tt,
        exp_rv,
        lambda_bar,
        lambda_t,
        ar,
        errored_bound,
        error_value_ar,
        rejected,
        hitting_horizon,
        adaptive,
        stick_or_thaw_event,
    )
end


struct PDMPHistory{T}
    X::Matrix{T}              # d × n
    V::Matrix{T}              # d × n
    t::Vector{T}              # n
    is_active::BitMatrix      # d × n（メモリ最小。速度優先なら Matrix{Bool} でもOK）
    horizon::Vector{T}        # n（一定なら scalar 化も可）
    ar::Vector{T}             # n
    errored_bound::Vector{Int32}
    error_value_ar::Matrix{T} # 5 × n
    rejected::Vector{Int32}
    hitting_horizon::Vector{Int32}
end

function PDMPHistory(d::Int, n::Int; T=Float64)
    PDMPHistory{T}(
        Matrix{T}(undef, d, n),
        Matrix{T}(undef, d, n),
        Vector{T}(undef, n),
        BitMatrix(undef, d, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{Int32}(undef, n),
        Matrix{T}(undef, 5, n),
        Vector{Int32}(undef, n),
        Vector{Int32}(undef, n),
    )
end

# Backward-compatible constructor (older code created history from a state)
function PDMPHistory(s::PDMPState)
    d = length(s.x)
    h = PDMPHistory(d, 1)
    record!(h, 1, s, d)
    return h
end

# Backward-compatible properties: expose `.x` / `.v` as column iterators
function Base.getproperty(h::PDMPHistory, name::Symbol)
    if name === :x
        # テストやユーザコードが `Vector{Float64}`（copy）を期待するので view を返さない
        return map(copy, eachcol(getfield(h, :X)))
    elseif name === :v
        return map(copy, eachcol(getfield(h, :V)))
    else
        return getfield(h, name)
    end
end

"""
    PDMPHistory オブジェクトに PDMPState オブジェクトから必要なフィールドを追記するメソッド
"""
@inline function record!(h::PDMPHistory{T}, k::Int, s::PDMPState, d::Int) where {T}
    # X, V は列メジャーなのでオフセットで一気に copyto!
    off = (k-1)*d + 1
    @inbounds begin
        copyto!(h.X, off, s.x, 1, d)
        copyto!(h.V, off, s.v, 1, d)
        h.t[k] = s.t
        h.horizon[k] = s.horizon
        h.ar[k] = s.ar
        h.errored_bound[k] = Int32(s.errored_bound)
        # error_value_ar は長さ5固定にしておく（後述）
        for j in 1:5
            h.error_value_ar[j, k] = s.error_value_ar[j]
        end
        h.rejected[k] = Int32(s.rejected)
        h.hitting_horizon[k] = Int32(s.hitting_horizon)
        for i in 1:d
            h.is_active[i, k] = s.is_active[i]
        end
    end
    return nothing
end