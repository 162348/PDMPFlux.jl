using Random
using Distributions

"""
    StickyZigZag(dim::Int, ∇U::Function; grid_size::Int=10, tmax::Float64=1.0, 
        vectorized_bound::Bool=true, signed_bound::Bool=true, adaptive::Bool=true, kwargs...)

# arguments for constructor
- `dim::Int`: dimension of the parameter space
- `∇U::Function`: gradient of the potential function (= negative log-likelihood function)
- `κ::Vector{Float64}`: thawing rate defined from prior inclusion probability. default is fill(0.5, dim).

- `grid_size::Int`: number of the grid points for discretization of the parameter space. default is 10.
- `tmax::Float64`: グリッドのホライズン．デフォルトは1.0．0の場合、適応的なtmaxが使用されます．
- `vectorized_bound::Bool`: 境界にベクトル化された戦略を使用するかどうか．デフォルトはtrue．
- `signed_bound::Bool`: 符号付き境界戦略を使用するかどうか．デフォルトはtrue．
- `adaptive::Bool`: 適応的なtmaxを使用するかどうか．デフォルトはtrue．
- `kwargs...`: その他のキーワード引数．

# attributes of a ZigZag construct
- `dim::Int`: 空間の次元．
- `refresh_rate::Float64`: リフレッシュレート．
- `∇U::Function`: ポテンシャルの勾配．
- `grid_size::Int`: 空間を離散化するためのグリッドポイントの数．
- `tmax::Float64`: グリッドのtmax．
- `adaptive::Bool`: 適応的なtmaxを使用するかどうか．
- `vectorized_bound::Bool`: ベクトル化された戦略を使用するかどうか．
- `signed_bound::Bool`: 符号付き戦略を使用するかどうか．
- `flow::Function`: インテグレータ関数．
- `rate::Array`: プロセスのレート．
- `rate_vect::Array`: ベクトル化されたレート．
- `signed_rate::Array`: 符号付きレート．
- `signed_rate_vect::Array`: ベクトル化され符号付きのレート．
- `velocity_jump::Function`: 速度ジャンプ関数．
- `state`: ZigZagサンプラーの状態．
"""
mutable struct StickyZigZag{G,KF,KR,KRV,KSR,KSRV,KVJ} <: StickyPDMP
    dim::Int
    ∇U::G
    κ::Vector{Float64}

    refresh_rate::Float64
    grid_size::Int
    tmax::Float64
    vectorized_bound::Bool
    signed_bound::Bool
    adaptive::Bool
    flow::KF
    rate::KR
    rate_vect::KRV
    signed_rate::KSR
    signed_rate_vect::KSRV
    velocity_jump::KVJ
    rng::AbstractRNG
    state::Any
    AD_backend::String

    """
    Constructor for ZigZag sampler
    """
    function StickyZigZag(dim::Int, ∇U, κ::Vector{Float64}; refresh_rate::Float64=0.0, grid_size::Int=10, tmax::Union{Float64, Int}=2.0, 
                    vectorized_bound::Bool=true, signed_bound::Bool=true, adaptive::Bool=true, AD_backend::String="FiniteDiff")
        
        tmax = Float64(tmax)  # convert tmax to Float64

        if tmax == 0.0   # select tmax adaptively if tmax was 0
            tmax = 1.0
            adaptive = true
        end

        if signed_bound && !vectorized_bound
            signed_bound = false
            @warn "Signed bound is not compatible with non-vectorized bound for ZigZag, switching to unsigned bound"
        end

        flow = (x, v, t) -> (x .+ v .* t, v)

        # Define rate functions
        rate = function _global_rate(x0, v0, t)
            xt, vt = flow(x0, v0, t)
            return sum(max.(0.0, ∇U(xt) .* vt))
        end
    
        rate_vect = function _global_rate_vect(x0, v0, t)
            xt, vt = flow(x0, v0, t)
            return max.(0.0, ∇U(xt) .* vt)
        end
    
        signed_rate = nothing
    
        signed_rate_vect = function _signed_rate_vect(x0, v0, t)
            xt, vt = flow(x0, v0, t)
            return ∇U(xt) .* vt
        end

        # Define velocity jump function
        function velocity_jump(x, v, key)
            lambda_t = max.(0.0, ∇U(x) .* v)
            p = lambda_t ./ sum(lambda_t)
            m = rand(key, Categorical(p))
            v[m] *= -1
            return v
        end

        rng = Random.default_rng()
        state = nothing
        return new{typeof(∇U), typeof(flow), typeof(rate), typeof(rate_vect), typeof(signed_rate), typeof(signed_rate_vect),
                   typeof(velocity_jump)}(
            dim, ∇U, κ, refresh_rate, grid_size, tmax, vectorized_bound, signed_bound, adaptive,
            flow, rate, rate_vect, signed_rate, signed_rate_vect, velocity_jump, rng, state, AD_backend)
    end
end


using Zygote, ForwardDiff, ReverseDiff

function StickyZigZagAD(dim::Int, U::Function, κ::Vector{Float64}; refresh_rate::Float64=0.0, grid_size::Int=10, tmax::Union{Float64, Int}=2.0, 
                    vectorized_bound::Bool=true, signed_bound::Bool=true, adaptive::Bool=true, AD_backend::String="Zygote")

    ∇U = set_AD_backend(AD_backend, U, dim)

    return StickyZigZag(dim, ∇U, κ, refresh_rate=refresh_rate, grid_size=grid_size, tmax=tmax,
            vectorized_bound=vectorized_bound, signed_bound=signed_bound, adaptive=adaptive, AD_backend=AD_backend)
end

