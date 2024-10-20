using Random
using Distributions

"""
    ZigZag(dim::Int, grad_U::Function; grid_size::Int=10, tmax::Float64=1.0, 
        vectorized_bound::Bool=true, signed_bound::Bool=true, adaptive::Bool=true, kwargs...)

ZigZag クラスは ZigZag サンプラーのためのクラスです。

# 引数
- `dim::Int`: 空間の次元。
- `grad_U::Function`: ポテンシャルエネルギー関数の勾配。
- `grid_size::Int`: 空間を離散化するためのグリッドポイントの数。デフォルトは10。
- `tmax::Float64`: グリッドのホライズン。デフォルトは1.0。0の場合、適応的なtmaxが使用されます。
- `vectorized_bound::Bool`: 境界にベクトル化された戦略を使用するかどうか。デフォルトはtrue。
- `signed_bound::Bool`: 符号付き境界戦略を使用するかどうか。デフォルトはtrue。
- `adaptive::Bool`: 適応的なtmaxを使用するかどうか。デフォルトはtrue。
- `kwargs...`: その他のキーワード引数。

# 属性
- `dim::Int`: 空間の次元。
- `refresh_rate::Float64`: リフレッシュレート。
- `grad_U::Function`: ポテンシャルの勾配。
- `grid_size::Int`: 空間を離散化するためのグリッドポイントの数。
- `tmax::Float64`: グリッドのtmax。
- `adaptive::Bool`: 適応的なtmaxを使用するかどうか。
- `vectorized_bound::Bool`: ベクトル化された戦略を使用するかどうか。
- `signed_bound::Bool`: 符号付き戦略を使用するかどうか。
- `integrator::Function`: インテグレータ関数。
- `rate::Array`: プロセスのレート。
- `rate_vect::Array`: ベクトル化されたレート。
- `signed_rate::Array`: 符号付きレート。
- `signed_rate_vect::Array`: ベクトル化され符号付きのレート。
- `velocity_jump::Function`: 速度ジャンプ関数。
- `state`: ZigZagサンプラーの状態。
"""
mutable struct ZigZag <: AbstractPDMP
    dim::Int
    refresh_rate::Float64
    grad_U::Function
    grid_size::Int
    tmax::Float64
    vectorized_bound::Bool
    signed_bound::Bool
    adaptive::Bool
    integrator::Function
    rate::Function
    rate_vect::Function
    signed_rate::Union{Function, Nothing}
    signed_rate_vect::Function
    velocity_jump::Function
    state::Any

    """
    ZigZagサンプラーのコンストラクタ。
    """
    function ZigZag(dim::Int, grad_U::Function; refresh_rate::Float64=0.0, grid_size::Int=10, tmax::Union{Float64, Int}=2.0, 
                    vectorized_bound::Bool=true, signed_bound::Bool=true, adaptive::Bool=true)
        
        tmax = Float64(tmax)  # convert tmax to Float64

        if tmax == 0.0   # select tmax adaptively if tmax was 0
            tmax = 1.0
            adaptive = true
        end

        if signed_bound && !vectorized_bound
            signed_bound = false
            @warn "Signed bound is not compatible with non-vectorized bound for ZigZag, switching to unsigned bound"
        end

        integrator = (x, v, t) -> (x + v * t, v)

        # Define rate functions
        rate = function _global_rate(x0, v0, t)
            xt, vt = integrator(x0, v0, t)
            return sum(max.(zeros(dim), grad_U(xt) .* vt))
        end
    
        rate_vect = function _global_rate_vect(x0, v0, t)
            xt, vt = integrator(x0, v0, t)
            return max.(zeros(dim), grad_U(xt) .* vt)
        end
    
        signed_rate = nothing
    
        signed_rate_vect = function _signed_rate_vect(x0, v0, t)
            xt, vt = integrator(x0, v0, t)
            return grad_U(xt) .* vt
        end

        # Define velocity jump function
        function velocity_jump(x, v, rng)
            lambda_t = max.(zeros(dim), grad_U(x) .* v)
            proba = lambda_t ./ sum(lambda_t)
            m = rand(rng, Categorical(proba))
            v[m] *= -1
            return v
        end

        new(dim, refresh_rate, grad_U, grid_size, tmax, vectorized_bound, signed_bound, adaptive, 
            integrator, rate, rate_vect, signed_rate, signed_rate_vect, velocity_jump, nothing)
    end
end

using Zygote, ForwardDiff, ReverseDiff

function ZigZagAD(dim::Int, U::Function; refresh_rate::Float64=0.0, grid_size::Int=10, tmax::Union{Float64, Int}=2.0, 
                    vectorized_bound::Bool=true, signed_bound::Bool=true, adaptive::Bool=true, AD_backend::String="Zygote")

    grad_U = nothing
    AD_backend = eval(Symbol(AD_backend))

    ## If U is one dimensional and takes Float64 instead of Vector{Float64}, change grad_U accordingly:
    if dim == 1
        try
            U([1.0])
        catch
            grad_U = function(x::Vector)
                return AD_backend.gradient(U, x[1])[1]
            end
        else
            grad_U = function(x::Vector)
                return AD_backend.gradient(U, x)[1]
            end
        end
    else
        grad_U = function(x::Vector)
            return AD_backend.gradient(U, x)[1]
        end
    end

    return ZigZag(dim, grad_U, refresh_rate=refresh_rate, grid_size=grid_size, tmax=tmax, vectorized_bound=vectorized_bound, 
                    signed_bound=signed_bound, adaptive=adaptive)
end

