using Random
using Distributions

"""
    SpeedUpZigZag(dim::Int, ∇U::Function; grid_size::Int=10, tmax::Float64=1.0, 
        vectorized_bound::Bool=true, signed_bound::Bool=true, adaptive::Bool=true, kwargs...)

# arguments for constructor
- `dim::Int`: 空間の次元。
- `∇U::Function`: ポテンシャルエネルギー関数の勾配。
- `grid_size::Int`: 空間を離散化するためのグリッドポイントの数。デフォルトは10。
- `tmax::Float64`: グリッドのホライズン。デフォルトは1.0。0の場合、適応的なtmaxが使用されます。
- `vectorized_bound::Bool`: 境界にベクトル化された戦略を使用するかどうか。デフォルトはtrue。
- `signed_bound::Bool`: 符号付き境界戦略を使用するかどうか。デフォルトはtrue。
- `adaptive::Bool`: 適応的なtmaxを使用するかどうか。デフォルトはtrue。
- `kwargs...`: その他のキーワード引数。

# attributes of a ZigZag construct
- `dim::Int`: 空間の次元。
- `refresh_rate::Float64`: リフレッシュレート。
- `∇U::Function`: ポテンシャルの勾配。
- `grid_size::Int`: 空間を離散化するためのグリッドポイントの数。
- `tmax::Float64`: グリッドのtmax。
- `adaptive::Bool`: 適応的なtmaxを使用するかどうか。
- `vectorized_bound::Bool`: ベクトル化された戦略を使用するかどうか。
- `signed_bound::Bool`: 符号付き戦略を使用するかどうか。
- `flow::Function`: インテグレータ関数。
- `rate::Array`: プロセスのレート。
- `rate_vect::Array`: ベクトル化されたレート。
- `signed_rate::Array`: 符号付きレート。
- `signed_rate_vect::Array`: ベクトル化され符号付きのレート。
- `velocity_jump::Function`: 速度ジャンプ関数。
- `state`: ZigZagサンプラーの状態。
"""
mutable struct SpeedUpZigZag <: AbstractPDMP
    dim::Int
    ∇U::Function
    grid_size::Int
    tmax::Float64
    refresh_rate::Float64
    vectorized_bound::Bool
    signed_bound::Bool
    adaptive::Bool
    flow::Function
    rate::Function
    rate_vect::Function
    signed_rate::Union{Function, Nothing}
    signed_rate_vect::Function
    velocity_jump::Function
    state::Any

    """
    Constructor for ZigZag sampler
        - `refresh_rate::Float64`: Not yet used 1/13/2025
    """
    function SpeedUpZigZag(dim::Int, ∇U::Function; grid_size::Int=10, tmax::Union{Float64, Int}=2.0, 
                    refresh_rate::Float64=0.0, vectorized_bound::Bool=true, signed_bound::Bool=true, adaptive::Bool=true)
        
        tmax = Float64(tmax)  # convert tmax to Float64

        # select tmax adaptively if tmax was 0
        tmax == 0.0 ? (tmax = 1.0; adaptive = true) : nothing

        if signed_bound && !vectorized_bound
            signed_bound = false
            @warn "Signed bound is not compatible with non-vectorized bound for ZigZag, switching to unsigned bound"
        end

        function flow(x, v, t)
          y = x - v[1] * x[1] * v
          c = v[1] * (y ⋅ v)
          a = (1 + y ⋅ y) / dim - (c^2) / (dim^2)
          Y_0 = x[1] + (c / dim)
          b_t = (Y_0 + sqrt(Y_0^2 + a)) * exp(sqrt(dim) * v[1] * t)
          X_1 = (b_t^2 - a) / (2 * b_t) - (c / dim)
          return y + v[1] * X_1 * v, v
        end

        speed(x) = sqrt(1.0 + x ⋅ x)
        ∇speed(x) = x / speed(x)
        ∇U_effective(x) = speed(x) * ∇U(x) - ∇speed(x)

        # Define rate functions
        rate = function _global_rate(x0, v0, t)
            xt, vt = flow(x0, v0, t)
            return sum(max.(zeros(dim), ∇U_effective(xt) .* vt))
        end
    
        rate_vect = function _global_rate_vectorized(x0, v0, t)
            xt, vt = flow(x0, v0, t)
            return max.(zeros(dim), ∇U_effective(xt) .* vt)
        end
    
        signed_rate = nothing
    
        signed_rate_vect = function _signed_rate_vect(x0, v0, t)
            xt, vt = flow(x0, v0, t)
            return ∇U_effective(xt) .* vt
        end

        # Define velocity jump function
        function velocity_jump(x, v, rng)
            lambda_t = max.(zeros(dim), ∇U_effective(x) .* v)
            p = lambda_t ./ sum(lambda_t)
            m = rand(rng, Categorical(p))
            v[m] *= -1
            return v
        end

        new(dim, ∇U, grid_size, tmax, refresh_rate, vectorized_bound, signed_bound, adaptive, 
            flow, rate, rate_vect, signed_rate, signed_rate_vect, velocity_jump, nothing)
    end
end

function SpeedUpZigZagAD(dim::Int, U::Function; refresh_rate::Float64=0.0, grid_size::Int=10, tmax::Union{Float64, Int}=2.0, 
                    vectorized_bound::Bool=true, signed_bound::Bool=true, adaptive::Bool=true, AD_backend::String="Zygote")

    ∇U = set_AD_backend(AD_backend, U, dim)

    return SpeedUpZigZag(dim, ∇U, refresh_rate=refresh_rate, grid_size=grid_size, tmax=tmax, vectorized_bound=vectorized_bound, 
                    signed_bound=signed_bound, adaptive=adaptive)
end

