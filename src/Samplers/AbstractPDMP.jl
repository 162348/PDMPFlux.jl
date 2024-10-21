using Random

"""
    In PDMPFlux, every PDMP sampler is defined to be of subtype of AbstractPDMP.
"""
abstract type AbstractPDMP end

"""
    ダイナミクスが抽象化された PDMP でサンプリングを行うための抽象構造体

    属性:
        dim::Int: 空間の次元。
        refresh_rate::Float64: リフレッシュレート。
        ∇U::Function: ポテンシャルの勾配。
        grid_size::Int: 空間を離散化するためのグリッドポイントの数。
        tmax::Float64: グリッドの最大時間。
        adaptive::Bool: 適応的なtmaxを使用するかどうか。
        vectorized_bound::Bool: ベクトル化された戦略を使用するかどうか。
        signed_bound::Bool: 符号付き戦略を使用するかどうか。
        integrator::Function: インテグレータ関数。
        rate::Array: プロセスのレート。
        rate_vect::Array: ベクトル化されたレート。
        signed_rate::Array: 符号付きレート。
        signed_rate_vect::Array: ベクトル化され符号付きのレート。
        velocity_jump::Function: 速度ジャンプ関数。
        state::Any: ZigZagサンプラーの状態。
    メソッド:
        init_state(xinit::Array{Float64,1}, vinit::Array{Float64,1}, seed::Int) -> PdmpState:
        sample_skeleton(n_sk::Int, xinit::Array{Float64,1}, vinit::Array{Float64,1}, seed::Int, verbose::Bool=true) -> PdmpOutput:
        sample_from_skeleton(N::Int, output::PdmpOutput) -> Array{Float64,2}:
        sample(N_sk::Int, N_samples::Int, xinit::Array{Float64,1}, vinit::Array{Float64,1}, seed::Int, verbose::Bool=true) -> Array{Float64,2}:
        _init_bps_rate() -> Tuple{Function, Nothing, Function, Nothing}:
"""
# struct PDMP <: AbstractPDMP
#     dim::Int
#     refresh_rate::Float64
#     ∇U::Function
#     grid_size::Int
#     tmax::Float64

#     vectorized_bound::Bool
#     signed_bound::Bool
#     adaptive::Bool

#     integrator::Function
#     rate::Function
#     velocity_jump::Function
#     state::Union{PDMPState, Nothing}
# end


"""
    init_state():
    PDMP オブジェクトの状態を初期化する．

    Args:
        xinit (Float[Array, "dim"]): The initial position.
        vinit (Float[Array, "dim"]): The initial velocity.
        seed (int): The seed for random number generation.
        upper_bound_vect (bool, optional): Whether to use vectorized upper bound function. Defaults to False.
        signed_rate (bool, optional): Whether to use signed rate function. Defaults to False.
        adaptive (bool, optional): Whether to use adaptive upper bound. Defaults to False.
        constant_bound (bool, optional): Whether to use constant upper bound. Defaults to False.

    Returns:
        PdmpState: The initialized PDMP state.
"""
function init_state(pdmp::AbstractPDMP, xinit::Array{Float64}, vinit::Array{Float64}, seed::Int)

    # xinit と vinit の次元が pdmp.dim に一致するか確認
    if length(xinit) != pdmp.dim || length(vinit) != pdmp.dim
        throw(DimensionMismatch("xinit と vinit の次元は pdmp.dim ($(pdmp.dim)) と一致する必要があります。現在の次元: xinit ($(length(xinit))), vinit ($(length(vinit)))"))
    end

    key = Random.seed!(seed)

    # rate, rate_vect, refresh_rate の設定は signed_bound に依存して異なる
    if pdmp.signed_bound
        rate = pdmp.signed_rate
        rate_vect = pdmp.signed_rate_vect
        refresh_rate = pdmp.refresh_rate
    else
        rate = pdmp.rate
        rate_vect = pdmp.rate_vect
        refresh_rate = 0.0
    end

    # グリッドサイズが0の場合、Brentのアルゴリズムを使用して定数上限戦略を使用
    if pdmp.grid_size == 0
        upper_bound_func = function(x, v, horizon)
            func = t -> pdmp.rate(x, v, t)  # pdmp.signed_bound のフラッグに依らず，符号なしの pdmp.rate を用いる．
            return upper_bound_constant(func, 0.0, horizon)
        end
    elseif !pdmp.vectorized_bound
        upper_bound_func = function(x, v, horizon)
            func = t -> rate(x, v, t)
            return upper_bound_grid(func, 0.0, horizon, pdmp.grid_size, refresh_rate)
        end
    else
        upper_bound_func = function(x, v, horizon)
            func = t -> rate_vect(x, v, t)
            return upper_bound_grid_vect(func, 0.0, horizon, pdmp.grid_size) 
        end
    end

    boundox = upper_bound_func(xinit, vinit, pdmp.tmax)
    state = PDMPState(
        xinit,
        vinit,
        0.0,
        pdmp.tmax,
        key,
        pdmp.integrator,
        pdmp.∇U,
        pdmp.rate,
        pdmp.velocity_jump,
        upper_bound_func,
        boundox,
        pdmp.adaptive
    )
    pdmp.state = state
    return state

end

# function init_state(pdmp::AbstractPDMP, xinit::Union{Float64, Int}, vinit::Union{Float64, Int}, seed::Int)
#     xinit = Float64(xinit)
#     vinit = Float64(vinit)
#     return init_state(pdmp, [xinit], [vinit], seed)
# end
