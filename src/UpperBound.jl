using Optim

"""
    upper_bound_constant(func, a, b, n_grid=100, refresh_rate=0.0)
    Computes the constant upper bound using the Brent's algorithm.
    Brent のアルゴリズムを通じて定数でバウンドすることを試みる．必然的に n_grid=2．

    Parameters:
    - func: The function for which the upper bound constant is computed.
    - a: The lower bound of the interval.
    - b: The upper bound of the interval.
    - n_grid: The number of grid points used for computation (default: 100).
    - refresh_rate: The refresh rate for the upper bound constant (default: 0).

    Returns:
    - Tuple: A tuple containing the grid, box_max, cum_sum, and interval length.
"""
function upper_bound_constant(func::Function, a::Union{Float64,Int}, b::Union{Float64,Int}, refresh_rate::Float64=0.0)
    a, b = Float64(a), Float64(b)
    # Define the function to minimize
    func_min = x -> -func(x)
    
    # Use Brent's method to find the minimum
    result = optimize(func_min, a, b, Brent())
    
    # Create the grid and calculate box_max
    t = collect(LinRange(a, b, 2))  # collect() into Vector{Float64}
    box_max = [-Optim.minimum(result)]  #? Optim.minimum の型はわからないよな？ → func は rate の partial であり，非負値である．
    box_max[1] += refresh_rate
    
    # Calculate cumulative sum
    cum_sum = zeros(2)
    cum_sum[2] = box_max[1] * (b - a)
    
    return BoundBox(t, box_max, cum_sum, b - a)
end

## TODO: Implement upper_bound_grid and upper_bound_grid_vect functions

using ForwardDiff

"""
    upper_bound_grid(func, start=0.0, horizon, n_grid, refresh_rate)
    Compute the upper bound as a piecewise constant function using a grid mechanism.

    Args:
        func: the function for which the upper bound is computed
        start (Float64): the lower bound of the interval
        horizon (Float64): the upper bound of the interval
        n_grid (Int64, optional): size of the grid for the upperbound of func. Defaults to 100.
        refresh_rate (Float64, optional): refresh rate for the upper bound. Defaults to 0.

    Returns:
        BoundBox: An object containing the upper bound constant information.
"""
function upper_bound_grid(func::Function, start::Float64, horizon::Float64, n_grid::Int=100, refresh_rate::Float64 = 0.0; AD_backend::String="ForwardDiff")::BoundBox
    # grid の生成
    t = range(start, stop=horizon, length=n_grid)
    step_size = t[2] - t[1]  # jax と最後の桁の数値が違う
    
    ## grid 上での値と微分係数の計算
    values = map(func, t)  # その結果後ろの方では結構数値誤差が蓄積している可能性があるが，jax と Julia のどっちがより正しいかは不明．
    if AD_backend == "ForwardDiff"
        grads = [ForwardDiff.derivative(func, x) for x in t]
    elseif AD_backend == "Zygote"
        grads = [Zygote.gradient(func, x)[1][1] for x in t]
    end
    
    intersection_pos = (values[1:end-1] .- values[2:end] .+ (grads[2:end] .* step_size)) ./ (grads[2:end] .- grads[1:end-1])
    intersection_pos = replace(intersection_pos, NaN => 0.0)
    intersection_pos = clamp.(intersection_pos, 0.0, step_size)
    
    intersection = values[1:end-1] .+ grads[1:end-1] .* intersection_pos
    box_max = max.(values[1:end-1], values[2:end])
    box_max = max.(box_max, intersection)
    box_max = max.(box_max, 0.0)
    box_max .+= refresh_rate
    
    cum_sum = zeros(Float64, n_grid)
    cum_sum[2:end] = cumsum(box_max) .* step_size
    
    return BoundBox(collect(t), box_max, cum_sum, step_size)
end

using LinearAlgebra

"""
    upper_bound_grid_vect(func, start, horizon, n_grid)
    Compute the upper bound using a grid with the vectorized strategy

    For this function, func(x) takes vector values with the dimension `dim`.

    Args:
        func: the function for which the upper bound is computed
        a (Float64): the lower bound of the interval
        b (Float64): the upper bound of the interval
        n_grid (Int64, optional): size of the grid for the upperbound of func. Defaults to 100.

    Returns:
        BoundBox: An object containing the upper bound constant information.
"""
function upper_bound_grid_vect(func::Function, start::Float64, horizon::Float64, n_grid::Int=100; AD_backend::String="ForwardDiff")::BoundBox
    t = range(start, stop=horizon, length=n_grid)
    step_size = t[2] - t[1]
    
    # Vectorized function evaluation and gradient computation
    values = hcat(map(func, t)...)
    if AD_backend == "ForwardDiff"
        grads = hcat([ForwardDiff.derivative(func, x) for x in t]...)
    elseif AD_backend == "Zygote"
        grads = hcat([Zygote.jacobian(func, x)[1] for x in t]...)
    end
    
    intersection_pos = (values[:,1:end-1] .- values[:,2:end] .+ (grads[:,2:end] .* step_size)) ./ (grads[:,2:end] .- grads[:,1:end-1])
    intersection_pos = replace(intersection_pos, NaN => 0.0)
    intersection_pos = clamp.(intersection_pos, 0.0, step_size)
    
    intersection = values[:,1:end-1] .+ grads[:,1:end-1] .* intersection_pos

    box_max = max.(values[:,1:end-1], values[:,2:end])
    box_max = max.(box_max, intersection)
    box_max = max.(box_max, 0.0)
    
    cum_sum = zeros(size(values))
    cum_sum[:,2:end] = cumsum(box_max, dims=2) .* step_size
    
    return BoundBox(collect(t), vec(sum(box_max, dims=1)), vec(sum(cum_sum, dims=1)), step_size)
end


"""
    next_event(boundbox, exp_rv):
    BoundBox オブジェクトを用いて次のイベント時間を Poisson 剪定によりシミュレーションする．
    イベント時刻 t_prop とその直前の grid 点での上界の値を返す．

    Args:
        boundbox: The boundbox object containing the cumulative sum and grid values.
        exp_rv: The exponential random variable.

    Returns:
        A tuple containing the next event time (t_prop) and the corresponding upper bound value.
"""
function next_event(boundbox::BoundBox, exp_rv::Float64)
    index = searchsortedfirst(boundbox.cum_sum, exp_rv)

    # if the index is the last element, meaning that exp_rv > cum_sum[end], it returns infinity
    t_prop = index > length(boundbox.cum_sum) ? Inf : boundbox.grid[index-1] + (exp_rv - boundbox.cum_sum[index-1]) / (boundbox.cum_sum[index] - boundbox.cum_sum[index-1]) * boundbox.step_size
    upper_bound = index > length(boundbox.cum_sum) ? boundbox.box_max[end] : boundbox.box_max[index-1]

    return t_prop, upper_bound
end

