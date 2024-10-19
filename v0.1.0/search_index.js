var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = PDMPFlux","category":"page"},{"location":"#PDMPFlux","page":"Home","title":"PDMPFlux","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for PDMPFlux.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [PDMPFlux]","category":"page"},{"location":"#PDMPFlux.AbstractPDMP","page":"Home","title":"PDMPFlux.AbstractPDMP","text":"In PDMPFlux, every PDMP sampler is defined to be of subtype of AbstractPDMP.\n\n\n\n\n\n","category":"type"},{"location":"#PDMPFlux.BoundBox","page":"Home","title":"PDMPFlux.BoundBox","text":"BoundBox <: Any\n\n上界関数の出力を格納するための構造体．\n\nAttributes:\n    grid (Float64 Array{T,1}): The grid values.\n    box_max (Float64 Array{T,1}): The maximum values on each segment of the grid.\n    cum_sum (Float64 Array{T,1}): The cumulative sum of box_max.\n    step_size (Float64): The step size of the grid.\n\n\n\n\n\n","category":"type"},{"location":"#PDMPFlux.PDMPState","page":"Home","title":"PDMPFlux.PDMPState","text":"PDMPState <: Any\n\nAbstractPDMP の状態空間の元．次のフィールドを持つ構造体として実装：\n\nAttributes:\n    x (Array{Float64, 1}): position\n    v (Array{Float64, 1}): velocity\n    t (Float64): time\n    horizon (Float64): horizon\n    key (Any): random key\n    integrator (Function): integrator function\n    grad_U (Function): gradient of the potential function\n    rate (Function): rate function\n    velocity_jump (Function): velocity jump function\n    upper_bound_func (Function): upper bound function\n    accept (Bool): accept indicator for the thinning\n    upper_bound (Union{Nothing, NamedTuple}): upper bound box\n    indicator (Bool): indicator for jumping\n    tp (Float64): time to the next event\n    ts (Float64): time spent\n    exp_rv (Float64): exponential random variable for the Poisson process\n    lambda_bar (Float64): upper bound for the Poisson process\n    lambda_t (Float64): rate at the current time\n    ar (Float64): acceptance rate for the thinning\n    error_bound (Int): count of the number of errors in the upper bound\n    rejected (Int): count of the number of rejections in the thinning\n    hitting_horizon (Int): count of the number of hits of the horizon\n    adaptive (Bool): adaptive indicator\n\n\n\n\n\n","category":"type"},{"location":"#PDMPFlux.ZigZag","page":"Home","title":"PDMPFlux.ZigZag","text":"ZigZag(dim::Int, grad_U::Function; grid_size::Int=10, tmax::Float64=1.0, \n    vectorized_bound::Bool=true, signed_bound::Bool=true, adaptive::Bool=true, kwargs...)\n\nZigZag クラスは ZigZag サンプラーのためのクラスです。\n\n引数\n\ndim::Int: 空間の次元。\ngrad_U::Function: ポテンシャルエネルギー関数の勾配。\ngrid_size::Int: 空間を離散化するためのグリッドポイントの数。デフォルトは10。\ntmax::Float64: グリッドのホライズン。デフォルトは1.0。0の場合、適応的なtmaxが使用されます。\nvectorized_bound::Bool: 境界にベクトル化された戦略を使用するかどうか。デフォルトはtrue。\nsigned_bound::Bool: 符号付き境界戦略を使用するかどうか。デフォルトはtrue。\nadaptive::Bool: 適応的なtmaxを使用するかどうか。デフォルトはtrue。\nkwargs...: その他のキーワード引数。\n\n属性\n\ndim::Int: 空間の次元。\nrefresh_rate::Float64: リフレッシュレート。\ngrad_U::Function: ポテンシャルの勾配。\ngrid_size::Int: 空間を離散化するためのグリッドポイントの数。\ntmax::Float64: グリッドのtmax。\nadaptive::Bool: 適応的なtmaxを使用するかどうか。\nvectorized_bound::Bool: ベクトル化された戦略を使用するかどうか。\nsigned_bound::Bool: 符号付き戦略を使用するかどうか。\nintegrator::Function: インテグレータ関数。\nrate::Array: プロセスのレート。\nrate_vect::Array: ベクトル化されたレート。\nsigned_rate::Array: 符号付きレート。\nsigned_rate_vect::Array: ベクトル化され符号付きのレート。\nvelocity_jump::Function: 速度ジャンプ関数。\nstate: ZigZagサンプラーの状態。\n\n\n\n\n\n","category":"type"},{"location":"#PDMPFlux.error_acceptance-Tuple{PDMPFlux.PDMPState}","page":"Home","title":"PDMPFlux.error_acceptance","text":"代理上界 lambda_bar で足りなかった場合は，ここで horizon を縮めて再度 Poisson 剪定を行う．\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.if_accept-Tuple{PDMPFlux.PDMPState}","page":"Home","title":"PDMPFlux.if_accept","text":"代理上界 lambda_bar を用いた剪定で accept された場合の処置\nここで one_step_while() を終了するために indicator = true とされる．\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.if_not_accept-Tuple{PDMPFlux.PDMPState}","page":"Home","title":"PDMPFlux.if_not_accept","text":"代理上界 lambda_bar を用いた剪定で accept されなかった場合の処置\nhorizon を超えるまで Poisson 剪定を繰り返す．\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.init_state-Tuple{PDMPFlux.AbstractPDMP, Array{Float64}, Array{Float64}, Int64}","page":"Home","title":"PDMPFlux.init_state","text":"init_state():\nPDMP オブジェクトの状態を初期化する．\n\nArgs:\n    xinit (Float[Array, \"dim\"]): The initial position.\n    vinit (Float[Array, \"dim\"]): The initial velocity.\n    seed (int): The seed for random number generation.\n    upper_bound_vect (bool, optional): Whether to use vectorized upper bound function. Defaults to False.\n    signed_rate (bool, optional): Whether to use signed rate function. Defaults to False.\n    adaptive (bool, optional): Whether to use adaptive upper bound. Defaults to False.\n    constant_bound (bool, optional): Whether to use constant upper bound. Defaults to False.\n\nReturns:\n    PdmpState: The initialized PDMP state.\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.inner_while-Tuple{PDMPFlux.PDMPState}","page":"Home","title":"PDMPFlux.inner_while","text":"lambda_bar は正確な上界ではなく，grid が粗い場合に足りない可能性がある．\nその場合は error_acceptance() で補正する．\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.move_before_horizon-Tuple{PDMPFlux.PDMPState}","page":"Home","title":"PDMPFlux.move_before_horizon","text":"horizon より先に event が起こった場合の処理\ninner_while() の繰り返しとして実装される．\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.move_to_horizon-Tuple{PDMPFlux.PDMPState}","page":"Home","title":"PDMPFlux.move_to_horizon","text":"event が horizon の先に起こった場合，もう一度 Poisson simulation を行う．\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.move_to_horizon2-Tuple{PDMPFlux.PDMPState}","page":"Home","title":"PDMPFlux.move_to_horizon2","text":"代理上界 lambda_bar を使った Poisson 剪定中に horizon を超えた場合の動き\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.next_event-Tuple{PDMPFlux.BoundBox, Float64}","page":"Home","title":"PDMPFlux.next_event","text":"next_event(boundbox, exp_rv):\nBoundBox オブジェクトを用いて次のイベント時間を Poisson 剪定によりシミュレーションする．\nイベント時刻 t_prop とその直前の grid 点での上界の値を返す．\n\nArgs:\n    boundbox: The boundbox object containing the cumulative sum and grid values.\n    exp_rv: The exponential random variable.\n\nReturns:\n    A tuple containing the next event time (t_prop) and the corresponding upper bound value.\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.ok_acceptance-Tuple{PDMPFlux.PDMPState}","page":"Home","title":"PDMPFlux.ok_acceptance","text":"代理上界 lambda_bar で足りた場合は，ここで簡単に Poisson 剪定を行う．\nPoisson 剪定中に horizon を超えた場合は move_to_horizon2() を呼び出す．\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.one_step-Tuple{PDMPFlux.PDMPState}","page":"Home","title":"PDMPFlux.one_step","text":"one_step(state::PDMPState)::PDMPState\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.one_step_while-Tuple{PDMPFlux.PDMPState}","page":"Home","title":"PDMPFlux.one_step_while","text":"state.indicator が false である限り実行する処理\nmove_before_horizon() で ok_acceptance() が呼ばれるまで繰り返す．\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.sample-Tuple{PDMPFlux.AbstractPDMP, Int64, Int64, Vector{Float64}, Vector{Float64}}","page":"Home","title":"PDMPFlux.sample","text":"sample()：PDMPSampler からサンプルをするための関数．\nsample_skeleton() と sample_from_skeleton() の wrapper．\n\nArgs:\n    N_sk (Int): Number of skeleton points to generate.\n    N_samples (Int): Number of final samples to generate from the skeleton.\n    xinit (Array{Float64, 1}): Initial position.\n    vinit (Array{Float64, 1}): Initial velocity.\n    seed (Int): Seed for random number generation.\n    verbose (Bool, optional): Whether to print progress information. Defaults to true.\n\nReturns:\n    Array{Float64, 2}: Array of samples generated from the PDMP model.\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.sample_from_skeleton-Tuple{PDMPFlux.AbstractPDMP, Int64, PDMPFlux.PDMPHistory}","page":"Home","title":"PDMPFlux.sample_from_skeleton","text":"スケルトンからサンプリングをし，各行ベクトルに次元毎の時系列が格納された Matrix{Float64} を返す．\n\nArgs:     N (Int): The number of samples to generate.     output (PdmpOutput): The PDMP output containing the trajectory information.\n\nReturns:     Array{Float64, 2}: The sampled points from the PDMP trajectory skeleton.\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.sample_skeleton-Tuple{PDMPFlux.AbstractPDMP, Int64, Vector{Float64}, Vector{Float64}}","page":"Home","title":"PDMPFlux.sample_skeleton","text":"sample_skeleton(): PDMP Samplers からスケルトンを抽出する．\n\nParameters:\n- n_sk (Int): The number of skeleton samples to generate.\n- xinit (Array{Float64, 1}): The initial position of the particles.\n- vinit (Array{Float64, 1}): The initial velocity of the particles.\n- seed (Int): The seed value for random number generation.\n- verbose (Bool): Whether to display progress bar during sampling. Default is true.\n\nReturns:\n- output: The output state of the sampling process.\n\n\n\n\n\n","category":"method"},{"location":"#PDMPFlux.upper_bound_constant","page":"Home","title":"PDMPFlux.upper_bound_constant","text":"upper_bound_constant(func, a, b, n_grid=100, refresh_rate=0.0)\nComputes the constant upper bound using the Brent's algorithm.\nBrent のアルゴリズムを通じて定数でバウンドすることを試みる．必然的に n_grid=2．\n\nParameters:\n- func: The function for which the upper bound constant is computed.\n- a: The lower bound of the interval.\n- b: The upper bound of the interval.\n- n_grid: The number of grid points used for computation (default: 100).\n- refresh_rate: The refresh rate for the upper bound constant (default: 0).\n\nReturns:\n- Tuple: A tuple containing the grid, box_max, cum_sum, and interval length.\n\n\n\n\n\n","category":"function"}]
}
