using PDMPFlux

using Zygote, Random, Plots, LaTeXStrings

function U_Gauss(x::Union{Float64, Int})
    x = Float64(x)
    return x^2 / 2
end

function U_Cauchy(x::Union{Float64, Int})
    x = Float64(x)
    return log(1 + x^2)
end

function plot_densities()
    # プロット範囲の設定
    x_values = -10:0.1:10

    # U_Gauss と U_Cauchy の値を計算
    y_gauss = [exp(-U_Gauss(x)) for x in x_values]
    y_cauchy = [exp(-U_Cauchy(x)) for x in x_values]
    plot(x_values, y_gauss, label="Gaussian density", xlabel=L"x", ylabel=L"p(x)", title="Gaussian vs Cauchy", color="#78C2AD")
    plot!(x_values, y_cauchy, label="Cauchy density", color="#E95420")
end

dim = 1
seed = 8
key = MersenneTwister(seed)
xinit = 1.0
vinit = 1.0

grid_size = 0  # constant bounds

N_sk = 100_000 # number of skeleton points
N = 100_000 # number of samples

sampler_ad = ZigZagAD(dim, U_Cauchy, grid_size=grid_size)
# samples_ad = sample(sampler_ad, N_sk, N, xinit, vinit, seed=seed, verbose = true)
out_ad = sample_skeleton(sampler_ad, N_sk, xinit, vinit, seed=seed, verbose = true)
plot_traj(out_ad, 300)
anim_traj(out_ad, 50, filename="Cauchy1D.gif", fps=60, plot_start=100)
diagnostic(out_ad)

"""
* ∇U も Vector 値に対して定義される必要がある．
* これを内部化してしまうという手もある．
"""

# new_variable = 0.0

# try
#     U_Gauss([1.0])
# catch e
#     new_variable = 1.0
#     println(e)
# end

# ∇U(x::Union{Float64, Int}) = gradient(U_Gauss, x)[1]
# ∇U(x::Vector{Float64}) = gradient(U_Gauss,x[1])[1]

# sampler = ZigZag(dim, ∇U, grid_size=grid_size)
# out = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed, verbose = true)
# samples = sample_from_skeleton(sampler, N, out)