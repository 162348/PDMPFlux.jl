using PDMPFlux
include("utils.jl")
using Main.MyUtils

function U(x::AbstractVector)
    return sum(x.^2) / 2
end

using Random
rng = MersenneTwister(20250428)

dim = 10000
sampler_FECMC = ForwardECMCAD(dim, U, grid_size=10, signed_bound=false, mix_p=1.0)

N_sk, N, xinit, vinit = 100_000, 10_000, randn(rng, dim), randn(rng, dim)
vinit = vinit ./ sqrt(sum(vinit.^2))

output_FECMC = sample_skeleton(sampler_FECMC, N_sk, xinit, vinit, seed=20250428)
using JLD2
@save "FECMC_output_$(dim)d.jld2" output_FECMC
@load "FECMC_output_10000d.jld2" output_FECMC
x_FECMC = hcat(output_FECMC.x...) # dim×100001 Matrix{Float64}
v_FECMC = hcat(output_FECMC.v...) # dim×100001 Matrix{Float64}
t_FECMC = output_FECMC.t # 100001-element Vector{Float64}

@save "FECMC_x_$(dim)d.jld2" x_FECMC
@save "FECMC_v_$(dim)d.jld2" v_FECMC
@save "FECMC_t_$(dim)d.jld2" t_FECMC
@load "FECMC_x_10000d.jld2" x_FECMC
@load "FECMC_v_10000d.jld2" v_FECMC
@load "FECMC_t_10000d.jld2" t_FECMC

### 第一成分の計算

using Plots, LaTeXStrings

plot(t_FECMC, x_FECMC[1,:],
     xlabel=L"t",
     ylabel=L"x_1",
     label="First Component",
     linewidth=1,
     grid=true,           # グリッド線を表示
     title="First Component of FECMC, d=$(dim)",  # タイトルを追加
     color="#78C2AD",         # 線の色を設定
     xlims=(0, 50000),
     legend=false)     # プロットのサイズを設定

savefig("FECMC_$(dim)d_FirstComponent50000.svg")

### 動径運動量の計算

# using LinearAlgebra
# s = diag(x' * v)  # ありえん重い

s_FECMC, t_FECMC_mixed = get_radial_momentum(sampler_FECMC, x_FECMC[:,1:20], v_FECMC[:,1:20], t_FECMC[1:20])

plot(t_FECMC_mixed[1:20], s_FECMC[1:20],
     xlabel=L"t",
     ylabel=L"⟨x|v⟩",
     label="Inner Product",
     linewidth=1,
     grid=true,           # グリッド線を表示
     title="Momentum of FECMC, d=$(dim)",  # タイトルを追加
     color="#78C2AD",         # 線の色を設定
     legend=false)     # プロットのサイズを設定

# anim_traj(output_FECMC, 20; filename="temp.gif")

savefig("FECMC_$(dim)d_Momentum.svg")

### ポテンシャルの計算

function Ψ(x::AbstractVector)
    return (sum(x.^2) - dim) / sqrt(dim)
end
U_values = map(Ψ, eachcol(x_FECMC))

@save "FECMC_U_$(dim)d_sqrt_scale.jld2" U_values
@load "FECMC_U_10000d_sqrt_scale.jld2" U_values

plot(t_FECMC[1:20], U_values[1:20],
     xlabel=L"t",
     ylabel=L"U(x)",
     label="Potential",
     linewidth=1,
     grid=true,           # グリッド線を表示
     title="Potential of FECMC, d=$(dim)",  # タイトルを追加
     color="#78C2AD",         # 線の色を設定
     legend=false)     # プロットのサイズを設定

savefig("FECMC_$(dim)d_Potential_sqrt_scale.svg")

anim_traj(output_FECMC, 300; filename="FECMC_$(dim)d_Trajectory_sqrt_scale.gif", title="Forward Event Chain Monte Carlo")

function get_radial_momentum(sampler, x, v, t)
flow = sampler.flow
tt = t[2:end] .- 0.01
tt_ = tt - t[1:end-1]
tuple = map(flow, eachcol(x[:,1:end-1]), eachcol(v[:,1:end-1]), tt_)
xx = hcat(getindex.(tuple, 1)...)
vv = hcat(getindex.(tuple, 2)...)

s = vec(sum(x .* v, dims=1))
ss = vec(sum(xx .* vv, dims=1))

s_mixed = collect(Iterators.flatten(zip(vec(s), vec(ss))))
t_mixed = collect(Iterators.flatten(zip(vec(t), vec(tt))))

return s_mixed, t_mixed
end