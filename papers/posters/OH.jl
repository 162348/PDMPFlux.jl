using PDMPFlux
using Plots
using LaTeXStrings

ω = 0.5
κ = (1-ω) / ω
ϵ = 0.01

function U(x::AbstractVector)
  return sum(x.^2) / 2 - log(1 + (κ / ϵ) * exp( sum(x.^2) * (ϵ^2 -1) / (2 * ϵ^2) ))
end

function U(x::Float64)
  return x^2 / 2 - log(1 + (κ / ϵ) * exp( x^2 * (ϵ^2 -1) / (2 * ϵ^2) )) -log(ω) + log(2π) / 2
end

function U_Gauss(x::Float64)
  return x^2 / 2
end

function U_Gauss(x::AbstractVector)
  return sum(x.^2) / 2
end

function p(x::Float64)
  return exp(-U(x))
end

dim = 1
sampler_ZZ = ZigZagAD(dim, U, grid_size=10)
N_sk, N, xinit, vinit = 1000, 1000, ones(dim), ones(dim)
output_ZZ = sample_skeleton(sampler_ZZ, N_sk, xinit, vinit, seed=20250428)

plt = plot_traj(output_ZZ, 150; title="", ylim=(-1,1), ylabel=L"\beta", n_start=10, background="#F0F1EB", color="#E95420", xlim=(30,60))

savefig("ZigZag_c.svg")

sampler_SZZ = StickyZigZagAD(dim, U_Gauss, [κ^-1], grid_size=10)
output_SZZ = sample_skeleton(sampler_SZZ, N_sk, xinit, vinit, seed=20250429)

plt = plot_traj(output_SZZ, 80; title="", ylim=(-1,1), ylabel=L"\beta", n_start=15, background="#F0F1EB", color="#E95420",xlim=(30,60))

savefig("StickyZigZag_d.svg")

# プロット用のグリッドを作成
x_range = -3:0.1:3
X = [x for x in x_range]

# 確率密度関数の値を計算
Y = [p(x) for x in x_range]

function p_exact(x::Float64)
  return 1 / (2π) * exp(-x^2 / 2)
end

# プロット
plt = plot(X, p_exact.(X), 
    xlabel=L"\beta", 
    ylim=(0,0.5),
    label=L"\pi_d",
    linewidth=2,
    color="#E95420",
    background_color="#F0F1EB",
    legendfontsize=15,
    )

# x=0の位置に赤い線を追加
plot!([0, 0], [0.160, 0.5], 
    linewidth=1.8,
    color="#E95420",
    label="")

# プロットを保存
savefig(plt, "pd.png")