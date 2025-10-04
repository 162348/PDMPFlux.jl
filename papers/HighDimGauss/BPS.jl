using PDMPFlux

function U(x::AbstractVector)
    return sum(x.^2) / 2
end

dim = 10000
sampler_BPS = BPSAD(dim, U, grid_size=10, refresh_rate = 1.0)

N_sk, N, xinit, vinit = 100_000, 10_000, randn(rng, dim), randn(rng, dim)
vinit = vinit ./ sqrt(sum(vinit.^2))

output_BPS = sample_skeleton(sampler_BPS, N_sk, xinit, vinit, seed=20250428)
x_BPS = hcat(output_BPS.x...) # dim×100001 Matrix{Float64}
v_BPS = hcat(output_BPS.v...) # dim×100001 Matrix{Float64}
t_BPS = output_BPS.t # 100001-element Vector{Float64}

### 第一成分の計算

plot(t_BPS, x_BPS[1,:],
     xlabel=L"t",
     ylabel=L"x_1",
     label="First Component",
     linewidth=1,
     grid=true,           # グリッド線を表示
     title="First Component of BPS, d=$(dim)",  # タイトルを追加
     color="#78C2AD",         # 線の色を設定
     legend=false)     # プロットのサイズを設定

savefig("BPS_$(dim)d_FirstComponent.svg")

### 動径運動量の計算

s_BPS = sum(x_BPS .* v_BPS, dims=1)
plot(t_BPS, vec(s_BPS),
     xlabel=L"t",
     ylabel=L"⟨x|v⟩",
     label="Inner Product",
     linewidth=1,
     grid=true,           # グリッド線を表示
     title="Momentum of BPS, d=$(dim)",  # タイトルを追加
     color="#78C2AD",         # 線の色を設定
     legend=false)     # プロットのサイズを設定


v_norms = sqrt.(sum(v_BPS.^2, dims=1))
    
# anim_traj(output, 20; filename="FECMC_HighDimGauss2D.gif", title="Forward ECMC Sampler")

savefig("BPS_$(dim)d_Momentum.svg")

### ポテンシャルの計算

function Ψ(x::AbstractVector)
  return (sum(x.^2) - dim) / sqrt(dim)
end
U_values = map(Ψ, eachcol(x_BPS))

plot(t_BPS, U_values,
     xlabel=L"t",
     ylabel=L"U(x)",
     label="Potential",
     linewidth=1,
     grid=true,           # グリッド線を表示
     title="Potential of BPS, d=$(dim)",  # タイトルを追加
     color="#78C2AD",         # 線の色を設定
     legend=false)     # プロットのサイズを設定

savefig("BPS_$(dim)d_Potential.svg")


anim_traj(output_BPS, 300; filename="BPS_$(dim)d_Trajectory.gif", title="Bouncy Particle Sampler")
plot_traj(output_BPS, 500; filename="BPS_unnormalized.svg", title="Bouncy Particle Sampler with Unnormalized Velocities")