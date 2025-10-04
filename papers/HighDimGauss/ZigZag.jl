using PDMPFlux

function U(x::AbstractVector)
    return sum(x.^2) / 2
end

dim = 1000
sampler_ZZ = ZigZagAD(dim, U, grid_size=10)

N_sk, N, xinit, vinit = 1_000_000, 1_000_000, ones(dim), ones(dim)

output_ZZ = sample_skeleton(sampler_ZZ, N_sk, xinit, vinit, seed=20250428)
x_ZZ = hcat(output_ZZ.x...) # dim×100001 Matrix{Float64}
v_ZZ = hcat(output_ZZ.v...) # dim×100001 Matrix{Float64}
t_ZZ = output_ZZ.t # 100001-element Vector{Float64}

### 第一成分の計算

plot(t_ZZ, x_ZZ[1,:],
     xlabel=L"t",
     ylabel=L"x_1",
     label="First Component",
     linewidth=1,
     grid=true,           # グリッド線を表示
     title="First Component of Zig-Zag, d=$(dim)",  # タイトルを追加
     color="#78C2AD",         # 線の色を設定
     legend=false)     # プロットのサイズを設定

savefig("ZigZag_$(dim)d_FirstComponent.svg")

### 動径運動量の計算

s_ZZ = sum(x_ZZ .* v_ZZ, dims=1)
p = plot(t_ZZ, vec(s_ZZ),
     xlabel=L"t",
     ylabel=L"⟨x|v⟩",
     label="Inner Product",
     linewidth=1,
     grid=true,           # グリッド線を表示
     title="Momentum of Zig-Zag, d=$(dim)",  # タイトルを追加
     color="#78C2AD",         # 線の色を設定
     legend=false)     # プロットのサイズを設定

savefig("ZigZag_$(dim)d_Momentum.svg")

### ポテンシャルの計算

function Ψ(x::AbstractVector)
    return (sum(x.^2) - dim) / sqrt(dim)
end
U_values = map(Ψ, eachcol(x_ZZ))

plot(t_ZZ, U_values,
     xlabel=L"t",
     ylabel=L"U(x)",
     label="Potential",
     linewidth=1,
     grid=true,           # グリッド線を表示
     title="Potential of Zig-Zag, d=$(dim)",  # タイトルを追加
     color="#78C2AD",         # 線の色を設定
     xlims = (0,2500),
     legend=false)     # プロットのサイズを設定

savefig("ZigZag_$(dim)d_Potential.svg")
