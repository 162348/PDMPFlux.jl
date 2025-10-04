using Distributions
using StatsFuns: logsumexp
using LaTeXStrings

# ===== Mixture parameters (自由に変更) =====
w  = [0.5, 0.5]              # weights (sum to 1)
μ  = [-2.0, 2.0]             # means
σ  = [0.7, 0.7]              # stds
@assert isapprox(sum(w), 1.0; atol=1e-10)

norms = [Normal(μ[i], σ[i]) for i in eachindex(μ)]
σ2    = σ .^ 2
logw  = log.(w)

# ===== Potential U(x) = -log p(x) and its derivative U'(x) =====
logp(x) = logsumexp(logw .+ [logpdf(norms[i], x) for i in eachindex(norms)])
U(x)    = -logp(x) + x.^2 / 2  # modified potential for a Boomerang sampler

using PDMPFlux

dim = 1
sampler = BoomerangAD(dim, U, grid_size=10, refresh_rate=0.5)
N_sk, N, xinit, vinit = 100, 100000, [3.0], [3.0]
output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2025934)
samples = sample_from_skeleton(sampler, N, output, discard_vt = false)

function dU(x)
  # U'(x) = (∑ w_k pdf_k(x) * (x - μ_k)/σ_k^2) / (∑ w_k pdf_k(x))
  pdfs = [w[i] * pdf(norms[i], x) for i in eachindex(norms)]
  den  = sum(pdfs) + eps()  # avoid 0
  num  = sum(pdfs[i] * (x - μ[i]) / σ2[i] for i in eachindex(norms))
  return num / den
end

xg = range(-5, 5; length=25)
vg = range(-4, 5; length=25)
DX = [v for x in xg, v in vg]            # dx/dt = v
DV = [-dU(x) for x in xg, v in vg]       # dv/dt = -U'(x)

# normalize arrow lengths for nicer plotting
len = sqrt.(DX.^2 .+ DV.^2) .+ 1e-9
DXn = DX ./ len
DVn = DV ./ len

using CairoMakie

# Create beautiful CairoMakie plot
fig = Figure(size=(900, 550), backgroundcolor=:transparent)
ax = Axis(fig[1, 1], 
          backgroundcolor=:transparent)

traj_x = samples[1,1:7000]
traj_v = samples[2,1:7000]
t = samples[3,1:7000]
N_events = 10
events_t = output.t[1:N_events]
indeces = searchsortedfirst.(Ref(t), events_t) .- 1  # previous index
indeces[end] -= 1

for i in 1:N_events-1
  lines!(ax, traj_x[indeces[i]+1:indeces[i+1]], traj_v[indeces[i]+1:indeces[i+1]], 
        color="#E95420", 
        linewidth=2.5,
        label="Trajectory")
end

for i in 1:N_events-2
  x_start = traj_x[indeces[i+1]]
  v_start = traj_v[indeces[i+1]]
  x_end = traj_x[indeces[i+1]+1]
  v_end = traj_v[indeces[i+1]+1]
  
  # 矢印の方向ベクトル
  dx = x_end - x_start
  dv = v_end - v_start
  # 矢印の長さを計算
  total_length = sqrt(dx^2 + dv^2)
  # 0.5 point短くするためのスケールファクター
  scale_factor = (total_length - 0.3) / total_length
  
  arrows!(ax, [x_start], [v_start], [dx], [dv * scale_factor];
          arrowsize=15, lengthscale=1, linewidth=2, color="#E95420", linestyle=:dash)
  scatter!(ax, [x_end], [v_end], color="#E95420", markersize=12)
  scatter!(ax, [x_start], [v_start], color="#E95420", markersize=12)
end

# Add vector field
arrows!(ax, repeat(collect(xg), outer=length(vg)),
            repeat(collect(vg), inner=length(xg)),
            vec(DXn), vec(DVn),
            arrowsize=12, 
            lengthscale=0.2, 
            linewidth=1.5, 
            color=(:gray, 0.6))

# Add starting point
CairoMakie.scatter!(ax, [traj_x[1]], [traj_v[1]], 
         color="#E95420", 
         markersize=12, 
         marker=:star5)

# Add ending point 

# Set axis limits and styling
CairoMakie.xlims!(ax, -5, 5)
CairoMakie.ylims!(ax, -4, 5)
hidexdecorations!(ax, grid=true)  # optional styling
hideydecorations!(ax, grid=true)

xs = range(-5, 5; length=401)
vs = range(-4, 5; length=401)
Ux = U.(xs)
Kv = 0.5 .* (vs .^ 2)
H = [Ux[i] + Kv[j] for i in eachindex(xs), j in eachindex(vs)]
custom_palette = cgrad([:white, :black])
contour!(ax, xs, vs, H; levels=30, colormap=custom_palette)

# Add legend

# Display the plot
save("Boomerang_flow.svg", fig; px_per_unit=2)
fig

# jointplot(samples)