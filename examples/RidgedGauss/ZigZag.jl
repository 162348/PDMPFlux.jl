using PDMPFlux

function ∇U(θ::Float64, ϵ::Float64)
    y₁(x) = x[1] * cos(θ) - x[2] * sin(θ)
    y₂(x) = x[1] * sin(θ) + x[2] * cos(θ)
    return (x) -> [y₁(x) * cos(θ) + ϵ^(-1) * y₂(x) * sin(θ), -y₁(x) * sin(θ) + ϵ^(-1) * y₂(x) * cos(θ)]
end

dim, θ, ϵ = 2, π/4, 0.01
sampler = ZigZag(dim, ∇U(θ, ϵ), grid_size=10)
# sampler = ZigZagAD(dim, U_Gauss, grid_size=10, signed_bound=false)

N_sk, N, xinit, vinit = 100_000, 100_000, zeros(dim), ones(dim)

output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2022)

# diagnostic(output)
# jointplot(samples)

using LaTeXStrings
# anim_traj(output, 300; filename="ZigZag_30.gif", title=L"\theta = \frac{\pi}{6}, \epsilon = 0.01")

using Plots

# なぜか y₁ が行き切らない！半直線に囚われて 0 を通らない！
x = hcat(output.x...)
y₁ = x[1,:] * cos(θ) .- x[2,:] * sin(θ)
plot(output.t[1:100], y₁[1:100], ylims=(-2.9,2.9))

# y₂ = ϵ^(-1) * (x[1,:] * sin(θ) + x[2,:] * cos(θ))
# plot(output.t[1:100], y₂[1:100])

# v = hcat(output.v...)
# plot(output.t[1:100], v[1,1:100])
# plot(output.t[1:100], v[2,1:100])
