function ∇U(x::Vector)
  return [x[1] - x[2]/2, x[2] - x[1]/2]
end

dim = 2
xinit, vinit = zeros(dim), ones(dim)

x, dt = [xinit], 0.0001
using Random, ProgressBars, Plots

for i in ProgressBar(1:10000)
  xᵢ = x[end] .- ∇U(x[end]) .* dt .+ sqrt(dt) .* randn(dim)
  push!(x, xᵢ)
end

x_matrix = hcat(x...)
x₁, x₂ = x_matrix[1,:], x_matrix[2,:]
args = (
  xlims=(-1, 1),
  ylims=(-1, 1),
  label=false,
  axis=false,
  title="Langevin Diffusion",
  color="#0096FF",
  background="#F0F1EB",
  linewidth=2,
  aspect_ratio=1.2,
  size=(400, 600),
  )

p = plot(x₁[1:1], x₂[1:1]; args...)

@gif for i in 1:10000
  Base.push!(p, x₁[i], x₂[i])
end every 10