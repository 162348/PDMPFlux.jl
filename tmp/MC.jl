function U(x::Vector)
  return (sum(x.^2) - x[1]*x[2]) / 2
end

function π(x::Vector)
  return exp(-U(x))
end

xinit = zeros(2)
x = [xinit]
using Random, Plots, ProgressBars

for i ∈ 1:1000
  xᵢ = x[end] .+ 0.8 * randn(2)
  α = min(1, π(xᵢ) / π(x[end]))
  if rand() < α
    push!(x, xᵢ)
  else
    push!(x, x[end])
  end
end

x_matrix = hcat(x...)
x₁, x₂ = x_matrix[1,:], x_matrix[2,:]
args = (
  xlims=(-2, 2.5),
  ylims=(-3, 3),
  label=false,
  axis=false,
  title="Random Walk Metropolis",
  color=:black,
  background="#F0F1EB",
  linewidth=1,
  ls=:dash,
  dash_pattern="on 1cm off 3cm",
  aspect_ratio=1.2,
  size=(400, 600),
  linealpha=0.5,
  )

p = plot(x₁[1:1], x₂[1:1]; args...)

markersize = Ref(4)

anim = @animate for i ∈ ProgressBar(1:100)
  Base.push!(p, x₁[i], x₂[i])
  if i > 1 && x₁[i] == x₁[i-1] && x₂[i] == x₂[i-1]
    markersize[] += 4
    scatter!(p, x₁[i:i], x₂[i:i], marker=:circle, markersize=markersize[], markeralpha=1, color="#78C2AD", label=false)
  else
    markersize[] = 4
    scatter!(p, x₁[i:i], x₂[i:i], marker=:circle, markersize=markersize[], markeralpha=1, color="#78C2AD", label=false)
  end
end

gif(anim, "RWMH.gif", fps=2)