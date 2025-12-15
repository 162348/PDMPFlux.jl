using PDMPFlux

function ∇U(x::Vector)  # σ=1 とした
  denominator = (1-γ) * exp(-x[1]^2 / 2) / sqrt(2π) + γ * exp(-x[1]^2 / (2 * ϵ^2)) / (sqrt(2π) * ϵ)
  numerator = x[1] * (1-γ) * exp(-x[1]^2 / 2) / sqrt(2π) + γ * x[1] * exp(-x[1]^2 / (2 * ϵ^2)) / (sqrt(2π) * ϵ^3)
  return numerator / denominator
end

dim = 1
sampler = ZigZag(dim, ∇U, grid_size=10)

N_sk, N, xinit, vinit = 100, 100, zeros(dim), ones(dim)

ϵ = 0.01
γ = 0.5

first_exit_times = []
using ProgressBars
for i in ProgressBar(1:10000)
  output = sample_skeleton(sampler, N_sk, xinit, vinit, verbose=false)
  i = findfirst(x -> abs(x[1]) > ϵ, output.x)
  push!(first_exit_times, output.t[i])
end

mean(first_exit_times)
using Distributions
d = Normal(0, 1)
integral = (cdf(d, 1) - cdf(d, -1)) * sqrt(2π)
expected_value = ϵ * (sqrt(ℯ) * integral + 1)
println(expected_value)

output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2024)
samples = sample_from_skeleton(sampler, N, output)

