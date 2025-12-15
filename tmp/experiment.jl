using PDMPFlux

function ∇U(x::Vector)  # σ=1 とした
    return x
end

dim = 1
sampler = ZigZag(dim, ∇U, grid_size=10)

N_sk, N, xinit, vinit = 1, 1, zeros(dim), ones(dim)

first_event_times = []
for i in 1:100000
  output = sample_skeleton(sampler, N_sk, xinit, vinit, verbose=false)
  push!(first_event_times, output.t[2])
end

mean(first_event_times)  # should be sqrt(2π)/2
println(sqrt(2π)/2)

output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2024)
samples = sample_from_skeleton(sampler, N, output)

