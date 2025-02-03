using PDMPFlux

function U_Gauss(x::Vector)
  return -log(exp(-sum((x .- 10.0).^2) / 2) + exp(-sum((x .+ 10.0).^2) / 2))
end

dim = 2
sampler = ZigZagAD(dim, U_Gauss, grid_size=10)

N_sk, N, xinit, vinit = 150, 150, zeros(dim), ones(dim)

output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2024)

anim_traj(output, 100; filename="ZigZag_GaussianMixture2D.gif", title="Zig-Zag Sampler")