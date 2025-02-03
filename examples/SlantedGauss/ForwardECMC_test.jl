using PDMPFlux

function U_Gauss(x::Vector)
    return (sum(x.^2)) / 2
end

function U_Banana(x::Vector)
  mean_x2 = (x[1]^2 - 1)
  return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
end

dim = 3
sampler = ForwardECMCAD(dim, U_Gauss, grid_size=10)
# sampler = ZigZagAD(dim, U_Gauss, grid_size=10, signed_bound=false)

N_sk, N, xinit, vinit = 100, 100, zeros(dim), ones(dim)

output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2024)
samples = sample_from_skeleton(sampler, N, output)

# diagnostic(output)
jointplot(samples)

anim_traj(output, 30; filename="ForwardECMC_StandardGauss2D.gif", title="Forward ECMC Sampler", plot_type="2D", background="#F0F1EB", color="#E95420")

function animate(filename::String; title::String, dim::Int=3, N_sk::Int=100, N::Int=100,
  xinit::Vector{Float64}=ones(dim), vinit::Vector{Float64}=ones(dim))

  sampler = ForwardECMCAD(dim, U_Gauss, grid_size=10)
  output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2024)
  anim_traj(output, 100; filename=filename, title=title)
end

v = hcat(output.v...)
# check = []
# histogram(v[1,:])

# animate("ForwardECMC_SlantedGauss2D.gif"; title="Forward ECMC Sampler", dim=2)

# ForwardDiff: なぜかサンプルの精度が酷い
## grid 10 + vectorized 1:23 / 1.18, Mean 0.94
## grid 10 + non-vectorized 1:35, Mean 0.94
## Brent 5:06, Mean ? → サンプルを 1/10 にして 0:12, Mean 0.87
## grid 100 + vectorized 6:35, Mean 0.94

# Zygote
## grid 10 + vectorized 0:48, Mean 0.966...
