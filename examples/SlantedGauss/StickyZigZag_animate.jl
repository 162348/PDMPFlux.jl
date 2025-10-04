using PDMPFlux

function U_Gauss(x::Vector)
  return (sum(x.^2) - x[1]*x[2]) / 2
end

function U_Cauchy(x)
  return log(1 + x^2)
end

function U_Banana(x::Vector)
  mean_x2 = (x[1]^2 - 1)
  return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
end

dim = 2
sampler = StickyZigZagAD(dim, U_Gauss, [0.1, 1.5], grid_size=10)
# sampler = StickyZigZagAD(dim, U_Cauchy, [15.0], grid_size=10)

N_sk, N, xinit, vinit = 10000, 10000, zeros(dim), ones(dim)

output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2023)
samples = sample_from_skeleton(sampler, N, output)
x = hcat(output.x...)

p = plot_traj(output, 100)
using Plots
scatter!(p, samples[1, 1:110], samples[2, 1:110], color=:red, label="")

# diagnostic(output)
# jointplot(samples)

using LaTeXStrings
plot_traj(output, 20, filename="StickyZigZag.svg", background="#F0F1EB", title="Sticky Zig-Zag Sampler " * L"\kappa = [0.1, 1.5]")
# anim_traj(output, 30, filename="StickyZigZag.gif", title="Sticky Zig-Zag Sampler " * L"\kappa = [0.1, 1.5]", background="#F0F1EB", color="#E95420")
# anim_traj(output, 30, filename="StickyZigZag_1st.gif", title="Sticky Zig-Zag Sampler " * L"\kappa = [0.1, 1.5]", plot_type="1D", coordinate_numbers=[1])
# anim_traj(output, 30; plot_start=10, filename="StickyZigZag_Cauchy1D.gif", title="Sticky Zig-Zag Sampler " * L"\kappa = 15")

# ForwardDiff: なぜかサンプルの精度が酷い
## grid 10 + vectorized 1:23 / 1.18, Mean 0.94
## grid 10 + non-vectorized 1:35, Mean 0.94
## Brent 5:06, Mean ? → サンプルを 1/10 にして 0:12, Mean 0.87
## grid 100 + vectorized 6:35, Mean 0.94

# Zygote
## grid 10 + vectorized 0:48, Mean 0.966...
