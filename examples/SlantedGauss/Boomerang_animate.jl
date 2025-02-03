using PDMPFlux

function U_Gauss(x::Vector)
    return (sum(x.^2) - x[1]*x[2]) / 2
end

dim = 2
sampler = BoomerangAD(dim, U_Gauss, refresh_rate=0.025, grid_size=10)
# sampler = ZigZagAD(dim, U_Gauss, grid_size=10, signed_bound=false)

N_sk, N, xinit, vinit = 100_000, 100_000, zeros(dim), [1.2,-3.4]

output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2024)
samples = sample_from_skeleton(sampler, N, output)

# diagnostic(output)
jointplot(samples)

anim_traj(output, 20; filename="Boomerang_SlantedGauss2D.gif", title="Boomerang Sampler", nonlinear_flow=sampler.flow)
# anim_traj(output, 100; filename="BPS_SlantedGauss3D.gif", title="BPS Sampler", plot_type="3D")

# ForwardDiff: なぜかサンプルの精度が酷い
## grid 10 + vectorized 1:23 / 1.18, Mean 0.94
## grid 10 + non-vectorized 1:35, Mean 0.94
## Brent 5:06, Mean ? → サンプルを 1/10 にして 0:12, Mean 0.87
## grid 100 + vectorized 6:35, Mean 0.94

# Zygote
## grid 10 + vectorized 0:48, Mean 0.966...
