using PDMPFlux

function U_Gauss(x::Vector)
    return sum(x.^2) / 2
end

dim = 1000
sampler = ForwardECMCAD(dim, U_Gauss, grid_size=10)

N_sk, N, xinit, vinit = 100_000, 100_000, zeros(dim), ones(dim)
# samples = sample(sampler, N_sk, N, xinit, vinit, seed=2024)

output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2024)
samples = sample_from_skeleton(sampler, N, output)

# diagnostic(output)
# jointplot(samples)

### 5/20/2025
# d=1000; 44, 

# ForwardDiff: なぜかサンプルの精度が酷い
## grid 10 + vectorized 1:23 / 1.18, Mean 0.94
## grid 10 + non-vectorized 1:35, Mean 0.94
## Brent 5:06, Mean ? → サンプルを 1/10 にして 0:12, Mean 0.87
## grid 100 + vectorized 6:35, Mean 0.94

# Zygote
## grid 10 + vectorized 0:48, Mean 0.966...
