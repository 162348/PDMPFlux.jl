using PDMPFlux

# using Test
using Zygote, Random, Plots

N_sk = 10_000 # number of skeleton points
N = 10_000 # number of samples

function runtest(N_sk::Int, N::Int, dim::Int=3)
    function U_banana(x::Vector)
        mean_x2 = (x[1]^2 - 1)
        return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
    end

    ∇U(x::Vector) = gradient(U_banana, x)[1]
    seed = 8
    key = MersenneTwister(seed)
    xinit = ones(dim)
    vinit = ones(dim)
    grid_size = 0  # constant bounds

    sampler = ZigZag(dim, ∇U, grid_size=grid_size)
    out = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed, verbose = true)
    samples = sample_from_skeleton(sampler, N, out)

    return out, samples
end

out, samples = runtest(N_sk, N)
anim_traj(out, 150; N_start=100, plot_start=100, filename="ZigZag_Banana3D.gif", plot_type="3D", background="#F0F1EB")