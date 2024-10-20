using PDMPFlux

# using Test
using ForwardDiff, ReverseDiff, Random, Plots

N_sk = 1_000_000 # number of skeleton points
N = 1_000_000 # number of samples

function runtest(N_sk::Int, N::Int)
    function U_banana(x)
        mean_x2 = (x[1]^2 - 1)
        return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
    end

    dim = 50
    grad_U(x::Vector) = ForwardDiff.gradient(U_banana, x)
    seed = 8
    key = MersenneTwister(seed)
    xinit = ones(dim)
    vinit = ones(dim)
    grid_size = 0  # constant bounds

    sampler = ZigZag(dim, grad_U, grid_size=grid_size)
    out = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed, verbose = true)
    samples = sample_from_skeleton(sampler, N, out)

    return out, samples
end

function ground_truth()
    function density_banana(x::Vector{Float64})
        mean_x2 = (x[1]^2 - 1)
        return exp((- x[1]^2 - (x[2] - mean_x2)^2) / 2)
    end
    x_range = -4.0:0.01:4.0
    y_range = -5.0:0.025:15.0
    z = [density_banana([x, y])^(1/2) for x in x_range, y in y_range]
    contourf(x_range, y_range, z, xlabel="x2", ylabel="x1", title="Banana Density Contour", color=:summer)
end

# ground_truth()
out, samples = runtest(N_sk, N)
jointplot(samples, coordinate_numbers=[2,1])
# anim_traj(out, 10000; filename="ZigZag_Banana2D.gif", dt=0.1)
# anim_traj(out, 10000; filename="ZigZag_Banana3D.gif", dt=0.1, plot_type="3D")
# plot_traj(out, 10000)
# diagnostic(out)
# jointplot(samples, coordinate_numbers=[2,3])




# sampler_ad = ZigZagAD(dim, U_banana, grid_size=grid_size)
# out_ad = sample_skeleton(sampler_ad, N_sk, xinit, vinit, seed=seed, verbose = true)
# samples_ad = sample_from_skeleton(sampler_ad, N, out_ad)
# samples_ad = sample(sampler_ad, N_sk, N, xinit, vinit, seed=seed, verbose = true)

# using StatsPlots

# marginalhist(samples_ad[1,:],samples_ad[2,:])
# plot(out_ad)