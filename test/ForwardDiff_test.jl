using PDMPFlux

using Random, Distributions, Plots, LaTeXStrings, ForwardDiff, LinearAlgebra

function U(x::Vector{Float64})
    v = x[1]
    log_density_v = logpdf(Normal(0.0, 3.0), v)
    variance_other = exp(v)
    other_dim = d - 1
    cov_other = I * variance_other
    mean_other = zeros(other_dim)
    log_density_other = logpdf(MvNormal(mean_other, cov_other), x[2:end])
    return - log_density_v - log_density_other
end

function V(x::Vector)
    y = x[1]
    log_density_y = - y^2 / 6

    variance_other = exp(y/2)

    log_density_other = - sum(x[2:end].^2) / (2 * variance_other)

    return - log_density_y - log_density_other
end

function run_ZigZag_on_funnel(N_sk::Int=100_000, N::Int=100_000, d::Int=10)
    ∇U(x::Vector{Float64}) = ForwardDiff.gradient(V, x)
    xinit = ones(d)
    vinit = ones(d)
    seed = 2024
    grid_size = 0  # constant bounds
    sampler = ZigZag(d, ∇U, grid_size=grid_size)
    out = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed, verbose = true)
    samples = sample_from_skeleton(sampler, N, out)
    return out, samples
end
output, samples = run_ZigZag_on_funnel()  # 05:29<00:00 → 00:20<00:00

jointplot(samples)
plot_traj(output, 10000)
plot_traj(output, 1000, plot_type="3D")

# anim_traj(output, 1000, plot_type="3D"; filename="ZigZag_Funnel3D_2.gif", dt=0.1)
# anim_traj(output, 1000; filename="ZigZag_Funnel2D.gif")

diagnostic(output)
