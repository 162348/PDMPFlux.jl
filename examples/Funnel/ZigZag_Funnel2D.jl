using PDMPFlux

using Random, Distributions, Plots, LaTeXStrings, ForwardDiff, LinearAlgebra

"""
    Funnel distribution for testing. Returns energy and sample functions.
    For reference, see Neal, R. M. (2003). Slice sampling. The Annals of Statistics, 31(3), 705–767.
"""
function funnel(d::Int=10, σ::Float64=3.0, clip_y::Int=11)

    function neg_energy(x::Vector)
        v = x[1]
        log_density_v = logpdf(Normal(0.0, 3.0), v)
        variance_other = exp(v)
        other_dim = d - 1
        cov_other = I * variance_other
        mean_other = zeros(other_dim)
        log_density_other = logpdf(MvNormal(mean_other, cov_other), x[2:end])
        return - log_density_v - log_density_other
    end

    function sample_data(n_samples::Int)
        # sample from Nd funnel distribution
        y = clamp.(σ * randn(n_samples, 1), -clip_y, clip_y)
        x = randn(n_samples, d - 1) .* exp.(-y / 2)
        return hcat(.- y, x)
    end

    return neg_energy, sample_data
end

function plot_funnel(d::Int=10, n_samples::Int=10000)
    _, sample_data = funnel(d)
    data = sample_data(n_samples)

    # 最初の2次元を抽出（yとx1）
    y = data[:, 1]
    x1 = data[:, 2]

    # 散布図をプロット
    scatter(y, x1, alpha=0.5, markersize=1, xlabel=L"y", ylabel=L"x_1", 
            title="Funnel Distribution (First Two Dimensions' Ground Truth)", grid=true, legend=false, color="#78C2AD")

    # xlim と ylim を追加
    xlims!(-8, 8)  # x軸の範囲を -8 から 8 に設定
    ylims!(-7, 7)  # y軸の範囲を -7 から 7 に設定
end
plot_funnel()

function run_ZigZag_on_funnel(N_sk::Int=100_000, N::Int=100_000; d::Int=10)
    U, _ = funnel(d)
    ∇U(x::Vector{Float64}) = ForwardDiff.gradient(U, x)
    xinit = ones(d)
    vinit = ones(d)
    seed = 2024
    grid_size = 0  # constant bounds
    sampler = ZigZag(d, ∇U, grid_size=grid_size)
    out = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed, verbose = true)
    samples = sample_from_skeleton(sampler, N, out)
    return out, samples
end
output, samples = run_ZigZag_on_funnel(d=2)

# jointplot(samples)
# plot_traj(output, 10000)
# plot_traj(output, 1000, plot_type="3D")

anim_traj(output, 1000; plot_start=100, filename="ZigZag_Funnel2D.gif")
# anim_traj(output, 1000; filename="ZigZag_Funnel2D.gif")

diagnostic(output)
