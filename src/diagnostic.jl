using StatsPlots  # histogram(), etc.
using Plots

function diagnostic(history::PDMPHistory; color="#78C2AD", background="#FFF", linewidth=2)
    p1 = StatsPlots.histogram(diff(history.t), bins=:auto, title="Time between events histogram", xlabel="Time", ylabel="Count", legend=false, color=color, linewidth=linewidth)
    p2 = StatsPlots.histogram(history.ar, bins=:auto, title="Acceptance rate histogram (Mean: $(round(mean(history.ar), digits=3)))", xlabel="Rate", ylabel="Relative Frequency", legend=false, color=color, normalize=true, linewidth=linewidth)
    Plots.vline!([mean(history.ar)], line=:dash, color="#E95420", legend=false, linewidth=linewidth/2)

    p3 = StatsPlots.histogram(history.hitting_horizon, bins=15, title="Hitting horizon histogram (Total: $(sum(history.hitting_horizon)))", xlabel="Horizon", ylabel="Log Frequency", yscale=:log10, legend=false, color=color, linewidth=linewidth)
    p4 = StatsPlots.histogram(history.rejected, bins=20, title="Rejection histogram (Total: $(sum(history.rejected)))", xlabel="Rejections", ylabel="Log Frequency", yscale=:log10, legend=false, color=color, linewidth=linewidth)

    # Combine plots into a grid
    combined_plot = Plots.plot(p1, p2, p3, p4, layout=(2, 2), background=background, size=(1600, 1200))

    display(combined_plot)

    # Print the number of error bounds
    println("number of error bound: ", sum(history.errored_bound))

    return combined_plot
end

@inline function _history_position_linear!(
    xout::AbstractVector{T},
    history::PDMPHistory{T},
    index::Int,
    τ::Float64,
) where {T}
    d = size(history.X, 1)
    @inbounds for j in 1:d
        vj = history.is_active[j, index] ? history.V[j, index] : zero(T)
        xout[j] = history.X[j, index] + vj * T(τ)
    end
    return xout
end

function RV_diagnostic(history::PDMPHistory, U::Function; B::Int64 = 0)
    t = history.t
    N = length(t)
    N == 0 && return 0.0

    T = t[end]
    if !(isfinite(T)) || T < 0.0
        throw(ArgumentError("history.t[end] must be finite and non-negative. Current value: $T"))
    end

    if B == 0
        B = max(1, floor(Int, sqrt(size(history.X, 2))))
    elseif B < 0
        throw(ArgumentError("B must be non-negative. Current value: $B"))
    end

    T == 0.0 && return 0.0

    boundaries = range(0.0, T; length=B+1)
    x_left = copy(@view history.X[:, 1])
    x_right = similar(x_left)
    RV = 0.0
    i = 1

    @inbounds for b in 2:length(boundaries)
        tb = boundaries[b]
        while i < N && t[i + 1] <= tb
            i += 1
        end

        τ = tb - t[i]
        _history_position_linear!(x_right, history, i, τ)
        increment = U(x_right) - U(x_left)
        RV += increment^2
        copyto!(x_left, x_right)
    end

    return RV / T
end