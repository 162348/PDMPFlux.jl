using StatsPlots  # histogram() など．
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
    println("number of error bound: ", sum(history.error_bound))

    return combined_plot
end

using LaTeXStrings

function plot_traj(history::PDMPHistory, T_max::Int; plot_type="2D", color="#78C2AD", background="#FFF", linewidth=2)
    T_max = min(T_max, length(history.t))  # avoids BoundsError
    traj = hcat(history.x...)
    if traj.size[1] == 1
        return Plots.plot(1:T_max, traj[1,1:T_max], xlabel = L"t", ylabel = L"x", title = "Trajectory (up to $T_max events)", label=false, color=color, background=background, linewidth=linewidth)
    elseif plot_type == "2D"
        return Plots.plot(traj[1,1:T_max], traj[2,1:T_max], xlabel = L"x_1", ylabel = L"x_2", title = "Trajectory (up to $T_max events)", label=false, color=color, background=background, linewidth=linewidth)
    else
        return Plots.plot(traj[1,1:T_max], traj[2,1:T_max], traj[3,1:T_max], xlabel = L"x_1", ylabel = L"x_2", zlabel = L"x_3", title = "Trajectory (up to $T_max events)", label=false, color=color, background=background, linewidth=linewidth)
    end
end

using ProgressBars

function anim_traj(history::PDMPHistory, T_max::Int; T_start::Int=1, plot_start::Int=1, filename::Union{String, Nothing}=nothing, plot_type="2D", color="#78C2AD", background="#FFF", coordinate_numbers=[1,2,3], dt::Float64=0.1, verbose::Bool=true, fps::Int=60, frame_upper_limit::Int=10000, linewidth=2, dynamic_range::Bool=false)
    T_max = min(T_max, length(history.t))  # avoids BoundsError
    trajectory = hcat(history.x...)

    if trajectory.size[1] == 1
        traj, event_time = traj_for_animation(trajectory, T_start, T_max; coordinate_numbers=coordinate_numbers[1], dt=dt)
        args = (
            xlims=(0, min(traj.size[2], frame_upper_limit)),
            ylims=(floor(minimum(traj[1,plot_start:end]),digits=1), ceil(maximum(traj[1,plot_start:end]),digits=1)),
            xlabel=L"t",
            ylabel=L"x",
            label=false,
            title="Trajectory (from $T_start to $T_max events)",
            color=color,
            background=background,
            linewidth=linewidth
            )
        args = dynamic_range ? (; args..., xlims=nothing, ylims=nothing) : args
        traj = traj[1,:]  # Vector に変換しないと @animate に掛かる時間が 10 倍くらいになる
        times = collect(Float64, 1:length(traj))  # なぜか Float64 にしないと @animate 内の push! エラー oundsError: attempt to access 2-element Vector{Plots.Series} at index [3] が出る

        upper_limit = min(length(traj), frame_upper_limit)
        if plot_start > upper_limit
            @warn "plot_start: $plot_start, upper_limit: $upper_limit"
            plot_start = upper_limit - 100
        end
        iter = verbose ? ProgressBar(plot_start:upper_limit, unit="B", unit_scale=true) : plot_start:upper_limit
        p = plot(times[1:plot_start], traj[1:plot_start]; args...)
        scatter!(p, [times[intersect(1:plot_start, event_time)]], traj[intersect(1:plot_start, event_time)], marker=:circle, markersize=3, markeralpha=0.6, color="#E95420", label=false)

        anim = @animate for i ∈ iter
            Base.push!(p, times[i], traj[i])
            if i ∈ event_time
                scatter!(p, [times[i]], traj[i:i], marker=:circle, markersize=3, markeralpha=0.6, color="#E95420", label=false)
            end
        end

    elseif plot_type == "2D"
        traj, event_time = traj_for_animation(trajectory, T_start, T_max; coordinate_numbers=coordinate_numbers[1:2], dt=dt)
        args = (
            xlims=(floor(minimum(traj[1,plot_start:end]),digits=1), ceil(maximum(traj[1,plot_start:end]),digits=1)),
            ylims=(floor(minimum(traj[2,plot_start:end]),digits=1), ceil(maximum(traj[2,plot_start:end]),digits=1)),
            xlabel=L"x_1",
            ylabel=L"x_2",
            label=false,
            title="Trajectory (from $T_start to $T_max events)",
            color=color,
            background=background,
            linewidth=linewidth
            )
        args = dynamic_range ? (; args..., xlims=nothing, ylims=nothing) : args
        traj_x = traj[1,:]
        traj_y = traj[2,:]

        upper_limit = min(length(traj_x), frame_upper_limit)
        if plot_start > upper_limit
            @warn "plot_start: $plot_start, upper_limit: $upper_limit"
            plot_start = upper_limit - 100
        end
        iter = verbose ? ProgressBar(plot_start:upper_limit, unit="B", unit_scale=true) : plot_start:upper_limit
        p = plot(traj_x[1:plot_start], traj_y[1:plot_start]; args...)
        scatter!(p, [traj_x[intersect(1:plot_start, event_time)]], traj_y[intersect(1:plot_start, event_time)], marker=:circle, markersize=3, markeralpha=0.6, color="#E95420", label=false)

        anim = @animate for i ∈ iter
            Base.push!(p, traj_x[i], traj_y[i])
            if i ∈ event_time
                scatter!(p, traj_x[i:i], traj_y[i:i], marker=:circle, markersize=3, markeralpha=0.6, color="#E95420", label=false)
            end
        end
    else
        traj, event_time = traj_for_animation(trajectory, T_start, T_max; coordinate_numbers=coordinate_numbers[1:3], dt=dt)
        args = (
            xlims=(floor(minimum(traj[1,plot_start:end]),digits=1), ceil(maximum(traj[1,plot_start:end]),digits=1)),
            ylims=(floor(minimum(traj[2,plot_start:end]),digits=1), ceil(maximum(traj[2,plot_start:end]),digits=1)),
            zlims=(floor(minimum(traj[3,plot_start:end]),digits=1), ceil(maximum(traj[3,plot_start:end]),digits=1)),
            xlabel=L"x_1",
            ylabel=L"x_2",
            zlabel=L"x_3",
            label=false,
            title="Trajectory (up to $T_max events)",
            color=color,
            background=background,
            linewidth=linewidth
            )
        args = dynamic_range ? (; args..., xlims=nothing, ylims=nothing, zlims=nothing) : args
        traj_x = traj[1,:]
        traj_y = traj[2,:]
        traj_z = traj[3,:]

        upper_limit = min(length(traj_x), frame_upper_limit)
        if plot_start > upper_limit
            @warn "plot_start: $plot_start, upper_limit: $upper_limit"
            plot_start = upper_limit - 100
        end
        iter = verbose ? ProgressBar(plot_start:upper_limit, unit="B", unit_scale=true) : plot_start:upper_limit
        p = plot(traj_x[1:plot_start], traj_y[1:plot_start], traj_z[1:plot_start]; args...)
        scatter!(p, [traj_x[intersect(1:plot_start, event_time)]], traj_y[intersect(1:plot_start, event_time)], traj_z[intersect(1:plot_start, event_time)], marker=:circle, markersize=3, markeralpha=0.6, color="#E95420", label=false)
        anim = @animate for i ∈ iter
            Base.push!(p, traj_x[i], traj_y[i], traj_z[i])
            if i ∈ event_time
                scatter!(p, traj_x[i:i], traj_y[i:i], traj_z[i:i], marker=:circle, markersize=3, markeralpha=0.6, color="#E95420", label=false)
            end
        end
    end

    filename = isnothing(filename) ? "PDMPFlux_Animation.gif" : filename
    filename = endswith(filename, ".gif") ? filename : filename * ".gif"
    gif(anim, filename, fps=fps)

    return anim
end

function traj_for_animation(trajectory::Matrix{Float64}, T_start::Int, T_max::Int; coordinate_numbers=[1,2,3], dt::Float64=0.01)
    traj = trajectory[coordinate_numbers, T_start:T_max]
    traj = isa(traj, Vector) ? reshape(traj, 1, :) : traj
    x = []
    event_time = []
    for (point, n) in zip(eachcol(traj),1:T_max-T_start+1)
        if n == 1  # initialize
            Base.push!(x, point)
            Base.push!(event_time, length(x))
        elseif n != T_max - T_start + 1
            displacement = traj[:,n+1] .- point
            distance = sqrt(sum(displacement.^2))
            step_number = round(Int, distance/dt)
            if step_number > 0
                step = displacement ./ step_number
                for i in 1:step_number
                    Base.push!(x, point + step .* i)
                end
                Base.push!(event_time, length(x))
            else
                Base.push!(x, traj[:,n+1])
                Base.push!(event_time, length(x))
            end
        end
    end
    return hcat(x...), event_time
end
