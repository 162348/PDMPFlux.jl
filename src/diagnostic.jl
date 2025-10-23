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
    println("number of error bound: ", sum(history.errored_bound))

    return combined_plot
end

using LaTeXStrings

function plot_traj(history::PDMPHistory, N_max::Int; plot_type="2D", color="#78C2AD", background="#FFF",
    title::Union{String, LaTeXString}="Trajectory (up to $N_max events)", linewidth=2,
    n_start::Int=1, filename::Union{String, Nothing}=nothing, xv_plot = false, kwargs...)

    N_max = min(N_max, length(history.t))  # to avoid BoundsError
    traj = hcat(history.x...)
    time_stamps = history.t[n_start:N_max]

    if !xv_plot
        if traj.size[1] == 1
            p = Plots.plot(time_stamps, traj[1,n_start:N_max], xlabel = L"t", ylabel = L"x", title = title, label=false, color=color, background=background, linewidth=linewidth; kwargs...)
        elseif plot_type == "2D"
            p = Plots.plot(traj[1,n_start:N_max], traj[2,n_start:N_max], xlabel = L"x_1", ylabel = L"x_2", title = title, label=false, color=color, background=background, linewidth=linewidth; kwargs...)
        else
            p = Plots.plot(traj[1,n_start:N_max], traj[2,n_start:N_max], traj[3,n_start:N_max], xlabel = L"x_1", ylabel = L"x_2", zlabel = L"x_3", title = title, label=false, color=color, background=background, linewidth=linewidth; kwargs...)
        end
    else
        traj_v = hcat(history.v...)
        p = Plots.plot(traj[1,n_start:N_max], traj_v[1,n_start:N_max], xlabel = L"x", ylabel = L"v", title = title, label=false, color=color, background=background, linewidth=linewidth; kwargs...)
    end

    if !isnothing(filename)
        filename = isnothing(filename) ? "PDMPFlux_Trajectory.svg" : filename
        filename = contains(filename, ".") ? filename : filename * ".svg"
        savefig(p, filename)
    end

    return p
end

using ProgressBars

function anim_traj(history::PDMPHistory, N_max::Int; N_start::Int=1, plot_start::Int=1,
    filename::Union{String, Nothing}=nothing, plot_type="2D", color="#78C2AD", background="#FFF",
    coordinate_numbers=[1,2,3], dt::Float64=0.1, verbose::Bool=true,
    fps::Int=60, frame_upper_limit::Int=10000, linewidth=2, dynamic_range::Bool=false,
    title::Union{String, LaTeXString}="Trajectory (from $N_start to $N_max events)",
    nonlinear_flow::Union{Function, Nothing}=nothing)

    N_max = min(N_max, length(history.t), frame_upper_limit)  # to avoid BoundsError
    time_stamps = history.t[N_start:N_max]

    if length(history.x[1]) == 1 || plot_type == "1D"  # if dim = 1, horizontal axis is time
        traj, event_indeces, times = traj_for_animation(history, time_stamps, N_start, N_max; coordinate_numbers=coordinate_numbers[1], dt=dt, nonlinear_flow=nonlinear_flow)
        args = (
            xlims=(0, times[min(end, frame_upper_limit)]),
            ylims=(floor(minimum(traj[1,plot_start:end]),digits=1), ceil(maximum(traj[1,plot_start:end]),digits=1)),
            xlabel=L"t",
            ylabel=L"x",
            label=false,
            title=title,
            color=color,
            background=background,
            linewidth=linewidth
            )
        args = dynamic_range ? (; args..., xlims=nothing, ylims=nothing) : args
        traj = traj[1,:]  # Vector に変換しないと @animate に掛かる時間が 10 倍くらいになる
        # times = collect(Float64, 1:length(traj))  # なぜか Float64 にしないと @animate 内の push! エラー oundsError: attempt to access 2-element Vector{Plots.Series} at index [3] が出る

        # maximum number of events to be plotted
        upper_limit = min(length(traj), frame_upper_limit)
        plot_start_frame = event_indeces[plot_start]
        if plot_start_frame > upper_limit
            @warn "plot_start_frame: $plot_start_frame > upper_limit: $upper_limit"
            plot_start_frame = upper_limit - 100
        end
        # initialize plot
        p = plot(times[1:plot_start_frame], traj[1:plot_start_frame]; args...)
        scatter!(p, [times[intersect(1:plot_start_frame, event_indeces)]], traj[intersect(1:plot_start_frame, event_indeces)],
        marker=:circle, markersize=3, markeralpha=0.6, color="#E95420", label=false)
        iter = verbose ? ProgressBar(plot_start_frame:upper_limit, unit="B", unit_scale=true) : plot_start_frame:upper_limit

        anim = @animate for i ∈ iter
            Base.push!(p, times[i], traj[i])
            if i ∈ event_indeces
                scatter!(p, [times[i]], traj[i:i], marker=:circle, markersize=3, markeralpha=0.6, color="#E95420", label=false)
            end
        end

    elseif plot_type == "2D"  # if dim > 1 & 2D plot is 
        traj, event_indeces, _times = traj_for_animation(history, time_stamps, N_start, N_max; coordinate_numbers=coordinate_numbers[1:2], dt=dt, nonlinear_flow=nonlinear_flow)
        args = (
            xlims=(floor(minimum(traj[1,plot_start:end]),digits=1), ceil(maximum(traj[1,plot_start:end]),digits=1)),
            ylims=(floor(minimum(traj[2,plot_start:end]),digits=1), ceil(maximum(traj[2,plot_start:end]),digits=1)),
            xlabel=L"x_1",
            ylabel=L"x_2",
            label=false,
            title=title,
            color=color,
            background=background,
            linewidth=linewidth
            )
        args = dynamic_range ? (; args..., xlims=nothing, ylims=nothing) : args
        traj_x = traj[1,:]
        traj_y = traj[2,:]

        upper_limit = min(length(traj_x), frame_upper_limit)
        plot_start_frame = event_indeces[plot_start]
        if plot_start_frame > upper_limit
            @warn "plot_start_frame: $plot_start_frame > upper_limit: $upper_limit"
            plot_start_frame = upper_limit - 100
        end
        p = plot(traj_x[1:plot_start_frame], traj_y[1:plot_start_frame]; args...)
        scatter!(p, [traj_x[intersect(1:plot_start_frame, event_indeces)]], traj_y[intersect(1:plot_start_frame, event_indeces)], marker=:circle, markersize=3, markeralpha=0.6, color="#E95420", label=false)
        iter = verbose ? ProgressBar(plot_start_frame:upper_limit, unit="B", unit_scale=true) : plot_start_frame:upper_limit

        anim = @animate for i ∈ iter
            Base.push!(p, traj_x[i], traj_y[i])
            if i ∈ event_indeces
                scatter!(p, traj_x[i:i], traj_y[i:i], marker=:circle, markersize=3, markeralpha=0.6, color="#E95420", label=false)
            end
        end
    else  # if dim > 1 & 3D plot is requested
        traj, event_indeces, _times = traj_for_animation(history, time_stamps, N_start, N_max; coordinate_numbers=coordinate_numbers[1:3], dt=dt, nonlinear_flow=nonlinear_flow)
        args = (
            xlims=(floor(minimum(traj[1,plot_start:end]),digits=1), ceil(maximum(traj[1,plot_start:end]),digits=1)),
            ylims=(floor(minimum(traj[2,plot_start:end]),digits=1), ceil(maximum(traj[2,plot_start:end]),digits=1)),
            zlims=(floor(minimum(traj[3,plot_start:end]),digits=1), ceil(maximum(traj[3,plot_start:end]),digits=1)),
            xlabel=L"x_1",
            ylabel=L"x_2",
            zlabel=L"x_3",
            label=false,
            title=title,
            color=color,
            background=background,
            linewidth=linewidth
            )
        args = dynamic_range ? (; args..., xlims=nothing, ylims=nothing, zlims=nothing) : args
        traj_x = traj[1,:]
        traj_y = traj[2,:]
        traj_z = traj[3,:]

        upper_limit = min(length(traj_x), frame_upper_limit)
        plot_start_frame = event_indeces[plot_start]
        if plot_start_frame > upper_limit
            @warn "plot_start_frame: $plot_start_frame > upper_limit: $upper_limit"
            plot_start_frame = upper_limit - 100
        end
        # initialize plot
        p = plot(traj_x[1:plot_start_frame], traj_y[1:plot_start_frame], traj_z[1:plot_start_frame]; args...)
        scatter!(p, [traj_x[intersect(1:plot_start_frame, event_indeces)]], traj_y[intersect(1:plot_start_frame, event_indeces)], traj_z[intersect(1:plot_start_frame, event_indeces)], marker=:circle, markersize=3, markeralpha=0.6, color="#E95420", label=false)
        iter = verbose ? ProgressBar(plot_start_frame:upper_limit, unit="B", unit_scale=true) : plot_start_frame:upper_limit

        anim = @animate for i ∈ iter
            Base.push!(p, traj_x[i], traj_y[i], traj_z[i])
            if i ∈ event_indeces
                scatter!(p, traj_x[i:i], traj_y[i:i], traj_z[i:i], marker=:circle, markersize=3, markeralpha=0.6, color="#E95420", label=false)
            end
        end
    end

    if !isnothing(filename)
        filename = isnothing(filename) ? "PDMPFlux_Animation.gif" : filename
        filename = endswith(filename, ".gif") ? filename : filename * ".gif"
        gif(anim, filename, fps=fps)
    end

    return anim
end

"""
    traj_for_animation(): アニメーション用の軌道を抽出する．

    Parameters:
    - trajectory (Matrix{Float64}): The trajectory to be animated.
    - N_start (Int): The starting index of the trajectory.
    - N_max (Int): The ending index of the trajectory.
    - coordinate_numbers (Array{Int, 1}): The indices of the coordinates to be plotted.
    - dt (Float64): The time step for the animation.

    Returns:
    - traj: The snapshots of the trajectory to be animated.
    - event_indeces: The indices of traj, where the events occur.
"""
function traj_for_animation(history::PDMPHistory, time_stamps::Vector{Float64}, N_start::Int, N_max::Int;
    coordinate_numbers=[1,2,3], dt::Float64=0.1, nonlinear_flow::Union{Function, Nothing}=nothing)

    trajectory = hcat(history.x...)
    traj = trajectory[coordinate_numbers, N_start:N_max]
    traj = isa(traj, Vector) ? reshape(traj, 1, :) : traj
    v_history = hcat(history.v...)[coordinate_numbers, N_start:N_max]  # not used if nonlinear_flow = nothing
    x, event_indeces, t = [], [], []

    for (xₙ, tₙ, n) in zip(eachcol(traj), time_stamps, 1:N_max-N_start+1)
        if n == 1  # initialize
            Base.push!(x, xₙ)
            Base.push!(t, tₙ)
            Base.push!(event_indeces, length(x))  # event_indeces[1] = 1
        elseif isnothing(nonlinear_flow)
            time_passed = tₙ - time_stamps[n-1]
            step_number = round(Int, time_passed/dt)
            if step_number > 0
                one_step = (xₙ .- traj[:,n-1]) ./ step_number
                one_step_time = time_passed ./ step_number
                for i in 1:step_number
                    Base.push!(x, traj[:,n-1] + one_step .* i)
                    Base.push!(t, time_stamps[n-1] + one_step_time * i)
                end
                Base.push!(event_indeces, length(x))
            else  # step_number == 0
                Base.push!(x, traj[:,n])
                Base.push!(t, time_stamps[n])
                Base.push!(event_indeces, length(x))
            end
        else
            time_passed = tₙ - time_stamps[n-1]
            step_number = round(Int, time_passed/dt)
            if step_number > 0
                one_step = (xₙ .- traj[:,n-1]) ./ step_number
                one_step_time = time_passed ./ step_number
                for i in 1:step_number
                    Base.push!(x, nonlinear_flow(traj[:,n-1], v_history[:,n-1], one_step_time * i)[1])
                    Base.push!(t, time_stamps[n-1] + one_step_time * i)
                end
                Base.push!(event_indeces, length(x))
            else  # step_number == 0
                Base.push!(x, traj[:,n])
                Base.push!(t, time_stamps[n])
                Base.push!(event_indeces, length(x))
            end
        end
    end
    return hcat(x...), event_indeces, t
end

