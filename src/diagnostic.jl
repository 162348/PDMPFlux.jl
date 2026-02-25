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

using LaTeXStrings

function plot_traj(history::PDMPHistory, N_max::Int; plot_type="2D", color="#78C2AD", background="#FFF",
    title::Union{String, LaTeXString}="Trajectory (up to $N_max events)", linewidth=2,
    n_start::Int=1, filename::Union{String, Nothing}=nothing, xv_plot = false, contour_plot = false, kwargs...)

    N_max = min(N_max, length(history.t))  # to avoid BoundsError
    traj = history.X
    time_stamps = history.t[n_start:N_max]

    if !xv_plot
        if size(traj, 1) == 1
            p = Plots.plot(time_stamps, traj[1,n_start:N_max], xlabel = L"t", ylabel = L"x", title = title, label=false, color=color, background=background, linewidth=linewidth; kwargs...)
        elseif plot_type == "2D"
            p = Plots.plot(traj[1,n_start:N_max], traj[2,n_start:N_max], xlabel = L"x_1", ylabel = L"x_2", title = title, label=false, color=color, background=background, linewidth=linewidth; kwargs...)
        else
            p = Plots.plot(traj[1,n_start:N_max], traj[2,n_start:N_max], traj[3,n_start:N_max], xlabel = L"x_1", ylabel = L"x_2", zlabel = L"x_3", title = title, label=false, color=color, background=background, linewidth=linewidth; kwargs...)
        end
    else
        traj_v = history.V
        p = Plots.plot(traj[1,n_start:N_max], traj_v[1,n_start:N_max], xlabel = L"x", ylabel = L"v", title = title, label=false, color=color, background=background, linewidth=linewidth; kwargs...)
    end

    if !isnothing(filename)
        filename = isnothing(filename) ? "PDMPFlux_Trajectory.svg" : filename
        filename = contains(filename, ".") ? filename : filename * ".svg"
        savefig(p, filename)
    end

    return p
end

function plot_traj!(p, history::PDMPHistory, N_max::Int; plot_type="2D", color="#78C2AD", background=:transparent,
    title::Union{String, LaTeXString}="Trajectory (up to $N_max events)", linewidth=2,
    n_start::Int=1, filename::Union{String, Nothing}=nothing, xv_plot = false,
    show_title::Bool=true, show_grid::Bool=false, show_axis::Bool=false,
    kwargs...)

    N_max = min(N_max, length(history.t))  # to avoid BoundsError
    traj = history.X
    time_stamps = history.t[n_start:N_max]

    if !xv_plot
        if size(traj, 1) == 1
            p = Plots.plot!(p, time_stamps, traj[1,n_start:N_max], xlabel = show_axis ? L"t" : "", ylabel = show_axis ? L"x" : "", grid=show_grid, title = show_title ? title : "", label=false, color=color, background=background, linewidth=linewidth, xticks=show_axis, yticks=show_axis, legend=false; kwargs...)
        elseif plot_type == "2D"
            p = Plots.plot!(p, traj[1,n_start:N_max], traj[2,n_start:N_max], xlabel = show_axis ? L"x_1" : "", ylabel = show_axis ? L"x_2" : "", grid=show_grid, title = show_title ? title : "", label=false, color=color, background=background, linewidth=linewidth, xticks=show_axis, yticks=show_axis, legend=false; kwargs...)
        else
            p = Plots.plot!(p, traj[1,n_start:N_max], traj[2,n_start:N_max], traj[3,n_start:N_max], xlabel = show_axis ? L"x_1" : "", ylabel = show_axis ? L"x_2" : "", zlabel = show_axis ? L"x_3" : "", grid=show_grid, title = show_title ? title : "", label=false, color=color, background=background, linewidth=linewidth, xticks=show_axis, yticks=show_axis, zticks=show_axis, legend=false; kwargs...)
        end
    else
        traj_v = history.V
        p = Plots.plot!(p, traj[1,n_start:N_max], traj_v[1,n_start:N_max], grid=show_grid, title = show_title ? title : "", label=false, color=color, background=background, linewidth=linewidth, xticks=show_axis, yticks=show_axis, legend=false; kwargs...)
    end

    if !isnothing(filename)
        filename = occursin(".", filename) ? filename : filename * ".svg"
        savefig(p, filename)
    end

    return p
end

"""
Create a contour plot of the potential function `U` (2D).

# Arguments
- `U`: potential function taking a 2D vector
- `x_range`: range for the x-axis (default: `range(-5, 5, length=100)`)
- `y_range`: range for the y-axis (default: `range(-5, 5, length=100)`)
- `levels`: number of contour levels (default: 20)
- `color`: color palette (default: `:viridis`)
- `fill`: whether to fill contours (default: true)
- `show_grid`: show grid (default: false)
- `show_axis`: show axis ticks/labels (default: false)
- `show_title`: show title (default: false)
- `xlabel`: x-axis label (default: `L"x_1"`)
- `ylabel`: y-axis label (default: `L"x_2"`)
- `background`: background (default: `:transparent`)
- `linewidth`: contour line width (default: 1)
- `filename`: output filename to save (default: `nothing`)
"""
function plot_U_contour(U;
    x_range = range(-5, 5, length=100),
    y_range = range(-5, 5, length=100),
    levels = 20,
    color = :viridis,
    fill = true,
    show_grid = false,
    show_axis = false,
    show_title = false,
    xlabel = L"x_1",
    ylabel = L"x_2",
    background = :transparent,
    linewidth = 1,
    filename = nothing)
    
    # Compute U values on the grid
    Z = [U([x, y]) for x in x_range, y in y_range]
    
    # Create the plot
    p = contour(x_range, y_range, Z,
        levels = levels,
        color = color,
        fill = fill,
        linewidth = linewidth,
        xlabel = show_axis ? xlabel : "",
        ylabel = show_axis ? ylabel : "",
        background = background,
        grid = show_grid,
        title = show_title ? L"Contour plot of $U(x)$" : "",
        xticks = show_axis,
        yticks = show_axis,
        legend = false)

    if !isnothing(filename)
        filename = occursin(".", filename) ? filename : filename * ".svg"
        savefig(p, filename)
    end

    return p
end

using ProgressBars

function anim_traj(history::PDMPHistory, N_max::Int; N_start::Int=1, plot_start::Int=1,
    filename::Union{String, Nothing}=nothing, plot_type="2D", color="#78C2AD", background="#FFF", scatter_color="#E95420",
    scatter_alpha=0.6, markerstrokewidth=0.0, scatter_markersize=3,
    skip_frames::Int=1,
    coordinate_numbers=[1,2,3], dt::Float64=0.1, verbose::Bool=true, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"x_3",
    fps::Int=60, frame_upper_limit::Int=10000, linewidth=2, dynamic_range::Bool=false,
    xlims=nothing, ylims=nothing,
    title::Union{String, LaTeXString}="Trajectory (from $N_start to $N_max events)",
    nonlinear_flow::Union{Function, Nothing}=nothing, kwargs...)

    N_max = min(N_max, length(history.t), frame_upper_limit)  # to avoid BoundsError
    time_stamps = history.t[N_start:N_max]

    if size(history.X, 1) == 1 || plot_type == "1D"  # if dim = 1, horizontal axis is time
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
            linewidth=linewidth,
            kwargs...
            )
        args = dynamic_range ? (; args..., xlims=nothing, ylims=nothing) : args
        traj = traj[1,:]  # Convert to Vector; otherwise @animate can be ~10x slower.
        # times = collect(Float64, 1:length(traj))  # NOTE: using Float64 avoided a push! BoundsError inside @animate in the past.

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
        marker=:circle, markersize=3, markeralpha=scatter_alpha, color=scatter_color, label=false)
        iter = verbose ? ProgressBar(plot_start_frame:upper_limit, unit="B", unit_scale=true) : plot_start_frame:upper_limit

        anim = @animate for i ∈ iter
            Base.push!(p, times[i], traj[i])
            if i ∈ event_indeces
                scatter!(p, [times[i]], traj[i:i], marker=:circle, markersize=3, markeralpha=scatter_alpha, color=scatter_color, label=false)
            end
        end every skip_frames

    elseif plot_type == "2D"  # if dim > 1 & 2D plot is 
        traj, event_indeces, _times = traj_for_animation(history, time_stamps, N_start, N_max; coordinate_numbers=coordinate_numbers[1:2], dt=dt, nonlinear_flow=nonlinear_flow)
        if isnothing(xlims)
            xlims = (floor(minimum(traj[1,plot_start:end]),digits=1), ceil(maximum(traj[1,plot_start:end]),digits=1))
        end
        if isnothing(ylims)
            ylims = (floor(minimum(traj[2,plot_start:end]),digits=1), ceil(maximum(traj[2,plot_start:end]),digits=1))
        end
        args = (
            xlims=xlims,
            ylims=ylims,
            xlabel=xlabel,
            ylabel=ylabel,
            label=false,
            title=title,
            color=color,
            background=background,
            linewidth=linewidth,
            kwargs...
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
        iter = verbose ? ProgressBar(plot_start_frame:upper_limit, unit="B", unit_scale=true) : plot_start_frame:upper_limit

        anim = @animate for i ∈ iter
            p = plot(traj_x[1:i], traj_y[1:i]; args...)
            scatter!(p, [traj_x[intersect(1:i, event_indeces)]], traj_y[intersect(1:i, event_indeces)], marker=:circle, markersize=scatter_markersize,
                markeralpha=scatter_alpha, color=scatter_color, markerstrokewidth=markerstrokewidth, label=false
                )
            scatter!(p, traj_x[i:i], traj_y[i:i];
                marker=:circle, color=color, markersize=6, markerstrokewidth=0, label=false
                )
        end every skip_frames
    else  # if dim > 1 & 3D plot is requested
        traj, event_indeces, _times = traj_for_animation(history, time_stamps, N_start, N_max; coordinate_numbers=coordinate_numbers[1:3], dt=dt, nonlinear_flow=nonlinear_flow)
        args = (
            xlims=(floor(minimum(traj[1,plot_start:end]),digits=1), ceil(maximum(traj[1,plot_start:end]),digits=1)),
            ylims=(floor(minimum(traj[2,plot_start:end]),digits=1), ceil(maximum(traj[2,plot_start:end]),digits=1)),
            zlims=(floor(minimum(traj[3,plot_start:end]),digits=1), ceil(maximum(traj[3,plot_start:end]),digits=1)),
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            label=false,
            title=title,
            color=color,
            background=background,
            linewidth=linewidth,
            kwargs...
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
        scatter!(p, [traj_x[intersect(1:plot_start_frame, event_indeces)]], traj_y[intersect(1:plot_start_frame, event_indeces)], traj_z[intersect(1:plot_start_frame, event_indeces)], marker=:circle, markersize=3, markeralpha=scatter_alpha, color=scatter_color, label=false)
        iter = verbose ? ProgressBar(plot_start_frame:upper_limit, unit="B", unit_scale=true) : plot_start_frame:upper_limit

        anim = @animate for i ∈ iter
            Base.push!(p, traj_x[i], traj_y[i], traj_z[i])
            if i ∈ event_indeces
                scatter!(p, traj_x[i:i], traj_y[i:i], traj_z[i:i], marker=:circle, markersize=3, markeralpha=scatter_alpha, color=scatter_color, label=false)
            end
        end every skip_frames
    end

    if !isnothing(filename)
        filename = isnothing(filename) ? "PDMPFlux_Animation.gif" : filename
        filename = endswith(filename, ".gif") ? filename : filename * ".gif"
        gif(anim, filename, fps=fps)
    end

    return anim
end

"""
    A buffer function under development
    supports faded color animation
"""
function anim_traj_(history::PDMPHistory, N_max::Int; N_start::Int=1, plot_start::Int=1,
    filename::Union{String, Nothing}=nothing, plot_type="2D", color="#78C2AD", background="#FFF", scatter_color="#E95420", scatter_alpha=0.6, scatter_size=3,
    coordinate_numbers=[1,2,3], dt::Float64=0.1, verbose::Bool=true, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"x_3",
    fps::Int=60, frame_upper_limit::Int=10000, linewidth=2, dynamic_range::Bool=false,
    xlims=nothing, ylims=nothing, skip_frames::Int=1,
    title::Union{String, LaTeXString}="Trajectory (from $N_start to $N_max events)",
    nonlinear_flow::Union{Function, Nothing}=nothing,
    fade::Bool=true,
    tail_length::Int=100,
    highlight_length::Int=15,
    history_stride::Int=1,
    history_color=:black,
    history_alpha::Float64=1.0,
    tail_color_start=history_color,
    tail_color_end=color,
    highlight_color=color,
    kwargs...)

    N_max = min(N_max, length(history.t), frame_upper_limit)  # to avoid BoundsError
    time_stamps = history.t[N_start:N_max]

    if size(history.X, 1) == 1 || plot_type == "1D"  # if dim = 1, horizontal axis is time
        traj, event_indeces, times = traj_for_animation(history, time_stamps, N_start, N_max; coordinate_numbers=coordinate_numbers[1], dt=dt, nonlinear_flow=nonlinear_flow)
        args = (
            xlims=(0, times[min(end, frame_upper_limit)]),
            ylims=(floor(minimum(traj[1,plot_start:end]),digits=1), ceil(maximum(traj[1,plot_start:end]),digits=1)),
            xlabel=L"t",
            ylabel=L"x",
            label=false,
            title=title,
            background=background,
            kwargs...
            )
        args = dynamic_range ? (; args..., xlims=nothing, ylims=nothing) : args
        traj = traj[1,:]  # Convert to Vector; otherwise @animate can be ~10x slower.
        # times = collect(Float64, 1:length(traj))  # NOTE: using Float64 avoided a push! BoundsError inside @animate in the past.

        # maximum number of events to be plotted
        upper_limit = min(length(traj), frame_upper_limit)
        plot_start_frame = event_indeces[plot_start]
        if plot_start_frame > upper_limit
            @warn "plot_start_frame: $plot_start_frame > upper_limit: $upper_limit"
            plot_start_frame = upper_limit - 100
        end
        iter = verbose ? ProgressBar(plot_start_frame:upper_limit, unit="B", unit_scale=true) : plot_start_frame:upper_limit

        tail_cgrad = cgrad([tail_color_start, tail_color_end])

        if fade
            anim = @animate for i ∈ iter
                i_tail_start = max(plot_start_frame, i - tail_length + 1)
                i_highlight_start = max(i_tail_start, i - highlight_length + 1)

                hist_end = max(i_tail_start - 1, plot_start_frame)
                hist_idx = 1:history_stride:hist_end

                p = plot(times[hist_idx], traj[hist_idx];
                    args...,
                    color=history_color,
                    alpha=history_alpha,
                    linewidth=linewidth,
                    colorbar = false,
                )

                # tail (gradient: black -> red)
                plot!(p, times[i_tail_start:i], traj[i_tail_start:i];
                    seriescolor=tail_cgrad,
                    line_z=1:(i - i_tail_start + 1),
                    linewidth=linewidth,
                    label=false
                )

                # highlight newest segment in red
                plot!(p, times[i_highlight_start:i], traj[i_highlight_start:i];
                    color=highlight_color,
                    linewidth=linewidth + 0.5,
                    label=false
                )

                # event markers up to current frame
                last_ev = searchsortedlast(event_indeces, i)
                if last_ev > 0
                    ev = @view event_indeces[1:last_ev]
                    scatter!(p, times[ev], traj[ev];
                        marker=:circle,
                        markersize=scatter_size,
                        markeralpha=scatter_alpha,
                        color=scatter_color,
                        label=false
                    )
                end

                p
            end every skip_frames
        else
            # legacy: single-color incremental drawing
            p = plot(times[1:plot_start_frame], traj[1:plot_start_frame]; args..., color=color, linewidth=linewidth)
            scatter!(p, times[intersect(1:plot_start_frame, event_indeces)], traj[intersect(1:plot_start_frame, event_indeces)];
                marker=:circle, markersize=scatter_size, markeralpha=scatter_alpha, color=scatter_color, label=false
            )
            anim = @animate for i ∈ iter
                Base.push!(p, times[i], traj[i])
                if i ∈ event_indeces
                    scatter!(p, [times[i]], traj[i:i]; marker=:circle, markersize=3, markeralpha=scatter_alpha, color=scatter_color, label=false)
                end
                p
            end every skip_frames
        end

    elseif plot_type == "2D"  # if dim > 1 & 2D plot is 
        traj, event_indeces, _times = traj_for_animation(history, time_stamps, N_start, N_max; coordinate_numbers=coordinate_numbers[1:2], dt=dt, nonlinear_flow=nonlinear_flow)
        if isnothing(xlims)
            xlims = (floor(minimum(traj[1,plot_start:end]),digits=1), ceil(maximum(traj[1,plot_start:end]),digits=1))
        end
        if isnothing(ylims)
            ylims = (floor(minimum(traj[2,plot_start:end]),digits=1), ceil(maximum(traj[2,plot_start:end]),digits=1))
        end
        args = (
            xlims=xlims,
            ylims=ylims,
            xlabel=xlabel,
            ylabel=ylabel,
            label=false,
            title=title,
            background=background,
            kwargs...
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
        iter = verbose ? ProgressBar(plot_start_frame:upper_limit, unit="B", unit_scale=true) : plot_start_frame:upper_limit

        tail_cgrad = cgrad([tail_color_start, tail_color_end])

        if fade
            anim = @animate for i ∈ iter
                i_tail_start = max(plot_start_frame, i - tail_length + 1)
                i_highlight_start = max(i_tail_start, i - highlight_length + 1)

                hist_end = max(i_tail_start - 1, plot_start_frame)
                hist_idx = 1:history_stride:hist_end+1

                p = plot(traj_x[hist_idx], traj_y[hist_idx];
                    args...,
                    color=history_color,
                    alpha=history_alpha,
                    linewidth=linewidth,
                    colorbar = false,
                )

                plot!(p, traj_x[i_tail_start:i], traj_y[i_tail_start:i];
                    seriescolor=tail_cgrad,
                    line_z=1:(i - i_tail_start + 1),
                    linewidth=linewidth,
                    label=false
                )

                plot!(p, traj_x[i_highlight_start:i], traj_y[i_highlight_start:i];
                    color=highlight_color,
                    linewidth=linewidth + 0.5,
                    label=false
                )

                last_ev = searchsortedlast(event_indeces, i)
                if last_ev > 0
                    ev = @view event_indeces[1:last_ev]
                    scatter!(p, traj_x[ev], traj_y[ev];
                        marker=:circle,
                        markersize=scatter_size,
                        markeralpha=scatter_alpha,
                        color=scatter_color,
                        label=false
                    )
                end

                p
            end every skip_frames
        else
            p = plot(traj_x[1:plot_start_frame], traj_y[1:plot_start_frame]; args..., color=color, linewidth=linewidth)
            scatter!(p, traj_x[intersect(1:plot_start_frame, event_indeces)], traj_y[intersect(1:plot_start_frame, event_indeces)];
                marker=:circle, markersize=3, markeralpha=scatter_alpha, color=scatter_color, label=false
            )
            anim = @animate for i ∈ iter
                Base.push!(p, traj_x[i], traj_y[i])
                if i ∈ event_indeces
                    scatter!(p, traj_x[i:i], traj_y[i:i]; marker=:circle, markersize=scatter_size, markeralpha=scatter_alpha, color=scatter_color, label=false)
                end
                p
            end every skip_frames
        end
    else  # if dim > 1 & 3D plot is requested
        traj, event_indeces, _times = traj_for_animation(history, time_stamps, N_start, N_max; coordinate_numbers=coordinate_numbers[1:3], dt=dt, nonlinear_flow=nonlinear_flow)
        args = (
            xlims=(floor(minimum(traj[1,plot_start:end]),digits=1), ceil(maximum(traj[1,plot_start:end]),digits=1)),
            ylims=(floor(minimum(traj[2,plot_start:end]),digits=1), ceil(maximum(traj[2,plot_start:end]),digits=1)),
            zlims=(floor(minimum(traj[3,plot_start:end]),digits=1), ceil(maximum(traj[3,plot_start:end]),digits=1)),
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            label=false,
            title=title,
            background=background,
            kwargs...
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
        iter = verbose ? ProgressBar(plot_start_frame:upper_limit, unit="B", unit_scale=true) : plot_start_frame:upper_limit

        tail_cgrad = cgrad([tail_color_start, tail_color_end])

        if fade
            anim = @animate for i ∈ iter
                i_tail_start = max(plot_start_frame, i - tail_length + 1)
                i_highlight_start = max(i_tail_start, i - highlight_length + 1)

                hist_end = max(i_tail_start - 1, plot_start_frame)
                hist_idx = 1:history_stride:hist_end

                p = plot(traj_x[hist_idx], traj_y[hist_idx], traj_z[hist_idx];
                    args...,
                    color=history_color,
                    alpha=history_alpha,
                    linewidth=linewidth,
                    colorbar = false,
                )

                # Note: some backends may ignore line_z in 3D; highlight overlay keeps "newest red" visible.
                plot!(p, traj_x[i_tail_start:i], traj_y[i_tail_start:i], traj_z[i_tail_start:i];
                    seriescolor=tail_cgrad,
                    line_z=1:(i - i_tail_start + 1),
                    linewidth=linewidth,
                    label=false
                )

                plot!(p, traj_x[i_highlight_start:i], traj_y[i_highlight_start:i], traj_z[i_highlight_start:i];
                    color=highlight_color,
                    linewidth=linewidth + 0.5,
                    label=false
                )

                last_ev = searchsortedlast(event_indeces, i)
                if last_ev > 0
                    ev = @view event_indeces[1:last_ev]
                    scatter!(p, traj_x[ev], traj_y[ev], traj_z[ev];
                        marker=:circle,
                        markersize=scatter_size,
                        markeralpha=scatter_alpha,
                        color=scatter_color,
                        label=false
                    )
                end

                p
            end every skip_frames
        else
            p = plot(traj_x[1:plot_start_frame], traj_y[1:plot_start_frame], traj_z[1:plot_start_frame]; args..., color=color, linewidth=linewidth)
            scatter!(p, traj_x[intersect(1:plot_start_frame, event_indeces)], traj_y[intersect(1:plot_start_frame, event_indeces)], traj_z[intersect(1:plot_start_frame, event_indeces)];
                marker=:circle, markersize=scatter_size, markeralpha=scatter_alpha, color=scatter_color, label=false
            )
            anim = @animate for i ∈ iter
                Base.push!(p, traj_x[i], traj_y[i], traj_z[i])
                if i ∈ event_indeces
                    scatter!(p, traj_x[i:i], traj_y[i:i], traj_z[i:i]; marker=:circle, markersize=3, markeralpha=scatter_alpha, color=scatter_color, label=false)
                end
                p
            end every skip_frames
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
    traj_for_animation(): extract a trajectory for animation.

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

    trajectory = history.X
    traj = trajectory[coordinate_numbers, N_start:N_max]
    traj = isa(traj, Vector) ? reshape(traj, 1, :) : traj
    v_history = history.V[coordinate_numbers, N_start:N_max]  # not used if nonlinear_flow = nothing
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

