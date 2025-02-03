using PDMPFlux

function U_Gauss(x::Vector)
    return (sum(x.^2) - x[1]*x[2]) / 2
end

dim = 2
sampler = ZigZagAD(dim, U_Gauss, grid_size=10)
# sampler = ZigZagAD(dim, U_Gauss, grid_size=10, signed_bound=false)

N_sk, N, xinit, vinit = 100_000, 100_000, zeros(dim), ones(dim)

output = sample_skeleton(sampler, N_sk, xinit, vinit, seed=2024)
samples = sample_from_skeleton(sampler, N, output)

# diagnostic(output)
# jointplot(samples)

# anim_traj(output, 90; filename="ZigZag_SlantedGauss2D_longer.gif", title="Zig-Zag Sampler", color="#E95420", background="#F0F1EB")

using Plots
time_stamps = output.t[1:20]
x, event_indeces, _times = traj_for_animation(output, time_stamps, 1, 20; coordinate_numbers=[1,2], dt=0.1, nonlinear_flow=nothing)
args = (
    xlims=(floor(minimum(x[1,1:end]),digits=1), ceil(maximum(x[1,1:end]),digits=1)),
    ylims=(floor(minimum(x[2,1:end]),digits=1), ceil(maximum(x[2,1:end]),digits=1)),
    label=false,
    axis=false,
    title="Zig-Zag Sampler",
    color="#E95420",
    background="#F0F1EB",
    linewidth=2,
    aspect_ratio=1.2,
    size=(400, 600)
    )


p = plot(x[1,1:26], x[2,1:26]; args...)
scatter!(p, x[1,intersect(1:26, event_indeces)], x[2,intersect(1:26, event_indeces)], marker=:circle, markersize=6, markeralpha=1, color="#78C2AD", label=false)

anim = @animate for i ∈ 27:40
    Base.push!(p, x[1,i], x[2,i])
    if i ∈ event_indeces
        scatter!(p, x[1,i:i], x[2,i:i], marker=:circle, markersize=6, markeralpha=1, color="#78C2AD", label=false)
    end
end

last_frame = deepcopy(p)

for _ ∈ 1:5
    frame(anim, last_frame)
end

gif(anim, "ZigZag_Simulation.gif", fps=10)

function traj_for_animation(history, time_stamps::Vector{Float64}, N_start::Int, N_max::Int;
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