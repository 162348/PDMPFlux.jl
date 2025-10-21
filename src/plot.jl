# using RecipesBase
# using Plots
using StatsPlots
# using Statistics  # mean()

function jointplot(samples::Matrix{Float64}; coordinate_numbers=[1,2], histcolor="#78C2AD", cmap=:summer, filename=nothing)
    p = marginalhist(samples[coordinate_numbers[1],:],samples[coordinate_numbers[2],:],linecolor=histcolor,color=cmap,xlabel=L"x_1",ylabel=L"x_2")
    if !isnothing(filename)
        filename = occursin(".", filename) ? filename : filename * ".svg"
        savefig(p, filename)
    end
    return p
end

function marginalplot(samples::Matrix{Float64}; d::Int64, histcolor="#78C2AD", filename=nothing, with_kde::Bool=true,
    bins=:auto, kdecolor="#E95420", linewidth=2, alpha=0.7, title=nothing, xlabel=nothing, ylabel="Density")
    data = samples[d, :]
    
    # デフォルトのラベル設定
    if isnothing(xlabel)
        xlabel = "x_$d"
    end
    if isnothing(title)
        title = "Histogram and KDE for dimension $d"
    end
    
    p = plot()
    
    histogram!(p, data, 
              bins=bins,
              color=histcolor,
              alpha=alpha,
              normalize=:pdf,  # 密度として正規化
              label="Histogram")
    
    if with_kde
        density!(p, data,
                color=kdecolor,
                linewidth=linewidth,
                label="KDE")
    end
    
    xlabel!(p, xlabel)
    ylabel!(p, ylabel)
    title!(p, title)
    
    if !isnothing(filename)
        filename = occursin(".", filename) ? filename : filename * ".svg"
        savefig(p, filename)
    end
    
    return p
end

# import Base: push!

# @recipe function f(history::PDMPFlux.PDMPHistory)
#     @series begin
#         seriestype := :histogram
#         x = diff(history.t)
#         bins := :auto
#         title := "Time between events histogram"
#         xlabel := "Time"
#         ylabel := "Frequency"
#         subplot := 1
#         x
#     end

#     @series begin
#         seriestype := :histogram
#         x = history.ar
#         bins := :auto
#         title := "Acceptance rate histogram"
#         xlabel := "Rate"
#         ylabel := "Frequency"
#         subplot := 2
#         x
#     end

#     # @series begin
#     #     seriestype := :vline
#     #     x = [mean(history.ar)]
#     #     line := :dash
#     #     label := "Mean: $(round(mean(history.ar), digits=3))"
#     #     color := :red
#     #     subplot := 2
#     #     x
#     # end

#     # @series begin
#     #     seriestype := :histogram
#     #     x = history.hitting_horizon
#     #     bins := :auto
#     #     title := "Hitting horizon histogram"
#     #     xlabel := "Horizon"
#     #     ylabel := "Frequency"
#     #     yscale := :log10
#     #     subplot := 3
#     #     x
#     # end

#     # @series begin
#     #     seriestype := :histogram
#     #     x = history.rejected
#     #     bins := :auto
#     #     title := "Rejection histogram"
#     #     xlabel := "Rejections"
#     #     ylabel := "Frequency"
#     #     yscale := :log10
#     #     subplot := 4
#     #     x
#     # end
# end