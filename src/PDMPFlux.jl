module PDMPFlux

include("Composites.jl")
include("UpperBound.jl")

include("Samplers/AbstractPDMP.jl")
# include("Samplers/Boomerang.jl")
# include("Samplers/BouncyParticle.jl")
# include("Samplers/ForwardEventChain.jl")
# include("Samplers/SpeedupZigZag.jl")
include("Samplers/ZigZagSamplers.jl")

export PDMP, ZigZag, ZigZagAD

include("SamplingLoop.jl")
include("sample.jl")

export sample, sample_skeleton, sample_from_skeleton

include("diagnostic.jl")
include("plot.jl")

export diagnostic, plot_traj, anim_traj, jointplot

using StatsPlots
export marginalhist

end
