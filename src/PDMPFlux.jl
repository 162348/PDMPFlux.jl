module PDMPFlux

include("Composites.jl")
include("ADBackend.jl")
include("UpperBound.jl")

include("Samplers/AbstractPDMP.jl")
include("Samplers/StickyPDMP.jl")
include("Samplers/ZigZagSamplers.jl")
include("Samplers/BouncyParticleSamplers.jl")
include("Samplers/ForwardEventChainMonteCarlo.jl")
include("Samplers/BoomerangSamplers.jl")
include("Samplers/SpeedUpZigZagSamplers.jl")
include("Samplers/StickyZigZagSamplers.jl")

export ZigZag, ZigZagAD, BPS, BPSAD, ForwardECMC, ForwardECMCAD, StickyZigZag, StickyZigZagAD, Boomerang, BoomerangAD, SpeedUpZigZag, SpeedUpZigZagAD

include("SamplingLoop.jl")
include("StickySamplingLoop.jl")
include("sample.jl")

export sample, sample_skeleton, sample_from_skeleton

include("diagnostic.jl")
include("plot.jl")

export diagnostic, plot_traj, anim_traj, jointplot

end
