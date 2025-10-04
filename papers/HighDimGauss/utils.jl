module MyUtils
# use this with:
# include("utils.jl")
# using .MyUtils

function U(x::AbstractVector)
  return sum(x.^2) / 2
end

function Ψ(x::AbstractVector)
  return (sum(x.^2) - dim) / sqrt(dim)
end

using Random
rng = MersenneTwister(20250428)

function get_radial_momentum(sampler, x, v, t)
  flow = sampler.flow
  tt = t[2:end] .- 0.01
  tt_ = tt - t[1:end-1]
  tuple = map(flow, eachcol(x[:,1:end-1]), eachcol(v[:,1:end-1]), tt_)
  xx = hcat(getindex.(tuple, 1)...)
  vv = hcat(getindex.(tuple, 2)...)

  s = vec(sum(x .* v, dims=1))
  ss = vec(sum(xx .* vv, dims=1))

  s_mixed = collect(Iterators.flatten(zip(vec(s), vec(ss))))
  t_mixed = collect(Iterators.flatten(zip(vec(t), vec(tt))))

  return s_mixed, t_mixed
end

end