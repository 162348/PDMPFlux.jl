using PDMPFlux

N_sk = 1_000_000 # number of skeleton points
N = 1_000_000 # number of samples

function U_banana(x::Vector)
  mean_x2 = (x[1]^2 - 1)
  return -(- x[1]^2 + -(x[2] - mean_x2)^2 - sum(x[3:end].^2)) / 2
end

function runtest(N_sk::Int, N::Int; dim::Int=50, seed::Int=8, vectorized::Union{Bool, Nothing}=nothing)

    xinit = ones(dim)
    vinit = ones(dim)
    sampler = ZigZagAD(dim, U_banana, grid_size=10, AD_backend="Zygote")
    if vectorized !== nothing && vectorized
      sampler = ZigZagAD(dim, U_banana, grid_size=10, vectorized_bound=true, signed_bound=false)
    elseif vectorized !== nothing && !vectorized
      sampler = ZigZagAD(dim, U_banana, grid_size=10, vectorized_bound=false, signed_bound=false)
    end
    
    out = sample_skeleton(sampler, N_sk, xinit, vinit, seed=seed, verbose = true)
    #samples = sample_from_skeleton(sampler, N, out)
    # diagnostic(out)

    return out#, samples
end

runtest(N_sk, N)  # 1:52 seconds, while pdmp_jax only takes 22 seconds
### AD_backend = "Zygote"
# 1:52, 1:47, 1:55 seconds

### AD_backend = "ForwardDiff"
# about 5 minutes

##### upper_bound_grid_vect
# Zygote を使うとあり得ないほど遅くなる
# PolyesterForwardDiff を使うとまだマシだが 3 分くらいかかる