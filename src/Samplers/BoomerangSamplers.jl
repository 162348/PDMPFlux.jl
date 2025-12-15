mutable struct Boomerang{G,KF,KR,KRV,KSR,KSRV,KVJ} <: AbstractPDMP
  dim::Int
  ∇U::G
  grid_size::Int
  tmax::Float64
  refresh_rate::Float64
  vectorized_bound::Bool
  signed_bound::Bool
  adaptive::Bool
  flow::KF
  rate::KR
  rate_vect::KRV
  signed_rate::KSR
  signed_rate_vect::KSRV
  velocity_jump::KVJ
  rng::AbstractRNG
  state::Any
  AD_backend::String

# Constructor for BouncyParticle
  function Boomerang(dim::Int, ∇U; grid_size::Int=10, tmax::Union{Float64, Int}=1.0,  
            refresh_rate::Float64=0.1, vectorized_bound::Bool=false, signed_bound::Bool=true,
            adaptive::Bool=true, AD_backend::String="FiniteDiff")
    
    tmax = Float64(tmax)

    # select tmax adaptively if tmax was 0
    tmax == 0.0 ? (tmax = 1.0; adaptive = true) : nothing

    # Definition of the flow
    flow = (x,v,t) -> (x * cos(t) + v * sin(t), -x * sin(t) + v * cos(t))

    # Define rate functions
    rate_vect = nothing
    signed_rate_vect = nothing
    vectorized_bound = false  # Just to make sure. We don't use vectorization for BPS

    rate = function _global_rate(x0, v0, t)
      xt, vt = flow(x0, v0, t)
      return max(0.0, dot(∇U(xt), vt)) + refresh_rate
    end

    signed_rate = function _signed_rate(x0, v0, t)
      xt, vt = flow(x0, v0, t)
      return dot(∇U(xt), vt) + refresh_rate
    end

    # Define velocity jump function
    function velocity_jump(x, v, rng)

      ∇U_effective = x -> ∇U(x) .- x
      ∇Ux = ∇U_effective(x)
      bounce_rate = max(0.0, ∇Ux ⋅ v)
      bounce_prob = bounce_rate / (bounce_rate + refresh_rate)
      u = rand(rng)

      function reflect(v, ∇Ux)
        e = ∇Ux ./ norm(∇Ux)
        return v .- 2 * (v ⋅ e) * e
      end

      if u < bounce_prob
        return reflect(v, ∇Ux)
      else
        return randn(dim)
      end
    end

    rng = Random.default_rng()
    state = nothing
    return new{typeof(∇U), typeof(flow), typeof(rate), typeof(rate_vect), typeof(signed_rate), typeof(signed_rate_vect),
               typeof(velocity_jump)}(
        dim, ∇U, grid_size, tmax, refresh_rate, vectorized_bound, signed_bound, adaptive,
        flow, rate, rate_vect, signed_rate, signed_rate_vect, velocity_jump, rng, state, AD_backend)
  end

end  # mutable struct BPS

function BoomerangAD(dim::Int, U::Function; refresh_rate::Float64=0.0, grid_size::Int=10, tmax::Union{Float64, Int}=2.0, 
                    vectorized_bound::Bool=true, signed_bound::Bool=true, adaptive::Bool=true, AD_backend::String="Zygote")

  ∇U = set_AD_backend(AD_backend, U, dim)

  return Boomerang(dim, ∇U, refresh_rate=refresh_rate, grid_size=grid_size, tmax=tmax, 
                    signed_bound=signed_bound, adaptive=adaptive, AD_backend=AD_backend)
end