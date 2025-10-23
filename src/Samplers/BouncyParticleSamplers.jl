mutable struct BPS <: AbstractPDMP
  dim::Int
  ∇U::Function
  grid_size::Int
  tmax::Float64
  refresh_rate::Float64
  vectorized_bound::Bool
  signed_bound::Bool
  adaptive::Bool
  flow::Function
  rate::Function
  rate_vect::Union{Function, Nothing}
  signed_rate::Function
  signed_rate_vect::Union{Function, Nothing}
  velocity_jump::Function
  state::Any

# Constructor for BouncyParticle
  function BPS(dim::Int, ∇U::Function; grid_size::Int=10, tmax::Union{Float64, Int}=1.0,  
            refresh_rate::Float64=0.1, vectorized_bound::Bool=false, signed_bound::Bool=true,
            adaptive::Bool=true)
    
    tmax = Float64(tmax)

    # select tmax adaptively if tmax was 0
    tmax == 0.0 ? (tmax = 1.0; adaptive = true) : nothing

    # Definition of the flow
    flow = (x, v, t) -> (x .+ v .* t, v)

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
        ∇Ux = ∇U(x)
        bounce_rate = max(0.0, dot(∇Ux, v))
        bounce_prob = bounce_rate / (bounce_rate + refresh_rate)
        u = rand(rng)

        function reflect(v, ∇Ux)
            e = ∇Ux ./ norm(∇Ux)
            return v .- 2 * dot(v, e) * e
        end

        if u < bounce_prob
            return reflect(v, ∇Ux)
        else
            v = randn(dim)
            return v ./ norm(v)
            # return randn(dim)
        end
    end

    return new(dim, ∇U, grid_size, tmax, refresh_rate, vectorized_bound, signed_bound, adaptive,
                          flow, rate, rate_vect, signed_rate, signed_rate_vect, velocity_jump, nothing)
  end

end  # mutable struct BPS

function BPSAD(dim::Int, U::Function; refresh_rate::Float64=0.0, grid_size::Int=10, tmax::Union{Float64, Int}=2.0, 
                    vectorized_bound::Bool=true, signed_bound::Bool=true, adaptive::Bool=true, AD_backend::String="Zygote")

  ∇U = nothing
  AD_backend = eval(Symbol(AD_backend))

  ## If U is one dimensional and takes Float64 instead of Vector{Float64}, change ∇U accordingly:
  if dim == 1
    try
        U([1.0])
    catch
        ∇U = function(x::Vector)
            return AD_backend.gradient(U, x[1])[1]
        end
    else
        ∇U = function(x::Vector)
            return AD_backend.gradient(U, x)[1]
        end
    end
  else
      ∇U = function(x::Vector)
          return AD_backend.gradient(U, x)[1]
      end
  end

  return BPS(dim, ∇U, refresh_rate=refresh_rate, grid_size=grid_size, tmax=tmax, 
                    signed_bound=signed_bound, adaptive=adaptive)
end