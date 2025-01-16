mutable struct ForwardECMC <: AbstractPDMP
  dim::Int
  ∇U::Function
  grid_size::Int
  tmax::Float64
  refresh_rate::Float64
  vectorized_bound::Bool
  signed_bound::Bool
  adaptive::Bool
  flow::Function
  rate::Any
  rate_vect::Any
  signed_rate::Any
  signed_rate_vect::Any
  velocity_jump::Function
  state::Any
  ran_p::Bool  # Whether to use ran-p-orthogonal refresh or orthogonal switch. See Michel, Drumus & Sénécal p.694 from: https://doi.org/10.1080/10618600.2020.1750417
  mix_p::Float64  # Mixture probability for refreshment
  # p::Int  # TODO: Number of components to be refreshed each time

  function ForwardECMC(dim::Int, ∇U::Function; grid_size::Int=10, tmax::Union{Float64, Int}=2.0,
    signed_bound::Bool=true,
    adaptive::Bool=true, ran_p::Bool=true, mix_p::Float64=0.5)

    tmax = Float64(tmax)
    # select tmax adaptively if tmax was 0
    tmax == 0.0 ? (tmax = 1.0; adaptive = true) : nothing
    if dim <= 2
        throw(ArgumentError("The dimension must be greater than 2 to use the ForwardEventChain"))
    end

    flow = (x, v, t) -> (x .+ v .* t, v)

    # Define rate functions
    rate_vect = nothing
    signed_rate_vect = nothing
    vectorized_bound = false  # Don't use vectorization for BPS
    refresh_rate = 0.0  # Can't be used

    rate = function _global_rate(x0, v0, t)
      xt, vt = flow(x0, v0, t)
      return max(0.0, ∇U(xt) ⋅ vt)
    end

    signed_rate = function _signed_rate(x0, v0, t)
      xt, vt = flow(x0, v0, t)
      return ∇U(xt) ⋅ vt
    end

    velocity_jump = function _velocity_jump_event_chain(x, v, key)
    u = rand(key)
    ρ = -sqrt(1 - u^(2 / (dim - 1)))
    n = norm(∇U(x)) == 0 ? zeros(dim) : ∇U(x) ./ norm(∇U(x))
    vₚ = (v ⋅ n) * n  # Parallel component of v along n (normalized ∇U(x))
    vₒ = v - vₚ  # Orthogonal component of v
  
    function _refresh_ortho(key, ran_p)
        g = randn(2, dim)
        g₁ = g[1, :]
        g₂ = g[2, :]
        g₁ = g₁ .- (g₁ ⋅ n) * n
        g₂ = g₂ .- (g₂ ⋅ n) * n
        e₁ = g₁ / norm(g₁)
        e₂ = g₂ - (g₂ ⋅ e₁) * e₁
        e₂ /= norm(e₂)
        θ = ran_p ? rand(key) * 2π : π/2
        vᵣ = vₒ .- (vₒ ⋅ e₁) * e₁ .- (vₒ ⋅ e₂) * e₂
        vₒ_new = vᵣ .+ (cos(θ) * e₁ + sin(θ) * e₂) * (e₁ ⋅ vₒ) .+ (sin(θ) * e₁ - cos(θ) * e₂) * (e₂ ⋅ vₒ)
        # vₒ_new /= norm(vₒ_new)  # 零除算が発生する
        vₒ_new *= sign(vₒ ⋅ vₒ_new)
        return vₒ_new
    end
  
    u2 = rand(key)
    vₒ_proposal = norm(vₒ) == 0 ? zeros(dim) : vₒ / norm(vₒ)
    vₒ_proposal = u2 < mix_p ? _refresh_ortho(key, ran_p) : vₒ_proposal
    v_out = vₒ_proposal * sqrt(1 - ρ^2) .+ ρ * n
    return v_out
  end

    new(dim, ∇U, grid_size, tmax, refresh_rate, vectorized_bound, signed_bound, adaptive, flow, rate, rate_vect, signed_rate, signed_rate_vect, velocity_jump, nothing, ran_p, mix_p)
  end
end  # mutable struct ForwardECMC



function ForwardECMCAD(dim::Int, U::Function; grid_size::Int=10, tmax::Union{Float64, Int}=2.0,
    signed_bound::Bool=true, adaptive::Bool=true,  AD_backend::String="Zygote",
    ran_p::Bool=true, mix_p::Float64=0.5)

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
  
    return ForwardECMC(dim, ∇U, grid_size=grid_size, tmax=tmax, 
                      signed_bound=signed_bound, adaptive=adaptive, ran_p=ran_p, mix_p=mix_p)
end