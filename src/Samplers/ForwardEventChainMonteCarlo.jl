# Constants for Forward Event Chain Monte Carlo
const TOLERANCE = 1e-10
const MIN_DIMENSION = 3

"""
    _global_rate(x0, v0, t, ∇U, flow)

Compute the global rate for ForwardECMC.

# Arguments
- `x0`: Initial position (Vector)
- `v0`: Initial velocity (Vector)  
- `t`: Time (Float64 or ForwardDiff.Dual)
- `∇U`: Gradient of potential function
- `flow`: Flow function

# Returns
- Global rate value
"""
function _global_rate(x0::AbstractVector, v0::AbstractVector, t, ∇U, flow)
    xt, vt = flow(x0, v0, t)
    return max(zero(eltype(t)), ∇U(xt) ⋅ vt)
end

"""
    _signed_rate(x0, v0, t, ∇U, flow)

Compute the signed rate for ForwardECMC.

# Arguments
- `x0`: Initial position (Vector)
- `v0`: Initial velocity (Vector)
- `t`: Time (Float64 or ForwardDiff.Dual)
- `∇U`: Gradient of potential function
- `flow`: Flow function

# Returns
- Signed rate value (can be negative)
"""
function _signed_rate(x0, v0, t, ∇U, flow)
    xt, vt = flow(x0, v0, t)
    return ∇U(xt) ⋅ vt
end

"""
    _orthogonal_switch(vₒ, n, key, ran_p, dim)

Refresh the orthogonal component of velocity using orthogonal transformation.

# Arguments
- `vₒ::Vector{Float64}`: Orthogonal component of velocity
- `n::Vector{Float64}`: Normalized gradient direction
- `key::Random.AbstractRNG`: Random number generator
- `ran_p::Bool`: Whether to use random angle
- `dim::Int`: Dimension

# Returns
- `Vector{Float64}`: Refreshed orthogonal component
"""
function _orthogonal_switch(vₒ::Vector{Float64}, n::Vector{Float64}, 
                           key::Random.AbstractRNG, ran_p::Bool, dim::Int, positive::Bool)::Vector{Float64}
    g = randn(key, 2, dim)
    g₁ = g[1, :]
    g₂ = g[2, :]
    
    # Project onto orthogonal complement of n
    g₁ = g₁ .- (g₁ ⋅ n) * n
    g₂ = g₂ .- (g₂ ⋅ n) * n
    
    # Gram-Schmidt orthogonalization
    e₁ = g₁ / norm(g₁)
    e₂ = g₂ - (g₂ ⋅ e₁) * e₁
    e₂ /= norm(e₂)
    
    # Apply rotation
    vᵣ = vₒ .- (vₒ ⋅ e₁) * e₁ .- (vₒ ⋅ e₂) * e₂
    vₒ_new = vᵣ .+ e₂ * (e₁ ⋅ vₒ) .+  e₁ * (e₂ ⋅ vₒ)
    if ran_p
        θ = rand(key) * 2π
        vₒ_new = vᵣ .+ (cos(θ) * e₁ + sin(θ) * e₂) * (e₁ ⋅ vₒ) .+ (sin(θ) * e₁ - cos(θ) * e₂) * (e₂ ⋅ vₒ)
    end
    # Align with old direction to avoid backtracking
    if positive
        vₒ_new *= sign(vₒ ⋅ vₒ_new)
    end
    
    return vₒ_new
end

"""
    _full_refresh(vₒ, n, key, ran_p, dim)

Refresh the orthogonal component of velocity using orthogonal transformation.

# Arguments
- `vₒ::Vector{Float64}`: Orthogonal component of velocity
- `n::Vector{Float64}`: Normalized gradient direction
- `key::Random.AbstractRNG`: Random number generator
- `ran_p::Bool`: Whether to use random angle
- `dim::Int`: Dimension

# Returns
- `Vector{Float64}`: Refreshed orthogonal component
"""
function _full_refresh(n::Vector{Float64}, key::Random.AbstractRNG, dim::Int)::Vector{Float64}

    vₒ_new_unnormalised = randn(key, dim)
    vₒ_new = vₒ_new_unnormalised ./ norm(vₒ_new_unnormalised)
    
    vₒ_new = vₒ_new .- (vₒ_new ⋅ n) * n
    
    return vₒ_new
end

"""
    _velocity_jump_event_chain(x, v, key, ∇U, dim, mix_p, ran_p)

Perform velocity jump for Forward Event Chain Monte Carlo.

# Arguments
- `x::Vector{Float64}`: Current position
- `v::Vector{Float64}`: Current velocity
- `key::Random.AbstractRNG`: Random number generator
- `∇U::Function`: Gradient of potential function
- `dim::Int`: Dimension
- `mix_p::Float64`: Mixture probability for refreshment
- `ran_p::Bool`: Whether to use random orthogonal refresh

# Returns
- `Vector{Float64}`: New velocity
"""
function _velocity_jump_event_chain(x::Vector{Float64}, v::Vector{Float64}, 
                                  key::Random.AbstractRNG, ∇U::Function, 
                                  dim::Int, mix_p::Float64, ran_p::Bool,
                                  switch::Bool, positive::Bool)::Vector{Float64}

    # Generate random parameter for reflection
    u = rand(key)
    ρ = -sqrt(1 - u^(2 / (dim - 1)))
    
    # Compute normalized gradient direction
    n = norm(∇U(x)) == 0 ? zeros(dim) : ∇U(x) ./ norm(∇U(x))
    
    # Decompose velocity into parallel and orthogonal components
    vₚ = (v ⋅ n) * n  # Parallel component along gradient
    vₒ = v - vₚ       # Orthogonal component
    
    # Handle degenerate case where orthogonal component is too small
    if norm(vₒ) < TOLERANCE
        vₒ = randn(key, dim)
        vₒ = vₒ .- (vₒ ⋅ n) * n
    end
    
    u2 = rand(key)
    if u2 >= mix_p
        # No refresh case - return early
        v_out = vₒ / norm(vₒ) * sqrt(1 - ρ^2) .+ ρ * n
        return v_out
    end
    
    # Refresh case
    vₒ_proposal = switch ? _orthogonal_switch(vₒ, n, key, ran_p, dim, positive) : _full_refresh(n, key, dim)
    v_out = vₒ_proposal / norm(vₒ_proposal) * sqrt(1 - ρ^2) .+ ρ * n
    
    return v_out
end

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
  rate::Union{Function, Nothing}
  rate_vect::Union{Function, Nothing}
  signed_rate::Union{Function, Nothing}
  signed_rate_vect::Union{Function, Nothing}
  velocity_jump::Function
  state::Union{PDMPState, Nothing}
  ran_p::Bool  # Whether to use ran-p-orthogonal refresh or orthogonal switch
  switch::Bool  # orthogonal switch or full refresh for the orthogonal component
  mix_p::Float64  # Mixture probability for refreshment
  AD_backend::String
  # p::Int  # TODO: Number of components to be refreshed each time

  """
  Constructor for ForwardECMC sampler.
  
  # Arguments
  - `dim::Int`: Dimension of the state space
  - `∇U::Function`: Gradient of the potential function
  
  # Keywords
  - `grid_size::Int=10`: Number of grid points for upper bound
  - `tmax::Union{Float64, Int}=2.0`: Maximum time horizon
  - `signed_bound::Bool=true`: Use signed bound strategy
  - `adaptive::Bool=true`: Use adaptive time horizon
  - `ran_p::Bool=false`: Use random orthogonal refresh
  - `mix_p::Float64=0.5`: Mixture probability for refreshment
  """
  function ForwardECMC(dim::Int, ∇U::Function; grid_size::Int=10, tmax::Union{Float64, Int}=2.0,
    signed_bound::Bool=true, adaptive::Bool=true, ran_p::Bool=false, mix_p::Float64=0.5, switch::Bool=true, positive::Bool=true, AD_backend::String="Undefined")

    # Input validation and preprocessing
    tmax = Float64(tmax)
    tmax == 0.0 ? (tmax = 1.0; adaptive = true) : nothing

    if dim < MIN_DIMENSION
        throw(ArgumentError("The dimension must be at least $MIN_DIMENSION to use the ForwardEventChain. Got dimension $dim"))
    end

    flow = (x, v, t) -> (x .+ v .* t, v)

    # Configure rate functions
    rate_vect = nothing
    signed_rate_vect = nothing
    vectorized_bound = false  # Don't use vectorization for FECMC
    refresh_rate = 0.0        # Not used in FECMCvectorization for BPS
    refresh_rate = 0.0  # Can't be used

    # Create rate function closures
    rate = (x0, v0, t) -> _global_rate(x0, v0, t, ∇U, flow)
    signed_rate = (x0, v0, t) -> _signed_rate(x0, v0, t, ∇U, flow)
    velocity_jump = (x, v, key) -> _velocity_jump_event_chain(x, v, key, ∇U, dim, mix_p, ran_p, switch, positive)

    new(dim, ∇U, grid_size, tmax, refresh_rate, vectorized_bound, signed_bound, adaptive, flow, rate, rate_vect, signed_rate, signed_rate_vect, velocity_jump, nothing, ran_p, switch, mix_p, AD_backend)
  end
end  # mutable struct ForwardECMC

"""
    create_gradient_function(U::Function, dim::Int, AD_backend::Module)

Create a gradient function compatible with automatic differentiation.

# Arguments
- `U::Function`: Potential function
- `dim::Int`: Dimension  
- `AD_backend::Module`: AD backend module

# Returns
- `Function`: Gradient function that works with Float64 and ForwardDiff.Dual
"""
function create_gradient_function(U::Function, dim::Int, AD_backend::String)::Function
    Backend = eval(Symbol(AD_backend))
    
    if dim == 1
        try
            U([1.0])
            return function(x)
                return Backend.gradient(U, x)[1]
            end
        catch e
            @warn "1D function detected, adapting gradient function: $e"
            return function(x)
                return Backend.gradient(U, x[1])[1]
            end
        end
    else
        if AD_backend == "Zygote"
            return function(x)
                return Backend.gradient(U, x)[1]
            end
        elseif AD_backend == "ForwardDiff"
            return function(x)
                return Backend.gradient(U, x)
            end
        else
            error("Unsupported backend: $AD_backend")
        end
    end
end

"""
    ForwardECMCAD(dim, U; kwargs...)

Create ForwardECMC sampler with automatic differentiation.

# Arguments
- `dim::Int`: Dimension of the state space
- `U::Function`: Potential function

# Keywords
- `grid_size::Int=10`: Number of grid points for upper bound
- `tmax::Union{Float64, Int}=2.0`: Maximum time horizon
- `signed_bound::Bool=true`: Use signed bound strategy
- `adaptive::Bool=true`: Use adaptive time horizon
- `AD_backend::String="Zygote"`: Automatic differentiation backend
- `ran_p::Bool=true`: Use random orthogonal refresh
- `mix_p::Float64=0.5`: Mixture probability for refreshment

# Returns
- `ForwardECMC`: Configured sampler instance
"""
function ForwardECMCAD(dim::Int, U::Function; grid_size::Int=10, tmax::Union{Float64, Int}=2.0,
    signed_bound::Bool=true, adaptive::Bool=true, AD_backend::String="Zygote",
    ran_p::Bool=true, mix_p::Float64=0.5, switch::Bool=true, positive::Bool=true)
    
    # Create gradient function using the helper function
    ∇U = create_gradient_function(U, dim, AD_backend)
    
    # Create and return sampler
    return ForwardECMC(dim, ∇U, grid_size=grid_size, tmax=tmax,
                      signed_bound=signed_bound, adaptive=adaptive, ran_p=ran_p, mix_p=mix_p, switch=switch, positive=positive, AD_backend=AD_backend)
end