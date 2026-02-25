using Random

"""
    In PDMPFlux, every PDMP sampler is defined to be of subtype of AbstractPDMP.
"""
abstract type AbstractPDMP end

"""
    _pdmp_ad_backend(pdmp) -> String

Choose the AD backend used for upper-bound computations (`upper_bound_grid*`) inside a PDMP.

- If `pdmp` has an `AD_backend` field, it is preferred.
- `"Undefined"` or an empty string is treated conservatively as `"FiniteDiff"`.
- If the field does not exist, defaults to `"FiniteDiff"`.

"""
function _pdmp_ad_backend(pdmp)::String
    if hasproperty(pdmp, :AD_backend)
        ab = getproperty(pdmp, :AD_backend)
        if ab === nothing
            return "FiniteDiff"
        end
        ab = String(ab)
        return (ab == "" || ab == "Undefined") ? "FiniteDiff" : ab
    end
    return "FiniteDiff"
end

"""
    Abstract PDMP sampler type.

    This docstring describes the common fields and methods expected from samplers that
    subtype `AbstractPDMP`.

    Common fields (by convention):
        dim::Int: dimension of the state space
        refresh_rate::Float64: refresh rate (if used by the sampler)
        ∇U::Function: gradient of the potential
        grid_size::Int: number of grid points used for discretizing time in upper bounds
        tmax::Float64: default horizon / maximum time for the bound grid
        adaptive::Bool: whether to adapt the horizon during sampling
        vectorized_bound::Bool: whether the bound strategy is vectorized
        signed_bound::Bool: whether signed-rate strategies are used
        flow::Function: deterministic flow / integrator
        rate::Function: (unsigned) event rate
        rate_vect::Function: vectorized event rate (if applicable)
        signed_rate::Function: signed event rate (if applicable)
        signed_rate_vect::Function: vectorized signed event rate (if applicable)
        velocity_jump::Function: velocity update at events
        state::Any: sampler state (often `PDMPState` or `nothing`)

    Common methods:
        init_state(pdmp, xinit, vinit, seed) -> PDMPState
        sample_skeleton(pdmp, n_sk, xinit, vinit; seed, verbose) -> PDMPHistory
        sample_from_skeleton(pdmp, N, history) -> Matrix{Float64}
        sample(pdmp, N_sk, N_samples, xinit, vinit; seed, verbose) -> Matrix{Float64}
"""
# struct PDMP <: AbstractPDMP
#     dim::Int
#     refresh_rate::Float64
#     ∇U::Function
#     grid_size::Int
#     tmax::Float64

#     vectorized_bound::Bool
#     signed_bound::Bool
#     adaptive::Bool

#     flow::Function
#     rate::Function
#     velocity_jump::Function
#     state::Union{PDMPState, Nothing}
# end


"""
    init_state():
    Initialize and attach a `PDMPState` to a PDMP sampler.

    Args:
        xinit (Float[Array, "dim"]): The initial position.
        vinit (Float[Array, "dim"]): The initial velocity.
        seed (int): The seed for random number generation.
        upper_bound_vect (bool, optional): Whether to use vectorized upper bound function. Defaults to False.
        signed_rate (bool, optional): Whether to use signed rate function. Defaults to False.
        adaptive (bool, optional): Whether to use adaptive upper bound. Defaults to False.
        constant_bound (bool, optional): Whether to use constant upper bound. Defaults to False.

    Returns:
        PDMPState: The initialized PDMP state.
"""
function init_state(pdmp::AbstractPDMP, xinit::AbstractVector{Float64}, vinit::AbstractVector{Float64}, seed::Union{Int, Nothing}=nothing)

    # Check that xinit and vinit match pdmp.dim
    if length(xinit) != pdmp.dim || length(vinit) != pdmp.dim
        throw(DimensionMismatch("xinit and vinit must have the same dimension as pdmp.dim ($(pdmp.dim)). Current dimensions: xinit ($(length(xinit))), vinit ($(length(vinit)))"))
    end

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    pdmp.rng = rng

    # rate / rate_vect / refresh_rate depend on signed_bound
    if pdmp.signed_bound
        rate = pdmp.signed_rate
        rate_vect = pdmp.signed_rate_vect
        refresh_rate = pdmp.refresh_rate
    else
        rate = pdmp.rate
        rate_vect = pdmp.rate_vect
        refresh_rate = 0.0
    end

    if pdmp.grid_size < 0
        throw(ArgumentError("grid_size must be non-negative. Current value: $(pdmp.grid_size)"))
    end

    ad_backend = _pdmp_ad_backend(pdmp)

    # If grid_size == 0, use a constant-bound strategy via Brent's method
    if pdmp.grid_size == 0
        upper_bound_func = function(x, v, horizon)
            func = t -> pdmp.rate(x, v, t)  # Always use the unsigned rate here (independent of pdmp.signed_bound).
            return upper_bound_constant(func, 0.0, horizon)
        end
    elseif !pdmp.vectorized_bound
        upper_bound_func = function(x, v, horizon)
            func = t -> rate(x, v, t)
            return upper_bound_grid(func, 0.0, horizon, pdmp.grid_size, refresh_rate; AD_backend=ad_backend)
        end
    else
        upper_bound_func = function(x, v, horizon)
            func = t -> rate_vect(x, v, t)
            return upper_bound_grid_vect(func, 0.0, horizon, pdmp.grid_size; AD_backend=ad_backend) 
        end
    end

    

    boundox = upper_bound_func(xinit, vinit, pdmp.tmax)
    state = PDMPState(
        xinit,
        vinit,
        0.0,
        pdmp.tmax,
        upper_bound_func,
        boundox,
        pdmp.adaptive
    )
    pdmp.state = state
    return state

end

# function init_state(pdmp::AbstractPDMP, xinit::Union{Float64, Int}, vinit::Union{Float64, Int}, seed::Int)
#     xinit = Float64(xinit)
#     vinit = Float64(vinit)
#     return init_state(pdmp, [xinit], [vinit], seed)
# end
