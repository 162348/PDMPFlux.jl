using Zygote, ForwardDiff, ReverseDiff, PolyesterForwardDiff
# Enzyme is conditionally loaded
const HAS_ENZYME = try
    using Enzyme
    true
catch
    false
end

"""
  Set the AD backend for the <sampler name>AD() constructors defined in src/Samplers/*.jl
"""
function set_AD_backend(AD_backend::String, U::Function, dim::Int)
  if dim <= 0
    throw(ArgumentError("dimension dim must be positive. Current value: $dim"))
  end
  return create_gradient_function(U, dim, AD_backend)
end

"""
    create_gradient_function(U::Function, dim::Int, AD_backend::String)

Common helper to build a gradient function `∇U(x)` from `U` and resolve
the effective AD backend once at initialization time.

- `x` is assumed to be an `AbstractVector`
- Always returns a **vector gradient of length `dim`**
  (even when `dim == 1` and the user provides `U(::Real)`, it returns `[dU/dx]`)
"""
function create_gradient_function(U::Function, dim::Int, AD_backend::String)
  # NOTE:
  # - This helper is used by `*AD(...)` constructors where the user *usually* provides a potential `U(x)::Real`.
  # - In practice, users sometimes (accidentally) pass an already-differentiated gradient function `∇U(x)::AbstractVector`.
  #   ForwardDiff.gradient/Zygote.gradient would then throw:
  #     "gradient(f, x) expects that f(x) is a real number ..."
  #   We detect this case once here and simply return the provided gradient.

  # Determine calling convention for dim==1: U(::Real) vs U(::AbstractVector)
  input_mode = :vector
  probe_x_vec = ones(Float64, dim)
  probe_y = nothing
  if dim == 1
    try
      probe_y = U([1.0])
      input_mode = :vector
    catch
      probe_y = U(1.0)
      input_mode = :scalar
    end
  else
    probe_y = U(probe_x_vec)
    input_mode = :vector
  end

  # Case 1: user already provided a gradient function ∇U(x)
  if probe_y isa AbstractVector
    if length(probe_y) != dim
      throw(ArgumentError("Expected U(x) to return a scalar potential (Real) or a gradient vector of length $dim. Got AbstractVector of length $(length(probe_y))."))
    end
    ∇U = function(x::AbstractVector)
      g = (dim == 1 && input_mode == :scalar) ? U(x[1]) : U(x)
      if dim == 1 && (g isa Real)
        return [g]
      end
      return g
    end
    return (; gradient=∇U, AD_backend=AD_backend)
  elseif !(probe_y isa Real)
    throw(ArgumentError("Expected U(x) to return a scalar potential (Real) or a gradient vector (AbstractVector). Got $(typeof(probe_y))."))
  end

  # Case 2: user provided a scalar potential U(x)::Real → build ∇U(x)
  resolved_backend = AD_backend
  if AD_backend == "ForwardDiff"
    resolved_backend = if dim == 1 && input_mode == :scalar
      try
        ForwardDiff.derivative(U, 1.0)
        "ForwardDiff"
      catch
        "Zygote"
      end
    else
      try
        ForwardDiff.gradient(U, probe_x_vec)
        "ForwardDiff"
      catch
        "Zygote"
      end
    end
  end

  if resolved_backend == "Zygote"
    if dim == 1 && input_mode == :scalar
      ∇U = function(x::AbstractVector)
        return [Zygote.gradient(U, x[1])[1]]
      end
    else
      ∇U = function(x::AbstractVector)
        return Zygote.gradient(U, x)[1]
      end
    end
  elseif resolved_backend == "ForwardDiff"
    if dim == 1 && input_mode == :scalar
      ∇U = function(x::AbstractVector)
        return [ForwardDiff.derivative(U, x[1])]
      end
    else
      ∇U = function(x::AbstractVector)
        return ForwardDiff.gradient(U, x)
      end
    end
  elseif resolved_backend == "ReverseDiff"
    if dim == 1 && input_mode == :scalar
      ∇U = function(x::AbstractVector)
        return [ReverseDiff.gradient(z -> U(z[1]), [x[1]])[1]]
      end
    else
      ∇U = function(x::AbstractVector)
        return ReverseDiff.gradient(U, x)
      end
    end
  elseif resolved_backend == "Enzyme"
    if !HAS_ENZYME
      throw(ArgumentError("Enzyme package is not available. Please install it with: ] add Enzyme"))
    end
    if dim == 1 && input_mode == :scalar
      ∇U = function(x::AbstractVector)
        grad_result = Enzyme.gradient(Enzyme.Reverse, z -> U(z[1]), Enzyme.Active([x[1]]))
        return [grad_result[1][1]]
      end
    else
      ∇U = function(x::AbstractVector)
        grad_result = Enzyme.gradient(Enzyme.Reverse, U, x)
        return grad_result[1]
      end
    end
  else
    throw(ArgumentError("Unsupported backend: $resolved_backend. Supported backends: Zygote, ForwardDiff, ReverseDiff, Enzyme"))
  end

  return (; gradient=∇U, AD_backend=resolved_backend)
end

function threaded_gradient(f, x::AbstractArray, chunk::ForwardDiff.Chunk{C}, check = Val{false}()) where {C}
  dx = similar(x)

  function scaler_f(x)  # f([x]) returns a vector with one element.
    return f(x)[1]
  end

  PolyesterForwardDiff.threaded_gradient!(scaler_f, dx, x, chunk, check)
  return dx
end