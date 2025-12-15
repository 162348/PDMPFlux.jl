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
  return create_gradient_function(U, dim, AD_backend)
end

"""
    create_gradient_function(U::Function, dim::Int, AD_backend::String)::Function

`U` の勾配関数 `∇U(x)` を生成する共通ヘルパ。

- `x` は `AbstractVector` を想定
- **常に長さ `dim` のベクトル勾配**を返す（`dim==1` で `U(::Real)` の場合も `[dU/dx]` を返す）
"""
function create_gradient_function(U::Function, dim::Int, AD_backend::String)
  # Detect 1D scalar-input U(::Real) vs vector-input U(::AbstractVector)
  scalar_1d = false
  if dim == 1
    try
      U([1.0])
    catch
      scalar_1d = true
    end
  end

  if AD_backend == "Zygote"
    if dim == 1 && scalar_1d
      return function(x::AbstractVector)
        return [Zygote.gradient(U, x[1])[1]]
      end
    else
      return function(x::AbstractVector)
        return Zygote.gradient(U, x)[1]
      end
    end
  elseif AD_backend == "ForwardDiff"
    if dim == 1 && scalar_1d
      return function(x::AbstractVector)
        return [ForwardDiff.derivative(U, x[1])]
      end
    else
      return function(x::AbstractVector)
        return ForwardDiff.gradient(U, x)
      end
    end
  elseif AD_backend == "ReverseDiff"
    if dim == 1 && scalar_1d
      return function(x::AbstractVector)
        return [ReverseDiff.gradient(z -> U(z[1]), [x[1]])[1]]
      end
    else
      return function(x::AbstractVector)
        return ReverseDiff.gradient(U, x)
      end
    end
  elseif AD_backend == "Enzyme"
    if !HAS_ENZYME
      throw(ArgumentError("Enzyme package is not available. Please install it with: ] add Enzyme"))
    end
    if dim == 1 && scalar_1d
      return function(x::AbstractVector)
        grad_result = Enzyme.gradient(Enzyme.Reverse, z -> U(z[1]), Enzyme.Active([x[1]]))
        return [grad_result[1][1]]
      end
    else
      return function(x::AbstractVector)
        grad_result = Enzyme.gradient(Enzyme.Reverse, U, x)
        return grad_result[1]
      end
    end
  else
    throw(ArgumentError("Unsupported backend: $AD_backend. Supported backends: Zygote, ForwardDiff, ReverseDiff, Enzyme"))
  end
end

function threaded_gradient(f, x::AbstractArray, chunk::ForwardDiff.Chunk{C}, check = Val{false}()) where {C}
  dx = similar(x)

  function scaler_f(x)  # f([x]) returns a vector with one element.
    return f(x)[1]
  end

  PolyesterForwardDiff.threaded_gradient!(scaler_f, dx, x, chunk, check)
  return dx
end