using Zygote, ForwardDiff, ReverseDiff, PolyesterForwardDiff

"""
  Set the AD backend for the <sampler name>AD() constructors defined in src/Samplers/*.jl
"""
function set_AD_backend(AD_backend::String, U::Function, dim::Int)::Function
  
  Backend = eval(Symbol(AD_backend))

  ## If U is one dimensional and takes Float64 instead of Vector{Float64}, change ∇U accordingly:
  if dim == 1
    try
      U([1.0])
    catch
      if AD_backend == "ForwardDiff"
        return function(x::Vector)
          return Backend.derivative(U, x[1])
        end
      elseif AD_backend == "Zygote"
      return function(x::Vector)
          return Backend.gradient(U, x[1])[1]
      end
      end
    end
  end

  if AD_backend == "Zygote"
    return function(x::Vector)
        return Zygote.gradient(U, x)[1]
    end
  elseif AD_backend == "ForwardDiff"
    return function(x::Vector)
        return Backend.gradient(U, x)
    end
  else
    error("Unsupported backend: $AD_backend")
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