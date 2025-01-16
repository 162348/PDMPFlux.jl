using Zygote, ForwardDiff, ReverseDiff, PolyesterForwardDiff

"""
  Set the AD backend for the <sampler name>AD() constructors defined in src/Samplers/*.jl
"""
function set_AD_backend(AD_backend::String, U::Function, dim::Int)::Function
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
      if AD_backend == eval(Symbol(AD_backend))
        ∇U = function(x::Vector)
          return AD_backend.jacobian(U, x)[1][1,1]
        end
      else
        ∇U = function(x::Vector)
          return AD_backend.jacobian(U, x)[1]
        end
      end
    end
  else
      ∇U = function(x::Vector)
          return AD_backend.gradient(U, x)[1]
      end
  end

  return ∇U
end

function threaded_gradient(f, x::AbstractArray, chunk::ForwardDiff.Chunk{C}, check = Val{false}()) where {C}
  dx = similar(x)

  function scaler_f(x)  # f([x]) returns a vector with one element.
    return f(x)[1]
  end

  PolyesterForwardDiff.threaded_gradient!(scaler_f, dx, x, chunk, check)
  return dx
end