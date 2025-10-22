using Zygote, ForwardDiff, ReverseDiff, PolyesterForwardDiff, Enzyme

struct _UF{F}
  f::F
end
# Enzyme が見るメソッドを具体化：Vector{Float64}→Float64
@inline (w::_UF)(x::Vector{Float64})::Float64 = w.f(x)

# Uvec は (AbstractVector)->Real の想定
function enzyme_gradient_builder(Uvec::Function, dim::Int)
  w = _UF(Uvec)  # 環境を捕まえた closure を具象 functor に

  return function (x::AbstractVector)
      # 具体型にそろえる（必要ならコピー）
      xv = Vector{Float64}(x)
      dx = similar(xv); fill!(dx, 0.0)

      # ⬇︎ sugar を使わず低レベル API 直叩き
      Enzyme.autodiff(Enzyme.Reverse, w, Enzyme.Duplicated(xv, dx))

      return dx
  end
end

"""
  Set the AD backend for the <sampler name>AD() constructors defined in src/Samplers/*.jl
"""
function set_AD_backend(AD_backend::Union{String,Symbol}, U::Function, dim::Int)::Function

  b = Symbol(AD_backend)

  # ベクトル→スカラーに正規化（dim==1 で U(::Float64) だけ持つ場合に対応）
  Uvec = if dim == 1 && applicable(U, 0.0) && !applicable(U, zeros(1))
      (x::AbstractVector)->U(x[1])
  elseif applicable(U, zeros(dim))
      (x::AbstractVector)->U(x)
  else
      error("U must accept Vector{<:Real} of length $dim, or Float64 when dim==1.")
  end

  y0 = Uvec(zeros(dim))
  y0 isa Real || error("U must return a scalar Real, got $(typeof(y0)).")

  if b === :ForwardDiff
      return x::AbstractVector -> ForwardDiff.gradient(Uvec, x)

  elseif b === :PolyesterForwardDiff
      return function (x::AbstractVector)
          dx = similar(x)
          # チャンクは自動選択（または 8 固定でも可）
          ch = ForwardDiff.Chunk(ForwardDiff.pickchunksize(length(x)))
          PolyesterForwardDiff.threaded_gradient!(Uvec, dx, x, ch)
          dx
      end

  elseif b === :Zygote
      return x::AbstractVector -> Zygote.gradient(Uvec, x)[1]

  elseif b === :ReverseDiff
      return x::AbstractVector -> ReverseDiff.gradient(Uvec, x)

  elseif b === :Enzyme
      return enzyme_gradient_builder(Uvec, dim)
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