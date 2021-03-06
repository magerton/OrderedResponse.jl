export orLLi!

# --------- likelihoods ----------

"likelihood of data"
function orLL!(grad::AbstractVector{T}, hess::AbstractMatrix{T}, tmpgrad::AbstractVector, η::AbstractVector, y::AbstractVector{<:Integer}, X::AbstractMatrix, β::AbstractVector, γ::AbstractVector, model::Type{Val{D}}, sgn::Real=-one(T)) where {D,T}

  # # ---------- SAFETY CHECKS ---------------
  D ∈ (:probit, :logit) || throw(error("Can only do :probit or :logit model"))
  k,n = size(X)
  L = num_categories(y)
  length(β) == k    || throw(DimensionMismatch())
  length(γ) == L-1  || throw(DimensionMismatch())
  length(η) == n    || throw(DimensionMismatch())
  all(X[:,1] .== 1.0) && throw(error("X cannot have intercept"))
  # ----------------------------------------

  # ensure intercepts don't cross
  any(diff(γ) .<= 0.0)  &&  return -Inf

  # reset grad + hess
  length(grad) > 0 && (grad .= 0.0)
  length(hess) > 0 && (hess .= 0.0)

  mul!(η, transpose(X), β)  # update η

  LL = zero(T)
  for j in Base.OneTo(length(y))
      LL += orLLi!(grad, hess, tmpgrad, y[j], η[j], @view(X[:,j]), γ, model)
  end

  # flip signs of these
  if sign(sgn) < zero(sgn)
      length(grad) > 0  && (grad .*= -one(T))
      length(hess) > 0  && (hess .*= -one(T))
      return -LL
  end

  return LL
end


"observation-specific likelihood"
function orLLi!(grad::AbstractVector{T}, hess::AbstractMatrix{T}, tmpgrad::AbstractVector, l::Integer, η::Real, x::AbstractVector, γ::AbstractVector{T}, model::Type) where {T}
  any(diff(γ) .< zero(T))  &&  return -Inf
  L = length(γ)
  k = length(x)
  η1 = l == 1   ? -Inf : γ[l-1] - η
  η2 = l == L+1 ?  Inf : γ[l]   - η

  # LL takes a short-cut to improve accuracy if l == 1 or L+1
  if l == 1
      F = cdf(η2, model)
      LL = logcdf(η2, model)
  elseif l == L+1
      F = ccdf(η1, model)
      LL = logccdf(η1, model)
  else
      F = cdf(η2, model) - cdf(η1, model)
      LL = log(F)
  end

# https://github.com/scipy/scipy/blob/a2ffe09aa751749f2372aa13c19c61b2dec5266f/scipy/stats/_continuous_distns.py
# https://github.com/JuliaStats/Distributions.jl/blob/master/src/truncated/normal.jl
# https://github.com/cossio/TruncatedNormal.jl/blob/master/notes/normal.pdf

  if length(grad) > 0
      dlogF1 = pdf(η1, model) / F
      dlogF2 = pdf(η2, model) / F
      dlogF  = dlogcdf_trunc(η1,η2,model) # (pdf(η2, model) - pdf(η1, model)) / F  # try using _F1 from truncated/normal.jl

      # so that we can update grad directly if no hessian
      if length(hess) == 0
          tmpgrad2 = grad
      else
          tmpgrad2 = tmpgrad
          tmpgrad .= 0.0
      end

      tmpgrad2[1:k] .-= dlogF .* x
      l > 1    && (tmpgrad2[k+l-1] -= dlogF1)
      l < L+1  && (tmpgrad2[k+l]   += dlogF2)

      # update actual gradient vector if computing hessian
      length(hess) > 0 && (grad .+= tmpgrad)
  end

  if length(hess) > 0
      d2logF1 = dpdf(η1, model) / F
      d2logF2 = dpdf(η2, model) / F
      d2logF  = d2logF2 - d2logF1

      LinearAlgebra.BLAS.ger!(-one(T), tmpgrad, tmpgrad, hess)
      LinearAlgebra.BLAS.ger!(d2logF, x, x, @view(hess[1:k,1:k]))
      if l > 1
          hess[k+l-1, 1:k  ] .+= d2logF1 * x
          hess[1:k  , k+l-1] .+= d2logF1 * x
          hess[k+l-1, k+l-1]  -= d2logF1
      end
      if l < L+1
          hess[k+l, 1:k] .-= d2logF2 * x
          hess[1:k, k+l] .-= d2logF2 * x
          hess[k+l, k+l]  += d2logF2
      end
  end

  return LL
end
