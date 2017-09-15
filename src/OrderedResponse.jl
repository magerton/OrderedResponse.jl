module OrderedResponse

using StatsFuns
# using Optim
# using DataFrames

export orm, response_vec, LL, LL!, normLL!, normLLgrad!

# response_vec(y::PooledDataVector) = y.refs
# response_vec(mf::ModelFrame) = response_vec(mf.df[:,mf.terms.eterms[1]])

function num_categories(y::Vector{<:Integer})
    y0, L = extrema(y)
    length(unique(y)) == L - y0 + 1  || throw(error("y must be coded 1:n with all values present"))
    y0 == 1                          || throw(error("y must be coded 1:n with all values present"))
    return L
end

# function orlm(fm::Formula, data::DataFrame, θ0::Vector, LinkFun::Type)
#
#     mf = ModelFrame(fm, data)
#     mf.terms.intercept  && throw(error("Cannot have intercept. Try `y ~ 0 + x1 + x2 ...`"))
#
#     X = ModelMatrix(mf).m
#     y = response_vec(mf)
#
#     L = num_categories(y)
#     n, k = size(X)
#     T = eltype(X)
#
#     length(θ0) = k + (L-1) || throw(DimensionMismatch())
#     η  = Vector{T}(n)
#     f(θ::Vector) = LL!(η, y, X, θ[1:k], θ[k+1:end], LinkFun)
#
#     return Optim.optimize(f, θ0)
#
# end

# --------- logistic ----------

# cdf
logisticcdf(z::Real)  = logistic(z)

# pdf
function logisticpdf(z::Real)
    isfinite(z) || return zero(z)
    F = logistic(z)
    return F * (one(z) - F)
end

function dlogisticpdf(z::Real)
    isfinite(z) || return zero(z)
    F = logistic(z)
    cF = one(z) - F
    return F * cF * (one(z)-F^2)
end

# logcdf
logisticccdf(z::Real) = one(z) - logistic(z)

logisticlogcdf(z::Real)  = z - log1pexp(z)
logisticlogccdf(z::Real) =   - log1pexp(z)
#
# # dlogcdf
# dlogisticlogcdf(z::Real)  = one(z) - logistic(z)
# dlogisticlogccdf(z::Real) =        - logistic(z)
#
# # d2logcdf
# d2logisticlogcdf(z::Real)  = - logisticpdf(z)
# d2logisticlogccdf(z::Real) = - logisticpdf(z)

# --------- norml ----------

# dlogcdf
dnormpdf(z::Real) = isfinite(z) ? -z * normpdf(z) : zero(z)
dnormlogcdf(z::Real) = normpdf(z) / normcdf(z)
dnormlogccdf(z::Real) = - normpdf(z) / normccdf(z)

# d2logcdf
function d2normlogcdf(z::Real)
    ϕΦ = normpdf(z) / normcdf(z)
    return - ϕΦ * (z + ϕΦ)
end
function d2normlogccdf(z::T) where {T<:AbstractFloat}
    ϕomΦ = normpdf(z) / normccdf(z)
    return ϕomΦ * (z + ϕomΦ)
end

# --------- likelihoods ----------

for dist in (:logistic, :norm)
  llfun = Symbol("$(dist)LL")
  llfun! = Symbol("$(dist)LL!")

  lcdf  = Symbol("$(dist)logcdf")
  lccdf = Symbol("$(dist)logccdf")
  cdf   = Symbol("$(dist)cdf")
  pdf   = Symbol("$(dist)pdf")
  dpdf  = Symbol("d$(dist)pdf")

  blk = quote
      function $(llfun!)(grad::Vector, hess::Matrix, tmpgrad::Vector, η::Vector, y::Vector{<:Integer}, X::Matrix, β::Vector, γ::Vector)

          length(grad) > 0 && (grad .= 0.0)
          length(hess) > 0 && (hess .= 0.0)

          k,n = size(X)
          L = num_categories(y)

          length(β) == k    || throw(DimensionMismatch())
          length(γ) == L-1  || throw(DimensionMismatch())
          length(η) == n    || throw(DimensionMismatch())
          all(X[:,1] .== 1.0) && throw(error("X cannot have intercept"))

          any(diff(γ) .<= 0.0)  && return -Inf

          At_mul_B!(η, X, β)  # update η

          LL = 0.0
          for i in 1:n
              LL += $(llgradinner!)(grad, hess, tmpgrad, y[i], η[i], X[i,:], γ)
          end

          length(grad) > 0  && (grad .*= -1.0)
          length(hess) > 0  && (hess .*= -1.0)

          return -LL
      end
  end

  eval(blk)

  # -------------------------------------------------------------------

  if l == 1
      z = γ[1] - η
      return z - log1pexp(z)
  elseif l == length(γ)+1
      z = γ[end] - η
      return - log1pexp(z)
  else
      p2 = logistic(γ[l] - η)
      p1 = logistic(γ[l-1] - η)
      return log(p2 - p1)
  end
  logisticlogcdf(z::Real)  = z - log1pexp(z)
  logisticlogccdf(z::Real) =   - log1pexp(z)


  # simple inner LL
  blk = quote
    function $(llfun)(l::Integer, η::Real, γ::Vector)
        l == 1           && return $(lcdf)(γ[1] - η)
        l == length(γ)+1 && return $(lccdf)(γ[end] - η)
        p2 = $(cdf)(γ[l] - η)
        p1 = $(cdf)(γ[l-1] - η)
        return log(p2 - p1)
    end
  end

  eval(blk)

  # inner w/ Grad
  blk = quote
      function $(llgradinner!)(grad::Vector, hess::Matrix, tmpgrad::Vector, l::Integer, η::Real, x::AbstractVector, γ::Vector)
          L = length(γ)
          k = length(x)
          η1 = l == 1   ? -Inf : γ[l-1] - η
          η2 = l == L+1 ?  Inf : γ[l]   - η

          if l == 1
              F = $(cdf)(η2)
              LL = $(ldcf)(η2)
          elseif l == L+1
              F = $(ccdf)(η1)
          F = $(cdf)(η2) - $(cdf)(η1)
          LL = log(F)

          if length(grad) > 0
              dlogF1 = $(pdf)(η1) / F
              dlogF2 = $(pdf)(η2) / F
              dlogF  = dlogF2 - dlogF1

              tmpgrad[1:k] .= - dlogF .* x
              tmpgrad[k+1:end] .= 0.0
              l > 1    && (tmpgrad[k+l-1] -= dlogF1)
              l < L+1  && (tmpgrad[k+l]   += dlogF2)

              grad .+= tmpgrad
          end

          if length(hess) > 0
              d2logF1 = $(dpdf)(η1) / F
              d2logF2 = $(dpdf)(η2) / F
              d2logF  = d2logF2 - d2logF1

              Base.LinAlg.BLAS.ger!(-1.0, tmpgrad, tmpgrad, hess)
              Base.LinAlg.BLAS.ger!(d2logF, x, x, @view(hess[1:k,1:k]))
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
  end

  eval(blk)

end






# end module
end
