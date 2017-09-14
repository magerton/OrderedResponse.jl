module OrderedResponse

using StatsFuns
using Optim
using DataFrames

export orm, response_vec, LL, LL!

response_vec(y::PooledDataVector) = y.refs
response_vec(mf::ModelFrame) = response_vec(mf.df[:,mf.terms.eterms[1]])

function orm(fm::Formula, data::DataFrame, θ0::Vector, LinkFun::Type)

    mf = ModelFrame(fm, data)
    mf.terms.intercept  && throw(error("Cannot have intercept. Try `y ~ 0 + x1 + x2 ...`"))

    X = ModelMatrix(mf).m
    y = response_vec(mf)

    L = num_categories(y)
    n, k = size(X)
    T = eltype(X)

    length(θ0) = k + (L-1) || throw(DimensionMismatch())
    η  = Vector{T}(n)
    f(θ::Vector) = LL!(η, y, X, θ[1:k], θ[k+1:end], LinkFun)

    return Optim.optimize(f, θ0)

end


logisticlogcdf(z::T)  where {T<:AbstractFloat} = z - log1pexp(z)
logisticlogccdf(z::T) where {T<:AbstractFloat} =   - log1pexp(z)
logisticpdf(z::T)     where {T<:AbstractFloat} = logistic(z)
logisticdpdf(z::T)    where {T<:AbstractFloat} = logistic(z) * logistic(one(T)-z)

function LL(l::Integer, η::T, wt::T, γ::Vector{T}, ::Type{Val{:logit}}) where {T<:AbstractFloat}
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
end

function LL(l::Integer, η::T, wt::T, γ::Vector{T}, ::Type{Val{:probit}}) where {T<:AbstractFloat}
    l == 1              && return normlogcdf(γ[1] - η)
    l == length(γ) + 1  && return normlogccdf(γ[end] - η)
    return                        log(normcdf(γ[l] - η) - normcdf(γ[l-1]  - η))
end


function num_categories(y::Vector{<:Integer})
    y0, L = extrema(y)
    length(unique(y)) == L - y0 + 1  || throw(error("y must be coded 1:n with all values present"))
    y0 == 1                          || throw(error("y must be coded 1:n with all values present"))
    return L
end



function LL!(η::Vector{T}, y::Vector{<:Integer}, X::Matrix{T}, β::Vector{T}, γ::Vector{T}, LinkFun::Type{Val{D}}) where {T<:AbstractFloat,D}

    D ∈ (:logit, :probit)  || throw(error("must have :probit or :logit"))
    n,k = size(X)
    L = num_categories(y)

    length(β) == k    || throw(DimensionMismatch())
    length(γ) == L-1  || throw(DimensionMismatch())
    length(η) == n    || throw(DimensionMismatch())
    all(X[:,1] .== 1.0) && throw(error("X cannot have intercept"))
    all(diff(γ) .> 0) || return -Inf

    A_mul_B!(η, X, β)  # update η
    return - sum(LL(x..., γ, LinkFun) for x in zip(y, η))
end



# end module
end
