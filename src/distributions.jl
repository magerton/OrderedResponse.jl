# ----------------- logit -----------------

# pdf
function logisticpdf(z::Real)
    isfinite(z) || return zero(z)
    lz = logistic(-abs(z))
    return lz * (1 - lz)
end

function logisticcdf(z::Real)
    isfinite(z)     && return logistic(z)
    z == typemin(z) && return zero(z)
    z == typemax(z) && return one(z)
    throw(DomainError())
end


# cdf
cdf(   z::Real, ::Type{Val{:logit}}) = logisticcdf(z)
ccdf(  z::Real, ::Type{Val{:logit}}) = logisticcdf(-z)
invcdf(z::Real, ::Type{Val{:logit}}) = logit(z)

@inline pdf(z::Real, ::Type{Val{:logit}}) = logisticpdf(z)

function dpdf(z::Real, ::Type{Val{:logit}})
    isfinite(z) || return zero(z)
    F = logistic(z)
    cF = one(z) - F
    return F * cF * (one(z)-2*F)
end

# logcdf
logcdf(   z::Real, ::Type{Val{:logit}}) = -log1pexp(-z)  # z - log1pexp(z)
logccdf(  z::Real, ::Type{Val{:logit}}) = -log1pexp( z)  #   - log1pexp(z)
dlogcdf(  z::Real, ::Type{Val{:logit}}) = one(z) - logistic(z)
dlogccdf( z::Real, ::Type{Val{:logit}}) =        - logistic(z)
d2logcdf( z::Real, ::Type{Val{:logit}}) = - pdf(z, Val{:logit})
d2logccdf(z::Real, ::Type{Val{:logit}}) = - pdf(z, Val{:logit})

dlogcdf_trunc(a::Real, b::Real, ::Type{Val{:logit}}) = (logisticpdf(b) - logisticpdf(a)) / (logisticcdf(b) - logisticcdf(a))

# ----------------- probit -----------------

# dlogcdf
cdf(     z::Real, ::Type{Val{:probit}}) = normcdf(z)
ccdf(    z::Real, ::Type{Val{:probit}}) = normccdf(z)
invcdf(  z::Real, ::Type{Val{:probit}}) = norminvcdf(z)
pdf(     z::Real, ::Type{Val{:probit}}) = isfinite(z) ?      normpdf(z) : zero(z)
dpdf(    z::Real, ::Type{Val{:probit}}) = isfinite(z) ? -z * normpdf(z) : zero(z)

logcdf(  z::Real, ::Type{Val{:probit}}) = normlogcdf(z)
logccdf( z::Real, ::Type{Val{:probit}}) = normlogccdf(z)
dlogcdf( z::Real, ::Type{Val{:probit}}) =   normpdf(z) / normcdf(z)
dlogccdf(z::Real, ::Type{Val{:probit}}) = - normpdf(z) / normccdf(z)

function dlogcdf_trunc(a::Real, b::Real, ::Type{Val{:probit}})
    a == typemin(a) && return   normpdf(b) / normcdf(b)
    b == typemax(b) && return - normpdf(a) / normccdf(a)
    return - sqrt(2/π) * _F1(a/sqrt2, b/sqrt2)
end

# d2logcdf
function d2logcdf(z::Real, ::Type{Val{:probit}})
    ϕΦ = normpdf(z) / normcdf(z)
    return - ϕΦ * (z + ϕΦ)
end
function d2logccdf(z::Real, ::Type{Val{:probit}})
    ϕomΦ = normpdf(z) / normccdf(z)
    return ϕomΦ * (z + ϕomΦ)
end
