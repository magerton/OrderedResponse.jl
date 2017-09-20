# ----------------- logit -----------------

# cdf
cdf(   z::Real, ::Type{Val{:logit}}) = logistic(z)
ccdf(  z::Real, ::Type{Val{:logit}}) = one(z) - logistic(z)
invcdf(z::Real, ::Type{Val{:logit}}) = logit(z)

# pdf
function pdf(z::Real, ::Type{Val{:logit}})
    isfinite(z) || return zero(z)
    F = logistic(z)
    return F * (one(z) - F)
end

function dpdf(z::Real, ::Type{Val{:logit}})
    isfinite(z) || return zero(z)
    F = logistic(z)
    cF = one(z) - F
    return F * cF * (one(z)-2*F)
end

# logcdf
logcdf(   z::Real, ::Type{Val{:logit}}) = z - log1pexp(z)
logccdf(  z::Real, ::Type{Val{:logit}}) =   - log1pexp(z)
dlogcdf(  z::Real, ::Type{Val{:logit}}) = one(z) - logistic(z)
dlogccdf( z::Real, ::Type{Val{:logit}}) =         - logistic(z)
d2logcdf( z::Real, ::Type{Val{:logit}}) = - pdf(z, Val{:logit})
d2logccdf(z::Real, ::Type{Val{:logit}}) = - pdf(z, Val{:logit})


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

# d2logcdf
function d2logcdf(z::Real, ::Type{Val{:probit}})
    ϕΦ = normpdf(z) / normcdf(z)
    return - ϕΦ * (z + ϕΦ)
end
function d2logccdf(z::Real, ::Type{Val{:probit}})
    ϕomΦ = normpdf(z) / normccdf(z)
    return ϕomΦ * (z + ϕomΦ)
end
