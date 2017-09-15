export orlm

response_vec(y::PooledDataVector) = y.refs
response_vec(y::DataVector{<:Integer}) = Vector(y)
response_vec(mf::ModelFrame) = response_vec(mf.df[:,mf.terms.eterms[1]])

"""
    Estimate ordered response of variety `model ∈ (:logit, :probit)` with specification `fm` and `data`. Optim options passed with `kwargs`
"""
function orlm(fm::Formula, data::DataFrame, model::Symbol; method=Optim.Newton(), kwargs...)

    model ∈ (:probit, :logit) || throw(error("Can only do :probit or :logit model"))

    mf = ModelFrame(fm, data)
    mf.terms.intercept  && throw(error("Cannot have intercept. Try `y ~ 0 + x1 + x2 ...`"))

    X = ModelMatrix(mf).m'
    y = response_vec(mf)

    L = num_categories(y)  # also checks that categories are 1:L
    k, n = size(X)
    T = eltype(X)

    KLm1 = k+L-1
    θ0 = zeros(T, KLm1)
    tmpgrad = similar(θ0)
    η = Vector{T}(n)

    # starting values for intercept
    cpy = cumsum(proportions(y))
    θ0[k+1:end] = invcdf.(cpy[1:end-1], Val{model})

    # closures for optim
    td = Optim.TwiceDifferentiable(
        (θ::Vector)               -> LL!(zeros(T,0)   , zeros(T,0,0), zeros(0), η, y, X, θ[1:k], θ[k+1:end], Val{model}, -1.0),
        (grad::Vector, θ::Vector) -> LL!(grad         , zeros(T,0,0), tmpgrad , η, y, X, θ[1:k], θ[k+1:end], Val{model}, -1.0),
        (grad::Vector, θ::Vector) -> LL!(grad         , zeros(T,0,0), tmpgrad , η, y, X, θ[1:k], θ[k+1:end], Val{model}, -1.0),
        (hess::Matrix, θ::Vector) -> LL!(zeros(T,KLm1),         hess, tmpgrad , η, y, X, θ[1:k], θ[k+1:end], Val{model}, -1.0)
    )

    return Optim.optimize(td, θ0, method, Optim.Options(;kwargs...))

end
