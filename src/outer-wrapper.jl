export orlm

response_vec(y::CategoricalArray) = y.refs
response_vec(y::Vector{Union{Missing,T}}) where {T<:Integer}= Vector{T}(y)
response_vec(y::Vector{T}) where {T<:Integer} = Vector{Int}(y)
response_vec(mf::ModelFrame) = response_vec(mf.df[!,mf.terms.eterms[1]])

function γinit(y::Vector, model::Symbol)
    cpy = cumsum(proportions(y))
    return invcdf.(cpy[1:end-1], Val{model})
end


"""
    orlm(fm::Formula, data::DataFrame, model::Symbol; method=Optim.Newton(), kwargs...)

Estimate ordered response model of variety `model ∈ (:logit, :probit)` with specification
`fm` and `data`. Optim options passed with `kwargs`. Set the optimizer with `method`.
Returns a tuple with negative log likelihood, gradient, hessian, and optimization results
`(LL::Real, grad::Vector, hess::Matrix, opt::Optim.MultivariateOptimizationResults)`
"""
function orlm(fm::Formula, data::DataFrame, model::Symbol; kwargs...)

    model ∈ (:probit, :logit) || throw(error("Can only do :probit or :logit model"))

    mf = ModelFrame(fm, data)
    mf.terms.intercept  && throw(error("Cannot have intercept. Try `y ~ 0 + x1 + x2 ...`"))

    X = ModelMatrix(mf).m'
    y = response_vec(mf)

    return orlm(y, X, model; kwargs...)

end

orlm(y::Vector{<:Integer}, X::AbstractMatrix{T}, model::Symbol; kwargs...) where {T} = orlm(y, Matrix(X), model; kwargs...)


function orlm(y::Vector{<:Integer}, X::Matrix{T}, model::Symbol; method=Optim.Newton(), kwargs...) where {T}
    model ∈ (:probit, :logit) || throw(error("Can only do :probit or :logit model"))

    # check dims
    k, n = size(X)
    length(y) == n || throw(DimensionMismatch("length(y) != size(X,2)"))

    L = num_categories(y)  # also checks that categories are 1:L
    KLm1 = k+L-1
    θ0 = zeros(T, KLm1)
    θ0[k+1:end] = γinit(y, model) # starting values
    tmpgrad = similar(θ0)
    η = Vector{T}(undef,n)

    # closures for optim
    td = NLSolversBase.TwiceDifferentiable(
        (θ::Vector)               -> orLL!(zeros(T,0)   , zeros(T,0,0), zeros(0), η, y, X, θ[1:k], θ[k+1:end], Val{model}, -1.0),
        (grad::Vector, θ::Vector) -> orLL!(grad         , zeros(T,0,0), tmpgrad , η, y, X, θ[1:k], θ[k+1:end], Val{model}, -1.0),
        (grad::Vector, θ::Vector) -> orLL!(grad         , zeros(T,0,0), tmpgrad , η, y, X, θ[1:k], θ[k+1:end], Val{model}, -1.0),
        (hess::Matrix, θ::Vector) -> orLL!(zeros(T,KLm1),         hess, tmpgrad , η, y, X, θ[1:k], θ[k+1:end], Val{model}, -1.0),
        θ0
    )

    opt =  Optim.optimize(td, θ0, method, Optim.Options(;kwargs...))

    # final eval
    hess = zeros(T, KLm1, KLm1)
    grad = similar(tmpgrad)
    θfinal = opt.minimizer
    LL = orLL!(grad, hess, tmpgrad, η, y, X, θfinal[1:k], θfinal[k+1:end], Val{model}, -1.0)

    return (opt.minimizer, LL, grad, hess, opt)

end
