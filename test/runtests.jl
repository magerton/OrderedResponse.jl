using OrderedResponse
using Optim
# using StatsBase
using StatsFuns
using Calculus
using DataFrames
using Base.Test

include("benchmarks.jl")

# read in data
df = readtable(joinpath(Pkg.dir("OrderedResponse")) * "/data/testdat.csv")

y = Vector(df[:y])
X = Matrix(df[:,[:x1,:x2]])

# initial vectors
n,k = size(X)
py = proportions(y)
γ0 = norminvcdf.(cumsum(py))[1:end-1]
β0 = zeros(k)
θ0 = [β0..., γ0...]

# tmp arrays
tmpη = Array{eltype(X)}(n)
tmpgrad = Vector{Float64}(length(θ0))

# -------------- closures for testing ---------

f1(θ::Vector)                = normLL!(                                    tmpη, y, X, θ[1:2], θ[3:4])
f2(θ::Vector)                = normLLgrad!(zeros(0), zeros(0,0), zeros(0), tmpη, y, X, θ[1:2], θ[3:4])
g!(grad::Vector, θ::Vector)  = normLLgrad!(grad    , zeros(0,0), tmpgrad , tmpη, y, X, θ[1:2], θ[3:4])
h!(hess::Matrix, θ::Vector)  = normLLgrad!(zeros(4),       hess, tmpgrad , tmpη, y, X, θ[1:2], θ[3:4])

function g(θ::Vector)
    grad = similar(θ)
    g!(grad, θ)
    return grad
end

function h(θ::Vector)
    grad = similar(θ)
    hess = Matrix{eltype(grad)}(length(grad), length(grad))
    h!(hess, θ)
    return hess
end

# --------- check that LL functions are the same ---------

@test f1(θ0) ≈ f2(θ0)

optf1 = Optim.optimize(f1, θ0)
optf2 = Optim.optimize(f2, θ0)

@test optf2.minimizer ≈ optf1.minimizer
@test norm(optf1.minimizer .- θstata_norm, Inf) < 3e-6
@test norm(optf2.minimizer .- θstata_norm, Inf) < 3e-6

# --------- opt with derivatives ---------

@test Calculus.derivative(f2, θ0) ≈ g(θ0)

optg = Optim.optimize(f2, g!, θ0)
@test norm(optg.minimizer .- θstata_norm, Inf) < 3e-6
@show optg.minimizer .- θpolr_norm

# -------- newton-rhapson ------------

opth = Optim.optimize(f2, g!, h!, θ0)
vcov = h(opth.minimizer)\eye(4)

@test norm(opth.minimizer .- θstata_norm, Inf) < 3e-6
@test norm((vcov .- vcovstata_norm)[:], Inf) < 1e-8
@show opth.minimizer .- θpolr_norm
@show sqrt.(diag(vcov)) .- sqrt.(diag(vcovstata_norm))







# #
