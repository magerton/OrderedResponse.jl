using OrderedResponse
using Optim
using StatsFuns
using Calculus
using DataFrames
using StatsModels
using StatsBase
using CategoricalArrays
using Test
using CSV
using LinearAlgebra

include("benchmarks.jl")

# read in data
# df = CSV.read(joinpath(dirname(pathof(OrderedResponse) * "/data/testdat.csv")))

df = CSV.read("D:/libraries/julia/dev/OrderedResponse" * "/data/testdat.csv")

# test functions in "likelihood" file
include("likelihood.jl")

# test outer wrapper
fm = @formula(y ~ 0 + x1 + x2)   # in StatsModels
mf = ModelFrame(fm, df)
y = OrderedResponse.response_vec(mf)
X = ModelMatrix(mf).m'

orlm(y, X, :logit)
orlm(y, X, :probit)

orlm(fm, df, :logit)
orlm(fm, df, :probit)

#
