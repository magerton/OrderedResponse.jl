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
using StatsFuns
using Distributions: _F1

include("benchmarks.jl")

# read in data
# df = CSV.read(joinpath(dirname(pathof(OrderedResponse) * "/data/testdat.csv")))

@testset "test dlogF for :probit" begin
    for model in (:probit,:logit,)
        for a in [-6.0:5.0..., typemin(1.0),]
            for b in [(max(a,-5)+1):6.0..., typemax(1.0),]
                FF = OrderedResponse.cdf(b, Val{model}) - OrderedResponse.cdf(a, Val{model})
                ff = OrderedResponse.pdf(b, Val{model}) - OrderedResponse.pdf(a, Val{model})
                dlogF  = ff / FF
                dlogF_new = OrderedResponse.dlogcdf_trunc(a,b,Val{model})
                @test dlogF ≈ dlogF_new
                @test dlogF_new != 0.0 || ff == 0.0
                @test FF > 0
            end
        end
    end
end

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
