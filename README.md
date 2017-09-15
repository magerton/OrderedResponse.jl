# OrderedResponse

[![Build Status](https://travis-ci.org/magerton/OrderedResponse.jl.svg?branch=master)](https://travis-ci.org/magerton/OrderedResponse.jl)

[![Coverage Status](https://coveralls.io/repos/magerton/OrderedResponse.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/magerton/OrderedResponse.jl?branch=master)

[![codecov.io](http://codecov.io/github/magerton/OrderedResponse.jl/coverage.svg?branch=master)](http://codecov.io/github/magerton/OrderedResponse.jl?branch=master)

Estimates ordered logit & probit models. Returns a tuple with the negative log likelihood, gradient, hessian, and the `Optim.MultivariateOptimizationResults` object.

# Example
```julia
using OrderedResponse
using DataFrames
using Optim

# ------- make some data ----------

srand(1234)

n,k = 1000,2
X = randn(n, k)
ϵ = randn(n)

β = [0.2, -1.0]
γ = [-0.4, 0.5]

ystar = X*β + ϵ
y = map((yi) -> searchsortedfirst(γ,yi), ystar)

df = convert(DataFrame, X)
df[:y] = y

# ------- run model ----------
@show orlm(@formula(y ~ 0 + x1 + x2), df, :logit)
@show orlm(@formula(y ~ 0 + x1 + x2), df, :probit)
```
