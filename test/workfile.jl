# using OrderedResponse
using StatsBase
using StatsFuns

srand(1234)

n,k = 100,2
X = randn(n, k)
β = [0.2, -1.0]
ϵ = randn(n)

ystar = X*β + ϵ

γ = [-0.4, 0.5]
y = map((yi) -> searchsortedfirst(γ,yi), ystar)

py = proportions(y)
γ0 = logit.(cumsum(py))[1:end-1]
β0 = zeros(k)
