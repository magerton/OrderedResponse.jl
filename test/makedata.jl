using DataFrames

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

writetable(joinpath(Pkg.dir("OrderedResponse")) * "/data/testdat.csv", df)
