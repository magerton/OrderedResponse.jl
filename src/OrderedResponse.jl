__precompile__()

module OrderedResponse

using StatsFuns
using DataFrames
using StatsModels
using Optim
using LinearAlgebra
using StatsBase
using NLSolversBase
using Distributions: _F1

function num_categories(y::AbstractVector{<:Integer})
    y0, L = extrema(y)
    length(unique(y)) == L - y0 + 1  || throw(error("y must be coded 1:n with all values present"))
    y0 == 1                          || throw(error("y must be coded 1:n with all values present"))
    return L
end

include("distributions.jl")
include("outer-wrapper.jl")
include("likelihood.jl")

# end module
end
