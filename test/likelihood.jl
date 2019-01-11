y = Vector{Int}(df[:y])
X = Matrix(Matrix{Float64}(df[:,[:x1,:x2]])')

# initial vectors
k,n = size(X)
py = proportions(y)
γ0 = norminvcdf.(cumsum(py))[1:end-1]
β0 = zeros(k)
θ0 = [β0..., γ0...]

# tmp arrays
tmpη = Array{eltype(X)}(undef,n)
tmpgrad = Vector{Float64}(undef,length(θ0))

# --------- check that LL functions are the same ---------

for model in (:logit,:probit,)

    @show model

    @testset "testing cdf/pdf derivatives for $model" begin
        for z in -10.:10.
            fd = Calculus.gradient((x) -> OrderedResponse.cdf(x, Val{model}), z)
            ad = OrderedResponse.pdf(z, Val{model})

            fh = Calculus.hessian( (x) -> OrderedResponse.cdf(x, Val{model}), z)
            ah = OrderedResponse.dpdf(z, Val{model})

            @test isapprox(fd, ad, atol=1e-9, rtol=1e-8)
            @test isapprox(fh, ah, atol=3.79e-7, rtol=1e-5)
            # @test fd ≈ ad
            # @test fh ≈ ah
            @test abs(fd - ad) < 1e-5
            @test abs(fh - ah) < 1e-5
        end
    end

    # ------------- closures ---------------

    # function LL!(grad::Vector{T}, hess::Matrix{T}, tmpgrad::Vector, η::Vector, y::Vector{<:Integer}, X::Matrix, β::Vector, γ::Vector, model::Type{Val{D}}, sgn::Real=-one(T)) where {D,T}

    f(θ::Vector)                = OrderedResponse.orLL!(zeros(0), zeros(0,0), zeros(0), tmpη, y, X, θ[1:2], θ[3:4], Val{model}, -1.0)
    g!(grad::Vector, θ::Vector) = OrderedResponse.orLL!(grad    , zeros(0,0), tmpgrad , tmpη, y, X, θ[1:2], θ[3:4], Val{model}, -1.0)
    h!(hess::Matrix, θ::Vector) = OrderedResponse.orLL!(zeros(4),       hess, tmpgrad , tmpη, y, X, θ[1:2], θ[3:4], Val{model}, -1.0)

    function g(θ::Vector)
        grad = similar(θ)
        g!(grad, θ)
        return grad
    end

    function h(θ::Vector)
        grad = similar(θ)
        hess = Matrix{eltype(grad)}(undef,length(grad), length(grad))
        h!(hess, θ)
        return hess
    end

    # --------- derivative -free search ---------

    optf = Optim.optimize(f, θ0)
    @show optf
    @test_broken optf.minimizer ≈ θpolr(model)
    @test norm(optf.minimizer .- θpolr(model), Inf) < 3e-5

    # --------- opt with derivatives ---------

    @test Calculus.derivative(f, θ0) ≈ g(θ0)
    @test_broken Calculus.derivative(f, optf.minimizer) ≈ g(optf.minimizer)
    @test norm(Calculus.derivative(f, optf.minimizer) .- g(optf.minimizer), Inf) < 1e-5

    optg = Optim.optimize(f, g!, θ0)
    @show optg
    # @test_broken optf.minimizer ≈ optg.minimizer
    # @test_broken optg.minimizer ≈ θpolr(model)
    @test norm(optf.minimizer .- optg.minimizer, Inf) < 1e-4
    @test norm(optg.minimizer .- θpolr(model), Inf) < 1e-7
    @show optg.minimizer .- θpolr(model)

    # -------- newton-rhapson ------------

    maxhessdiff = maximum(abs.(Calculus.hessian(f, θ0) .- h(θ0)))
    println("\nMaximum FD for Hessian in $model model is $maxhessdiff\n")
    @test maxhessdiff < 0.2

    opth = Optim.optimize(f, g!, h!, θ0)
    vcov = h(opth.minimizer)\Matrix(I,4,4)

    @test norm(opth.minimizer .- θpolr(model), Inf) < 1e-7
    @test norm((vcov .- vcovpolr(model))[:], Inf) < 2.7e-4
    # @test_broken opth.minimizer ≈ θpolr(model)
    @test_broken vcov ≈ vcovpolr(model)

    @show opth.minimizer .- θpolr(model)
    @show sqrt.(diag(vcov)) .- sqrt.(diag(vcovstata(model)))

end
