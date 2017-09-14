library(MASS)

data <- read.csv("../OrderedResponse.jl/data/testdat.csv")

data$y <- factor(data$y)

l <- polr(y ~ x1 + x2, data=data, method = "logistic")
p <- polr(y ~ x1 + x2, data=data, method = "probit")

c(coefficients(l), l$zeta)
c(coefficients(p), p$zeta)
vcov(l)
vcov(p)
