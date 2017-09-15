library(MASS)

data <- read.csv("E:/projects/OrderedResponse.jl/data/testdat.csv")

data$y <- factor(data$y)

l <- polr(y ~ x1 + x2, data=data, method = "logistic")
p <- polr(y ~ x1 + x2, data=data, method = "probit")

cat(paste(sprintf("%a", c(coefficients(l), l$zeta)) , collapse = ", "))
cat(paste(sprintf("%a", c(coefficients(p), p$zeta)) , collapse = ", "))
cat(paste(sprintf("%a", vcov(l)                   ) , collapse = ", "))
cat(paste(sprintf("%a", vcov(p)                   ) , collapse = ", "))
