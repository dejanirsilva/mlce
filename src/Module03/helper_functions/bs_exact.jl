function BlackScholes(m::BlackScholesModel, s::Float64)
    (; σS, r, K, T) = m
    d1 = (s - log(K) + (r + 0.5 * σS^2) * T)/(σS * sqrt(T))
    d2 = d1 - σS * sqrt(T)
    return exp(s) * cdf(Normal(), d1)  - cdf(Normal(), d2) * K * exp(-r * T)
end