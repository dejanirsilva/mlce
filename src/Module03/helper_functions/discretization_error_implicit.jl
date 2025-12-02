function discretization_error_implicit(m::BlackScholesModel)
    bs = BlackScholes.([m], m.sgrid)
    vi = fd_implicit(m)
    return log(sqrt(mean((vi.v - bs).^2)))
end