function discretization_error(m::BlackScholesModel)
    bs = BlackScholes.([m], m.sgrid)
    forward = m.r > 0.5 * m.σS^2 ? true : false
    vf = fd_scheme(m, forward = forward)
    return log(sqrt(mean((vf.v - bs).^2)))
end
