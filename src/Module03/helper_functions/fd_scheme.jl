function fd_scheme(m::BlackScholesModel; forward::Bool = true)
    (; σS, r, K, T, Ns, Nt, sgrid) = m # unpack model parameters
    Δs, Δt = sgrid[2] - sgrid[1], T / (Nt - 1) # spatial/time steps
    r̅ = r - 0.5 * σS^2 # risk-adjusted drift
    # Coefficients for the tridiagonal matrix
    pu = r̅ * Δt / Δs * forward + σS^2 * Δt / (2 * Δs^2)
    ps = 1 - r̅ * Δt / Δs * (2*forward-1) - σS^2 * Δt / (Δs^2)
    pd = -r̅ * Δt / Δs * (1-forward) + σS^2 * Δt / (2 * Δs^2)
    e  = zeros(Ns)
    e[1], e[end] = pd, pu # set boundary conditions
    P = Tridiagonal(pd*ones(Ns-1), ps*ones(Ns)+e, pu*ones(Ns-1))
    # Boundary conditions
    b = zeros(Ns)
    b[end] = pu * exp(sgrid[end]) * Δs
    # Initial condition
    v = @. max(0.0, exp(sgrid) - K) # terminal condition
    for n in 2:Nt
        v = (P - r * Δt * I) * v + b # update rule
    end
    return (; P, b, v)
end