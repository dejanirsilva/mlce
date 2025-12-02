function chebyshev_solver(m::TwoTrees)
    (; ρ, σ, N) = m
    # Chebyshev grid and mapping to [0,1]
    x = reverse(cos.(pi .* (0:N) ./ N))
    s = (x .+ 1) ./ 2
    # Assemble the linear operator
    L, b = zeros(N+1, N+1), copy(s)
    for i in 1:N+1, j in 1:N+1
        T̃, dT, d2T = chebyshev_derivatives(j-1, s[i]; zmin = 0)
        if i == 1 || i == N+1
            L[i,j] = T̃ # Boundary points
        else
            μs = -2 * σ^2 * s[i] * (1 - s[i]) * (s[i] - 1/2)
            σs = sqrt(2) * σ * s[i] * (1 - s[i])
            L[i,j] = ρ * T̃ - dT * μs - d2T * σs^2 / 2
        end
    end
    b[end] = 1/ρ # Boundary condition at s = 1
    a = L \ b  # Solve for the coefficients
    return (; v = z -> ChebyshevT(a)(2 * z - 1), s = s) 
end