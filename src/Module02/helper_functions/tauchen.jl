"""
Tauchen (1986) discretization of the AR(1) process
    z_{t+1} = μ + ρ z_t + ε_{t+1},   ε ~ N(0, σϵ^2).
"""
function tauchen(M::Int, ρ::Real, σϵ::Real; μ::Real=0.0, m::Real=3.0)
    @assert M ≥ 2 "Need at least M=2 grid points."
    @assert abs(ρ) < 1 "Require |ρ|<1 for stationary AR(1)."
    σz = σϵ / sqrt(1 - ρ^2)                # unconditional std. dev.
    z̄ = μ / (1 - ρ)                        # unconditional mean
    if σϵ == 0.0 return (; z = [z̄], P = [1.0]) end # degenerate case
    zmin, zmax = z̄ - m*σz, z̄ + m*σz
    Δ = (zmax - zmin) / (M - 1)
    z = collect(range(zmin, zmax, length=M))
    P = zeros(Float64, M, M)
    for i in 1:M
        mean_next = μ + ρ*z[i]
        dist = Normal(mean_next, σϵ)
        # First bin: (−∞, midpoint_1]
        P[i, 1] = cdf(dist, z[1] + Δ/2)
        # Interior bins: (midpoint_{j-1}, midpoint_j]
        for j in 2:M-1
            upper = z[j] + Δ/2
            lower = z[j] - Δ/2
            P[i, j] = cdf(dist, upper) - cdf(dist, lower)
        end
        # Last bin: (midpoint_{N-1}, +∞)
        P[i, M] = 1 - cdf(dist, z[M] - Δ/2)
    end
    return (;z, P)
end