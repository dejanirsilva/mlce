function egm_step(m::ConsumptionSavingsDT, iter::Int, c0::Function)
    (; agrid, Z, Y, R, γ, ρ) = m # unpack model parameters
    agrid_shifted = -Y[1] * sum(R.^(-(1:iter))) .+ agrid 
    # compute the consumption policy
    c1 = [sum(exp(-ρ) * Z.P[1,j] * R * c0(R * a + Y[j]+1e-12)^(-γ) 
        for j in eachindex(Y))^(-1/γ) for a in agrid_shifted] 
    M1 = agrid_shifted .+ c1 # compute the cash-on-hand
    return (; c = linear_interpolation(M1, c1; 
        extrapolation_bc=Line()), M = M1)
end