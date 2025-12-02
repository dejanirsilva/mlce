function vf_iteration(m::ConsumptionSavingsDT, V0::Function; 
    NM::Int64 = 11, Nc::Int64 = 101)
    (; Y, Z, R, γ, ρ) = m # unpack model parameters
    Mgrid = collect(range(-Y[1]/R, 3.0, length=NM)) # grid for M
    cgrid = [range(0.0, m + Y[1]/R, length=Nc) 
        for m in Mgrid] # collection of grids for c
    # Action-value function
    V1(M, c) = c^(1-γ)/(1-γ) + exp(-ρ) * sum(Z.P[1,j] * 
            V0(m,R * (M-c) + Y[j]+1e-12) for j in eachindex(Y))
    # Policy and value functions
    C = [cgrid[j][argmax([V1(Mgrid[j], c) for c in cgrid[j]])] 
        for j in eachindex(Mgrid)] 
    V = [V1(Mgrid[j], C[j]) for j in eachindex(Mgrid)]
    return (; C, V, Mgrid) # return a named tuple
end