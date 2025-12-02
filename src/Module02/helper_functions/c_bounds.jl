# Compute bounds for consumption
function c_bounds(m, Mgrid)
    (; Y, R, γ, ρ) = m
    min = 1/(1 + R^(1/γ-1)*exp(-ρ/γ)) * (Mgrid .+ Y[1]/R) 
    max = 1/(1 + R^(1/γ-1)*exp(-ρ/γ)) * (Mgrid .+ sum(Y .* m.Z.P[1,:])/R) 
    return (; min, max)
end