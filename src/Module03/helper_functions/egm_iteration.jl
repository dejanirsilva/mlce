# Endogenous gridpoint method iteration
m = ConsumptionSavingsDT(N = 100, α = 1.5)
policies = [egm_step(m, 1, M->M)]
for i = 2:8
    push!(policies, egm_step(m, i, M->policies[i-1][1](M)))
end