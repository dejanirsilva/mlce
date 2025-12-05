function egm_iteration(m::ConsumptionSavingsDT, c0::Function; max_iter::Int = 100, tol::Float64 = 1e-6)
    policies = [egm_step(m, 1, c0)]
    for i = 2:max_iter
        c_prev = policies[i-1][1]
        policy = egm_step(m, i, c_prev)
        push!(policies, policy)
        # Check for convergence (simple check on policy function)
        if i > 1 && maximum(abs.(policy[1].(m.Mgrid) .- c_prev.(m.Mgrid))) < tol
            break
        end
    end
    return policies
end

