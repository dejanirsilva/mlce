function fd_implicit(m::IncomeFluctuationsModel; Δt::Float64 = Inf, 
    tol::Float64 = 1e-6, max_iter::Int64 = 100, print_residual::Bool = true)
    (; ρ, r, γ, Y, λ, Wgrid, Wmin, Wmax, N) = m # unpack parameters
    ΔW = Wgrid[2] - Wgrid[1]
    # Initial guess
    v = [1/ρ * (y + r * w)^(1-γ) / (1-γ) for w in Wgrid, y in Y] 
    c, vW, residual = similar(v), similar(v), 0.0 # pre-allocation
    for i = 1:max_iter
        # Compute derivatives
        Dv = (v[2:end,:] - v[1:end-1,:]) / ΔW
        vB = [(@. (r * Wmin + Y)^(-γ))'; Dv] # backward difference
        vF = [Dv; (@. (r * Wmax + Y)^(-γ))'] # forward difference
        v̅  = (r * Wgrid .+ Y').^(-γ) # zero-savings case
        μB = r * Wgrid .+ Y' - vB.^(-1/γ) # backward drift
        μF = r * Wgrid .+ Y' - vF.^(-1/γ) # forward drift
        vW = ifelse.(μF .> 0.0, vF, ifelse.(μB .< 0.0, vB, v̅))
        # Assemble matrix
        c  = vW.^(-1/γ)          # consumption
        u  = c.^(1-γ) / (1-γ)    # utility
        L = -min.(μB, 0) / ΔW    # subdiagonal
        R = max.(μF, 0) / ΔW     # superdiagonal
        S = @. 1/Δt + L + R + ρ + λ' # diagonal
        Aj = [Tridiagonal(-L[2:end,j], S[:,j], -R[1:end-1,j]) 
                for j in eachindex(Y)] # tridiagonal matrices
        A = [sparse(Aj[1]) -λ[1] * I
            -λ[2] * I sparse(Aj[2])] # block matrix
        # Update
        vp = A \ (u + v/Δt)[:]
        residual = sqrt(mean((vp - v[:]).^2)) 
        v = reshape(vp, N, length(Y)) # reshape vector to matrix
        if print_residual println("Iteration $i, residual = $residual") end
        if residual < tol
            break
        end
    end
    s = r * Wgrid .+ Y' - c   # savings
    return (; v, vW, c, s, residual) # return solution
end