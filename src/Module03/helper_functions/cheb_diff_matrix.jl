function cheb_diff_matrix(N::Int64)
    x = cos.(pi .* (0:N) ./ N) # Chebyshev-Lobatto nodes
    c = [2; ones(N-1); 2] .* (-1).^(0:N) # c_j weights
    D = zeros(N+1, N+1) # Chebyshev differentiation matrix
    for i in 1:N+1, j in 1:N+1
        if i != j
            D[i,j] = (c[i] / c[j]) / (x[i] - x[j]) # off-diagonals
        end
    end
    D[1,1], D[end,end] =  (2N^2 + 1) / 6, -(2N^2 + 1) / 6
    for j in 2:N
        D[j,j] = -x[j] / (2*(1 - x[j]^2)) # diagonals
    end
    return x, D
end