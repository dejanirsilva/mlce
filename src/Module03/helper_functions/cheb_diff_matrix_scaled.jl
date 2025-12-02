function cheb_diff_matrix(N::Integer, a::Real, b::Real)
    x, D = cheb_diff_matrix(N)
    z = (a + b)/2 .+ (b - a)/2 .* x   # map [-1,1] -> [a,b]
    Dz = (2/(b - a)) .* D
    return z, Dz
end