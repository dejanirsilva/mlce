function chebyshev_derivatives(n::Int, z::Real; 
        zmin::Real = -1.0, zmax::Real = 1.0)
    a, b = 2 / (zmax - zmin), -(zmin + zmax) / (zmax - zmin)
    x = a * z + b # Map to [-1,1]
    p = ChebyshevT([zeros(n);1.0]) # Degree n Chebyshev polynomial
    d1p = derivative(p) # First derivative
    d2p = derivative(d1p) # Second derivative
    return p(x), d1p(x) * a, d2p(x) * a^2
end