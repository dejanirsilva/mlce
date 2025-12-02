function make_grid(zmin, zmax, Nz; α = 1.0)
    u = range(0.0, 1.0, length=Nz)
    double_exp = α == 0 ? u : @. (exp(exp(α * u) - 1.0) - 1.0) / (exp(exp(α) - 1.0) - 1.0)
    return @. zmin + (zmax - zmin) * double_exp
end