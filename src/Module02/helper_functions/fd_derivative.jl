# Compute finite difference derivative
function fd_derivative(grid, x)
    x_interp = linear_interpolation(grid, x)
    return [Interpolations.gradient(x_interp, x)[1] for x in grid]
end