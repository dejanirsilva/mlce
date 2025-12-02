function fd_derivative(grid, x;points = grid)
    x_interp = linear_interpolation(grid, x)
    return [Interpolations.gradient(x_interp, x)[1] for x in points]
end