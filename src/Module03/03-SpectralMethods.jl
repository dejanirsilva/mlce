using Polynomials, Plots, LaTeXStrings, Printf, Interpolations
pgfplotsx()
default(legend_font_halign=:left)

############################################################
### Include helper functions ###
############################################################

for f in ("finite_difference_fun.jl", "two_trees_struct.jl", "chebyshev_derivatives.jl", "chebyshev_solver.jl")
    include(joinpath(@__DIR__,"helper_functions/", f))
end

############################################################    
# Finite-difference approximation of the derivative
############################################################

# Function and exact derivative
f(x)      = exp(x^2) + 2 * sin(x)
fprime(x) = 2 * x * exp(x^2) + 2 * cos(x)
Δx_grid   = range(1e-2, 1e-1, length = 100)
orders    = 1:6
x0        = 0.0

# Finite-difference approximation of the derivative
orders = 1:6
Δx_grid = range(1e-17, 1e-1, length = 100)
errors = [log10(abs(finite_difference(f, x0, order, Δx = Δx) - fprime(x0))) for Δx in Δx_grid, order in orders]
p_errors = plot(log10.(Δx_grid), errors,  xlabel = L"\log_{10}(\Delta x)", line = 2,
        # linestyle = [:solid :dash :dot :dashdot :dashdotdot :dashdotdotdot :dashdotdotdotdot],
        label = permutedims(string.("Order ", orders)), legend = :bottomleft, foreground_color_legend=:transparent, background_color_legend = :transparent,
        ylabel = L"\log_{10}(|Error|)", title = "")

############################################################    
### Runge's phenomenon ###
############################################################

f(x) = 1 / (1 + 25 * x^2)
n_range = 4:8
P = []
for n in n_range
    interp_points = range(-1.0, 1.0, length = n+1)
    V = [x^k for x in interp_points, k in 0:n]
    v = f.(interp_points)
    a = V \ v
    push!(P, Polynomial(a))
end
xgrid = range(-1.0,1.0,100)
interp_points = range(-1.0, 1.0, length = 9)
p_runge = plot(xgrid, f.(xgrid), label = "True function", line = 3)
    plot!(xgrid, [p.(x) for x in xgrid, p in P], legend = :bottomright, foreground_color_legend=:transparent, background_color_legend = :transparent,
    line = 1.5, alpha = 0.5, label = permutedims(string.("Order ", n_range)), xlabel = L"x", ylabel = L"f(x)")
    scatter!(interp_points, f.(interp_points), label = "Interpolation points (N = $(n_range[end]))", color = palette(:auto)[1])

############################################################    
### Spectral differentiation error ###
############################################################

## Two-tree model
# Analytic solution
function v_analytic(m, s)
    (; ρ, σ, μ, N) = m
    return s == 0 ? 0.0 : s == 1 ? 1/ρ : 1/(2*ρ) * (1 + (1-s)/s * log(1-s) - s/(1-s) * log(s))
end

# Chebyshev solution
m = TwoTrees(N = 7)
(;v, s) = chebyshev_solver(m)

sgrid = range(0.0, 1.0, length = 100)
p_chebyshev = plot(sgrid, v.(sgrid), label = "Chebyshev", line = 2, xlabel = L"s", ylabel = L"v(s)")
plot!(sgrid, v_analytic.([m], sgrid), label = "Analytic", line = 2, linestyle = :dash, legend = :bottomright, foreground_color_legend=:transparent, background_color_legend = :transparent)
