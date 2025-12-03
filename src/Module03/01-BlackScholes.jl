using Interpolations, Plots, LaTeXStrings, Distributions
using LinearAlgebra, SparseArrays, Statistics
pgfplotsx()
default(legend_font_halign=:left)

############################################################
### Include helper functions ###
############################################################

for f in ("bs_struct.jl", "fd_scheme.jl", "bs_exact.jl", "discretization_error.jl", "discretization_error_implicit.jl", "fd_implicit.jl")
    include(joinpath(@__DIR__,"helper_functions/", f))
end

############################################################
### Example: Black-Scholes Option Pricing ###
############################################################

m1 = BlackScholesModel(Nt = 300, Ns = 150, T = 3.0, r = +0.03, σS = 0.01)
m2 = BlackScholesModel(Nt = 300, Ns = 150, T = 3.0, r = -0.03, σS = 0.01)

vf1, vb1 = fd_scheme(m1, forward = true), fd_scheme(m1, forward = false)
vf2, vb2 = fd_scheme(m2, forward = true), fd_scheme(m2, forward = false)
bs1, bs2 = BlackScholes.([m1], m1.sgrid), BlackScholes.([m2], m2.sgrid)

# Plots: Forward difference
p_vf1 = plot(m1.sgrid, bs1, line = 3, label = "Black-Scholes", xlabel = L"s",   
    ylabel = L"v_T(s)", legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent)
    plot!(m1.sgrid, vf1.v, line = 3, label = "Finite-difference (forward)", linestyle = :dash)   

p_vf2 = plot(m2.sgrid, bs2, line = 3, label = "Black-Scholes", xlabel = L"s",   
    ylabel = L"v_T(s)", legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent)
    plot!(m2.sgrid, vf2.v, line = 3, label = "Finite-difference (forward)", linestyle = :dash, xlabel = L"s", ylabel = L"v")

# Plots: Backward difference
p_vb1 = plot(m1.sgrid, bs1, line = 3, label = "Black-Scholes", xlabel = L"s", ylabel = L"v_T(s)", legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent)
    plot!(m1.sgrid, vb1.v, line = 3, label = "Finite-difference (backward)", linestyle = :dash, xlabel = L"s", ylabel = L"v", color = palette(:auto)[3] )

p_vb2 = plot(m2.sgrid, bs2, line = 3, label = "Black-Scholes", xlabel = L"s", ylabel = L"v_T(s)", legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent)    
    plot!(m2.sgrid, vb2.v, line = 3, label = "Finite-difference (backward)", linestyle = :dash, xlabel = L"s", ylabel = L"v", color = palette(:auto)[3] )

p_vf = plot(p_vf1, p_vb1, p_vf2, p_vb2, layout = (2, 2), size = (1000, 800))

############################################################
### Discretization error ###
############################################################

m = BlackScholesModel(Nt = 200, Ns = 150, T = 1.0, r = 0.01, σS = 0.30)

Ns_grid = 100:5:200
Nt_grid = 150:25:300
errors = [discretization_error(BlackScholesModel(Nt = Nt, Ns = Ns, T = 1.0, r = 0.01, σS = 0.30)) for Nt in Nt_grid, Ns in Ns_grid]

# Heatmap of RMSE across (Ns, Nt)
heatmap_err = heatmap(Ns_grid, Nt_grid, errors,
    xlabel = L"N_s", ylabel = L"N_t", colorbar_title = L"\log(RMSE)")

############################################################
### Implicit scheme ###
############################################################

m  = BlackScholesModel(Nt = 25, Ns = 100, T = 1.0, r = 0.01, σS = 0.30)
vi = fd_implicit(m)
ve = fd_scheme(m, forward = true)
bs = BlackScholes.([m], m.sgrid)
discretization_error(m), discretization_error_implicit(m)

p_vi = plot(m.sgrid, bs, line = 3, label = "Black-Scholes", xlabel = L"s", ylabel = L"v_T(s)", 
    legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent);
    plot!(m.sgrid, vi.v, line = 3, label = "Finite-difference (implicit)", linestyle = :dash)

Ns_grid = 100:5:200
Nt_grid = 150:25:300
errors = [discretization_error_implicit(BlackScholesModel(Nt = Nt, Ns = Ns, T = 1.0, r = 0.01, σS = 0.30)) for Nt in Nt_grid, Ns in Ns_grid]

# Heatmap of RMSE across (Ns, Nt)
heatmap_err = heatmap(Ns_grid, Nt_grid, errors,
    xlabel = L"N_s", ylabel = L"N_t", colorbar_title = L"\log(RMSE)")