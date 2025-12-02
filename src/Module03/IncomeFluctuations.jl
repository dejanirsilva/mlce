using Interpolations, Plots, LaTeXStrings, Distributions
using LinearAlgebra, SparseArrays, Statistics
pgfplotsx()
default(legend_font_halign=:left)

############################################################
### Include helper functions ###
############################################################

for f in ("income_fluctuations_struct.jl", "income_fluctuations_scheme.jl", "check_HJB.jl", "fd_derivative.jl")
    include(joinpath(@__DIR__,"helper_functions/", f))
end

############################################################
### Income Fluctuations Problem ###
############################################################

m = IncomeFluctuationsModel(N = 500, Wmax = 2.0, Wmin = -0.2)

# Solving Income Fluctuations Problem with implicit scheme
v, vW, c, s, residual = fd_implicit(m)
check_HJB(m, v, vW, c, s).^2 |> mean |> sqrt


p_if1 = plot(m.Wgrid, v[:,1], line = 3, xlabel = L"W", ylabel = L"v_j(W)", legend = :topleft, 
            foreground_color_legend=:transparent, background_color_legend = :transparent, label ="Low income");
        plot!(m.Wgrid, v[:,2], line = 3, label ="High income", linestyle = :dash);
        vline!([m.Wmin], line = :dash, label = L"W = W_{\min}", color = :black)

p_if2 = plot(m.Wgrid, fd_derivative(m.Wgrid, c[:,1]), line = 3, xlabel = L"W", ylabel = L"v_{j,W}(W)", legend = :topright, 
            foreground_color_legend=:transparent, background_color_legend = :transparent, label ="Low income");
        plot!(m.Wgrid, fd_derivative(m.Wgrid, c[:,2]), line = 3, label ="High income", linestyle = :dash);
        vline!([m.Wmin], line = :dash, label = L"W = W_{\min}", color = :black)

p_if3 = plot(m.Wgrid, c[:,1], line = 3, xlabel = L"W", ylabel = L"c_j(W)", legend = :topleft, 
            foreground_color_legend=:transparent, background_color_legend = :transparent, label ="Low income");
        plot!(m.Wgrid, c[:,2], line = 3, label ="High income", linestyle = :dash);
        vline!([m.Wmin], line = :dash, label = L"W = W_{\min}", color = :black)

p_if4 = plot(m.Wgrid, s[:,1], line = 3, xlabel = L"W", ylabel = L"s_j(W)", legend = :topright, 
            foreground_color_legend=:transparent, background_color_legend = :transparent, label ="Low income");
        plot!(m.Wgrid, s[:,2], line = 3, label ="High income", linestyle = :dash);
        vline!([m.Wmin], line = :dash, label = L"W = W_{\min}", color = :black)

plot(p_if1, p_if2, p_if3, p_if4, layout = (2, 2), size = (1000, 800))