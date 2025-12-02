using Interpolations, Plots, LaTeXStrings, Distributions
pgfplotsx()
default(legend_font_halign=:left)

############################################################
### Include helper functions ###
############################################################

for f in ("make_grid.jl", "tauchen.jl", "model_struct.jl", "fd_derivative.jl", "vf_iteration.jl", "c_bounds.jl", "egm_step.jl")
    include(joinpath(@__DIR__,"helper_functions/", f))
end

############################################################
### One-step of Value function iteration ###
############################################################

# Instantiate the model
m = ConsumptionSavingsDT(ρ = 0.05, σ = 0.25)
V_terminal(m, W) = W == 0 ? (m.γ > 1 ? -Inf : 0.0) : m.A * W.^(1-m.γ)/(1-m.γ)
C1, V1, Mgrid = vf_iteration(m, V_terminal)
Cbounds = c_bounds(m, Mgrid)

# Consumption policy
p_vf1a = plot(Mgrid, C1, line = 3, xlabel = L"M", ylabel = L"c", label = L"c_1", legend = :topleft,
    foreground_color_legend=:transparent, background_color_legend = :transparent)
    plot!(Mgrid, Cbounds.min, line = 2, ls = :dash, label = L"\underline{c}")
    plot!(Mgrid, Cbounds.max, line = 2, ls = :dash, label = L"\overline{c}")

# Value function
p_vf1b = plot(Mgrid, V1, line = 3, xlabel = L"M", ylabel = L"V", label = L"V_1", legend = :topleft,
    foreground_color_legend=:transparent, background_color_legend = :transparent)
    plot!(0.19:0.25:Mgrid[end], V_terminal(m, 0.19:0.25:Mgrid[end]), line = 3, ls = :dash, label = L"V_0")

plot(p_vf1a, p_vf1b, layout = (1, 2), size = (1000, 400))

# Finite difference derivatives
C10, V10, Mgrid0 = vf_iteration(m, V_terminal, NM = 11,  Nc = 101)
C11, V11, Mgrid1 = vf_iteration(m, V_terminal, NM = 11,  Nc = 1001)
C12, V12, Mgrid2 = vf_iteration(m, V_terminal, NM = 11,  Nc = 10001)
C13, V13, Mgrid3 = vf_iteration(m, V_terminal, NM = 101, Nc = 1001)
C14, V14, Mgrid4 = vf_iteration(m, V_terminal, NM = 101, Nc = 10001)
C15, V15, Mgrid5 = vf_iteration(m, V_terminal, NM = 101, Nc = 100001)

p_vf2a = plot(Mgrid0, fd_derivative(Mgrid0, C10), line = 2, xlabel = L"M", ylabel = L"MPC(M)", label = L"N_c = 101", 
    legend = :topright, foreground_color_legend=:transparent, background_color_legend = :transparent)
    plot!(Mgrid1, fd_derivative(Mgrid1, C11), line = 2, ls = :dash, label = L"N_c = 1,001")
    plot!(Mgrid2, fd_derivative(Mgrid2, C12), line = 2, ls = :dot, label = L"N_c = 10,001")

p_vf2b = plot(Mgrid3, fd_derivative(Mgrid3, C13), line = 2, xlabel = L"M", ylabel = L"MPC(M)", label = L"N_c = 1,001", 
    legend = :topright, foreground_color_legend=:transparent, background_color_legend = :transparent)
    plot!(Mgrid4, fd_derivative(Mgrid4, C14), line = 2, ls = :dash, label = L"N_c = 10,001")
    plot!(Mgrid5, fd_derivative(Mgrid5, C15), line = 2, ls = :dot, label = L"N_c = 100,001",  foreground_color_legend=:transparent, background_color_legend = :transparent)

plot(p_vf2a, p_vf2b, layout = (1, 2), size = (1000, 400))

############################################################
### Endogenous gridpoint method ###
############################################################

m = ConsumptionSavingsDT(N = 100, α = 1.5)
policies = [egm_step(m, 1, M-> M)]
for i = 2:8
    push!(policies, egm_step(m, i, M->policies[i-1][1](M)))
end

grids = [m.Mgrid .+ policies[j][2][1] for j = 1:8]
cs    = [policies[j][1](grids[j]) for j = 1:8]
p_vf3a = plot(m.Mgrid, cs, line = 2, label = permutedims([latexstring("T = $(j)") for j = 1:8]), xlabel = L"M - M_{min}", ylabel = L"c", 
    legend = :topleft, title = "Policy functions", foreground_color_legend=:transparent, background_color_legend = :transparent)
    
MPCs = [fd_derivative(grids[j], cs[j]) for j = 1:8]
p_vf3b = plot(m.Mgrid, MPCs, line = 2, label = permutedims([latexstring("T = $(j)") for j = 1:8]), xlabel = L"M - M_{min}", ylabel = L"MPC = \frac{\Delta c}{\Delta M}", 
    legend = :topright, title = "Marginal propensity to consume", foreground_color_legend=:transparent, background_color_legend = :transparent)

plot(p_vf3a, p_vf3b, layout = (1, 2), size = (1000, 400))