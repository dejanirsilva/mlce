using Plots, LaTeXStrings, Distributions, LinearAlgebra, BenchmarkTools
using Lux, Optimisers, ForwardDiff, Zygote, Random, ProgressMeter
pgfplotsx()
default(legend_font_halign=:left)

#########################################################
### Lucas Orchard model ###
#########################################################

# Model parameters
@kwdef struct LucasOrchard
	ρ::Float64 = 0.04
    N::Int = 10
	σ::Vector{Float64} = sqrt(0.04) * ones(N)
	μ::Vector{Float64} = 0.02 * ones(N)
    μc::Function = s -> μ' * s
    σc::Function = s -> [s[i,:]' * σ[i] for i in 1:N]
	μₛ::Function = s ->  s .* (μ .- μc(s)- s.*σ.^2 .+ sum(σc(s)[i].^2 for i in 1:N))
	σₛ::Function = s -> [s .* ([j == i ? σ[i] : 0 for j in 1:N] .- σc(s)[i]) for i in 1:N] 
end;

N   = 10
m   = LucasOrchard(N = N)
rng = MersenneTwister(0)
α   = ones(N)
d   = Dirichlet(α)
s_samples = rand(rng, d, 1_000)
vcat(m.μₛ(s_samples), m.σₛ(s_samples)...)

# Hyper-dual approach to Ito's lemma
function drift_hyper(V::Function, s::AbstractMatrix, m::LucasOrchard)
	N, σs, μs = m.N, m.σₛ(s), m.μₛ(s) # Preallocations
    F(ϵ) = sum(V(s .+ σs[i] .* (ϵ / sqrt(2)) .+ μs .* (ϵ^2 / (2 * N))) for i in 1:N)
	return ForwardDiff.derivative(ϵ -> ForwardDiff.derivative(F, ϵ), 0.0)
end
V_test(s) = sum(s.^2, dims = 1)

s = s_samples
drifts_exact = map(1:size(s, 2)) do i
    ∇V, H = 2 * s[:,i], 2 * Matrix(I, length(s[:,i]), length(s[:,i]))
    dot(∇V, m.μₛ(s[:,i])) +  0.5 * tr(vcat(m.σₛ(s[:,i])'...)' * H * vcat(m.σₛ(s[:,i])'...))
end'
drifts_hyper = drift_hyper(V_test, s_samples, m)
maximum(abs.(drift_hyper(V_test, s_samples, m) - drifts_exact))

### Defining the neural net
model = Chain(
	Dense(10 => 25, Lux.gelu),
	Dense(25 => 25, Lux.gelu),
	Dense(25 => 1)
)

# Initializaion:
ps, ls = Lux.setup(rng, model) |> f64
opt = Adam(1e-3)
os = Optimisers.setup(opt, ps)

# Loss function
function loss_fn(ps, ls, s, target)
	return mean(abs2, model(s, ps, ls)[1] - target)
end

# Target
function target(v, s, m; Δt = 0.2)
	v̅ = v(s)
    hjb = s[1,:]' + drift_hyper(v, s, m) - m.ρ * v̅
	return v̅ + hjb * Δt, mean(abs2, hjb)
end
hjb_residual(v, s, m) = mean(abs2, s[1,:]' + drift_hyper(v, s, m) - m.ρ * v(s))

# # Training loop
max_iter, Δt = 40_000, 1.0
# Sampling interior and edge states
d_int  = Dirichlet(ones(m.N))
d_edge = Dirichlet(0.05 .* ones(m.N))  # tune 0.3 up/down
# Loss history and EMA
loss_history, loss_ema_history, α_ema = Float64[], Float64[], 0.99
p = Progress(max_iter; desc="Training...", dt=1.0)
for i = 1:max_iter
    if rand(rng) < 0.50
        s_batch = rand(rng, d_int, 128)
    else
        s_batch = rand(rng, d_edge, 128)
    end
    tgt, hjb_res = target(s-> model(s, ps, ls)[1], s_batch, m, Δt = Δt)
    loss, back = Zygote.pullback(p -> loss_fn(p, ls, s_batch, tgt), ps)
    grad 	   = first(back(1.0))
    os, ps     = Optimisers.update(os,ps, grad)
    loss_ema   = i==1 ? loss : α_ema * loss_ema + (1.0 - α_ema) * loss
    push!(loss_history, loss)
    push!(loss_ema_history, loss_ema)
    next!(p, showvalues = [(:iter, i),("Loss", loss), ("Loss EMA", loss_ema), ("HJB residual", hjb_res)])
end

p_history = plot((1:length(loss_history))/1000, loss_history, yaxis = :log10,  xlabel = "Iteration (in thousands)", ylabel = "Loss", title = "", legend = :topright,
    foreground_color_legend=:transparent, background_color_legend = :transparent, label = "Loss")
    plot!((1:length(loss_ema_history))/1000, loss_ema_history, l = 2.0, label = "Loss EMA")

α_range = range(0.1, 1.5,15)
collect(α_range)
ds1 = [Dirichlet(α*ones(m.N)) for α in α_range]
αs = ones(10,10)
for i in 1:10
    αs[i, i] = 4.0
end
ds2 = [Dirichlet(αs[i,:]) for i in 1:10]
ds = vcat(ds1,ds2)
s_test_samples1 = [rand(rng, d, 10_000) for d in ds1]
s_test_samples2 = [rand(rng, d, 10_000) for d in ds2]

function hjb_residuals(s, m, model, ps, ls)
    v = s -> model(s, ps, ls)[1]
    v̅ = v(s)
    hjb = s[1,:]' + drift_hyper(v, s, m) - m.ρ * v̅
	return mean(abs2, hjb)
end

residuals1 = [hjb_residuals(s, m, model, ps, ls) for s in s_test_samples1]
residuals2 = [hjb_residuals(s, m, model, ps, ls) for s in s_test_samples2]
residuals = vcat(residuals1, residuals2)
p_residuals1 = bar(α_range, residuals1, xlabel = L"\alpha_{scale}", ylabel = "HJB residual (MSE)", title = "", label = "",
    yscale = :log10, ylims = (1e-6, 1e-3), xticks = α_range)
p_residuals2 = bar(1:10, residuals2, xlabel = "tree number", ylabel = "HJB residual (MSE)", title = "", label = "",
    yscale = :log10, ylims = (1e-6, 1e-3), color = palette(:auto)[2], xticks = 1:1:10)
p_residuals = plot(p_residuals1, p_residuals2, layout = (2, 1), size = (600, 550))

plot(p_history, p_residuals, layout = (1, 2), size = (1000, 600))

# Special case: two trees
v_two_trees(s₁) = model(reshape([s₁;1.0-s₁;zeros(8)], 10,1), ps, ls)[1][1]
function v_analytic(m, s)
    ρ = m.ρ
    return s == 0 ? 0.0 : s == 1 ? 1/ρ : 1/(2*ρ) * (1 + (1-s)/s * log(1-s) - s/(1-s) * log(s))
end
s₁_range = range(0.0, 1.0, 100)
p_prediction = plot(s₁_range, [v_two_trees(s₁) for s₁ in s₁_range], label = "Prediction", xlabel = L"s_1", ylabel = L"v(\mathbf{s})", legend = :topleft, l = 3, alpha = 0.85)
plot!(s₁_range, [v_analytic(m, s₁) for s₁ in s₁_range], label = "Exact", xlabel = L"s_1", ylabel = L"v(\mathbf{s})", legend = :topleft, ls = :dash, l = 2, foreground_color_legend=:transparent, background_color_legend = :transparent)

#########################################################
### Dirichlet distribution plotting ###
#########################################################

using CairoMakie, LaTeXStrings
using Makie: colgap!, rowgap!
using TernaryDiagrams      
using Distributions

"""
    dirichlet_simplex_grid(dist; n=80)

Return (a1, a2, a3, w) where a1,a2,a3 are coordinates on the simplex
and w is pdf(dist, [a1[i],a2[i],a3[i]]).
n controls the grid resolution.
"""
function dirichlet_simplex_grid(dist::Dirichlet; n::Int = 80)
    h = 1f0 / (n + 1)  # step size so we never hit 0 or 1 exactly

    a1 = Float32[]
    a2 = Float32[]
    a3 = Float32[]
    w  = Float32[]

    for i in 0:n
        x1 = (i + 0.5f0) * h
        for j in 0:(n - i)
            x2 = (j + 0.5f0) * h
            x3 = 1f0 - x1 - x2
            if x3 <= 0f0       # outside simplex, skip
                continue
            end

            push!(a1, x1)
            push!(a2, x2)
            push!(a3, x3)

            val = pdf(dist, [x1, x2, x3])
            push!(w, isfinite(val) ? Float32(val) : NaN32)
        end
    end

    return a1, a2, a3, w
end

alphas = [
    (1.2, 1.2, 1.2),
    (0.8, 0.8, 0.8),
    (4.0,  1.0,  1.0),
    (1.0,  4.0,  1.0),
]

fig = Figure(size = (900, 800), figure_padding = 20)
colgap!(fig.layout, 50)
rowgap!(fig.layout, 40)

for (k, α) in enumerate(alphas)
    dist = Dirichlet(collect(α))

    a1, a2, a3, w = dirichlet_simplex_grid(dist; n = 320)
    # w_clipped = clamp.(w, 0f0, 50f0)
    row = (k <= 2) ? 1 : 2
    col = (k % 2 == 1) ? 1 : 2

    panel = fig[row, col] = GridLayout()
    colgap!(panel, 20)

    ax = Axis(panel[1, 1], aspect = DataAspect())
    hidespines!(ax)
    ternaryaxis!(ax; labelx="", labely="", labelz="")  # no auto labels

    # manually place along edges
    text!(ax, 0.5, -0.08; text = L"x_1", align = (:center, :top), fontsize = 16)
    text!(ax, 0.1, 0.5;  text = L"x_2", align = (:left,  :bottom), fontsize = 16)
    text!(ax, 0.9, 0.5;  text = L"x_3", align = (:right, :bottom), fontsize = 16)

    contour_plot = ternarycontourf!(ax, a1, a2, a3, w; levels = 15)
    Colorbar(panel[1, 2], contour_plot; label = "pdf", width = 14)

    hidedecorations!(ax)

    Label(fig[row, col, Top()],
          latexstring("\\alpha = [", α[1], ",\\; ", α[2], ",\\; ", α[3], "]"),
          tellwidth = false,
          fontsize = 14)
end

fig
