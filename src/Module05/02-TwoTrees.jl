using Plots, LaTeXStrings, Distributions, LinearAlgebra, BenchmarkTools, ProgressMeter
using Lux, Optimisers, ForwardDiff, Zygote, Random, Serialization
pgfplotsx()
default(legend_font_halign=:left)

#########################################################
### Two trees model ###
#########################################################

# Model parameters
@kwdef struct TwoTrees
	ρ::Float64 = 0.04
	σ::Float64 = sqrt(0.04)
	μ::Float64 = 0.02
	μₛ::Function = s -> @. -2 * σ^2 * s * (1-s) * (s-0.5)   # drift of s
	σₛ::Function = s -> @. sqrt(2) * σ * s * (1-s)            # diffusion of s
end;

# Drift and diffusion take vector input
m = TwoTrees()
vcat( m.μₛ( ones(1,6) ), m.σₛ( ones(1,6) ) )

# Hyper-dual approach to Ito's lemma
function drift_hyper(V::Function, s::AbstractMatrix, m::TwoTrees)
	F(ϵ) = V(s + m.σₛ(s)/sqrt(2)*ϵ + m.μₛ(s)/2*ϵ^2)
	return ForwardDiff.derivative(ϵ -> ForwardDiff.derivative(F, ϵ), 0.0)
end;

# Small test: exact vs. automatic differentiation
rng = Xoshiro(0)
s   = rand(rng, 1, 1000)
# Exact drift for test function
V_test(s) = sum(s.^2, dims = 1)
drifts_exact = map(1:size(s, 2)) do i
    ∇V, H = 2 * s[:,i], 2 * Matrix(I, length(s[:,i]), length(s[:,i]))
    ∇V' * m.μₛ(s[:,i]) + 0.5 * tr(m.σₛ(s[:,i])' * H * m.σₛ(s[:,i]))
end'
drifts_hyper = drift_hyper(V_test, s, m)
errors = maximum(abs.((drifts_exact - drifts_hyper)))

### Defining the neural net
model = Chain(
	Dense(1 => 25, Lux.gelu),
	Dense(25 => 25, Lux.gelu),
	Dense(25 => 1)
)

# Initialization
ps, ls = Lux.setup(rng, model) |> f64
opt = Adam(1e-3)
os = Optimisers.setup(opt, ps)

# Loss function
function loss_fn(ps, ls, s, target)
	return mean(abs2, model(s, ps, ls)[1] - target)
end

# Target
function target(v, s, m; Δt = 0.2)
	hjb = s + drift_hyper(v, s, m) - m.ρ * v(s)
	return v(s) + hjb * Δt
end

# # Training loop
max_iter = 40_000
loss_history = Float64[]
p = Progress(max_iter; desc="Training...", dt=1.0)
for i = 1:max_iter
    s_batch = rand(rng, 1,  128)
    tgt 	= target(s-> model(s, ps, ls)[1], s_batch, m, Δt = 1.0)
    loss    = loss_fn(ps, ls, s_batch, tgt)
    grad 	= gradient(p -> loss_fn(p, ls, s_batch, tgt), ps)[1]
    os, ps  = Optimisers.update(os,ps, grad)
    push!(loss_history, loss)
    next!(p, showvalues = [(:iter, i),("Loss", loss)])
end

# Training history
p_history = plot((1:length(loss_history)) / 1000, loss_history, yaxis = :log10, xlab = "iteration (in thousands)", ylab = "loss", 
        label = L"\Delta t = 1.0", title = "", legend = :topright, foreground_color_legend=:transparent, background_color_legend = :transparent)

# Plot the prediction
s_range = range(0.0, 1.0, 100)
v_pred  = model(s_range', ps, ls)[1][:]
function v_analytic(m, s)
    ρ = m.ρ
    return s == 0 ? 0.0 : s == 1 ? 1/ρ : 1/(2*ρ) * (1 + (1-s)/s * log(1-s) - s/(1-s) * log(s))
end
v_exact = v_analytic.([m], s_range)
p_prediction = plot(s_range, v_pred, xlabel = L"s", ylabel = L"v(s)", 
     label = "DPI", l = 3, alpha = 0.85)
plot!(s_range, v_exact, label = "Exact", ls = :dash, l = 2,
     legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent)

plot(p_history, p_prediction, layout = (1,2), size = (1000, 350))