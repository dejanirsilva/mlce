using Random, Plots, LaTeXStrings, Statistics, LinearAlgebra
using Optimisers, Zygote, Roots
pgfplotsx()
default(legend_font_halign=:left)

############################################################
# Simple SGD example
############################################################
# True parameter
rng = Random.MersenneTwister(123)
θ_true, sample_size, batch_size = 2.0, 100_000, 32
noisy_sample = θ_true .+ 0.5 .* randn(rng, sample_size)

# Gradient functions: function and mini-batch version
grad_full(θ) = 2 * (θ - mean(noisy_sample))
grad_sgd(θ, B) = 2 * (θ - 
    mean(noisy_sample[rand(rng, 1:sample_size, B)]))

# Training loop
η = 0.05
θ_full, θ_sgd = 0.0, 0.0
θ_path_full, θ_path_sgd = Float64[], Float64[]
for t in 1:200
    θ_full -= η * grad_full(θ_full)
    θ_sgd  -= η * grad_sgd(θ_sgd, batch_size)
    push!(θ_path_full, θ_full)
    push!(θ_path_sgd, θ_sgd)
end

θ_path_full
plot(1:200, θ_path_full, lw=3, label="Full-batch GD")
plot!(1:200, θ_path_sgd, lw=3, label="SGD (B=32)", xlabel="Iteration", ylabel=L"\theta",
      title="", legend = :bottomright, linestyle = :dash,
      foreground_color_legend=:transparent, background_color_legend = :transparent)
hline!([θ_true[1]], ls=:dash, color=:black, label="True θ")

############################################################
# Non-convex loss landscape
############################################################
# Loss function and gradient
L(θ) = θ^4 - 3θ^2 + θ          # multiple minima
gradL(θ) = 4θ^3 - 6θ + 1

# Gradient descent and noisy SGD comparison
function run_descent(θ0; η=0.05, σ=0.0, T=200, rng = Random.MersenneTwister(123))
    θ = θ0
    path = Float64[θ0]
    for t in 1:T
        θ -= η * (gradL(θ) + σ * randn(rng))  # noise in gradient
        push!(path, θ)
    end
    path
end

θ0s = -2.0:0.5:2.0

θ_gd  = [run_descent(θ0; η=0.05, σ=0.0)[end] for θ0 in θ0s]   # deterministic GD
θ_sgd = [run_descent(θ0; η=0.05, σ=4.5)[end] for θ0 in θ0s]   # noisy SGD
[θ_gd'; θ_sgd']
θ_gd  = run_descent(θ0s[end-1]; η=0.05, σ=0.0, T = 250)   # deterministic GD
θ_sgd = run_descent(θ0s[end-1]; η=0.05, σ=4.2, T = 250)   # noisy SGD

# Plot trajectories on loss surface
θgrid = range(-2.0, 2.0; length=400)
p = plot(θgrid, L.(θgrid), lw=3, label="Loss surface L(θ)", xlabel=L"\theta", ylabel="Loss", 
    linewidth = 2, foreground_color_legend=:transparent, background_color_legend = :transparent, 
    legend = :topleft, ylims = (-4.2,6.2))
plot!(θ_gd[1:50:end], L.(θ_gd)[1:50:end], lw=4, alpha = 0.35, marker=:circle, label="Full GD (stuck in local min)", color = palette(:auto)[2])
plot!(θ_sgd[1:25:151], L.(θ_sgd)[1:25:151], lw=4, alpha = 0.35, marker=:diamond, label="SGD (escapes)", color = palette(:auto)[3])

for i = 1:25:(151)
    j = Int64(1 + (i-1)/25)
    annotate!(p, θ_sgd[i], L.(θ_sgd)[i]-0.35, Plots.text(j;pointsize=10, color = palette(:auto)[3]))
end
p

############################################################
# Momentum
############################################################

A = Diagonal([1.0, 100.0])
μ, Lm = 1.0, 100.0
η_gd = 2/(Lm+μ)                     # ≈ 0.01980
η_hb = 4/( (sqrt(Lm)+sqrt(μ))^2 )   # ≈ 0.03306
β_hb = ((sqrt(Lm)-sqrt(μ))/(sqrt(Lm)+sqrt(μ)))^2  # ≈ 0.66942

∇L(θ) = A*θ

function run_gd(θ0, T)
    θ = θ0; path = [copy(θ)]; losses = Float64[]
    for _ in 1:T
        push!(losses, 0.5*dot(θ, A*θ))
        θ -= η_gd * ∇L(θ)
        push!(path, copy(θ))
    end
    hcat(path...)', losses
end

function run_hb(θ0, T)
    θ = θ0; v = zeros(2); path = [copy(θ)]; losses = Float64[]
    for _ in 1:T
        push!(losses, 0.5*dot(θ, A*θ))
        g = ∇L(θ)
        v = β_hb*v + (1-β_hb)*g
        θ -= η_hb * v
        push!(path, copy(θ))
    end
    hcat(path...)', losses
end

θ0 = [2.0, 2.0]; T = 160
pgd, lgd = run_gd(θ0, T)
phb, lhb = run_hb(θ0, T)

# Trajectories
θ1 = range(-2,2; length=200); θ2 = range(-2,2; length=200)
Z = [0.5*([x,y]'*A*[x,y]) for x in θ1, y in θ2]
contour(θ1, θ2, Z; levels=20, xlabel=L"\theta_1", ylabel=L"\theta_2", color = :grays, legend=:outerbottom,
        title="")
plot!(pgd[:,1], pgd[:,2], lw=1, label="Gradient descent", alpha = 0.75, color = palette(:auto)[1], legend_column = 1, 
    extra_kwargs = Dict(
      :subplot => Dict(
        :legend_style => "{at={(0.5,-0.12)},anchor=north,draw=none,fill=none}"  # tweak -0.24 as needed
      )
    ))
plot!(phb[:,1], phb[:,2], lw=3, label="Momentum", alpha = 1.0, color = palette(:auto)[2], legend_column = 2, foreground_color_legend=:transparent, background_color_legend = :transparent,
    )

# Loss evolution
plot(0:T-1, lgd, lw=3, label="Gradient descent", yscale=:log10, xlabel="iteration", ylabel="loss", 
    legend = :topright, 
     title="", foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(0:T-1, lhb, lw=3, label="Momentum", alpha = 1.0, color = palette(:auto)[2])

############################################################
# RMSProp
############################################################

# ----- Problem: ill-conditioned quadratic
A = Diagonal([1.0, 100.0])              # eigenvalues μ=1, L=100
f(θ) = 0.5 * dot(θ, A*θ)
∇f(θ) = A * θ

# ----- Optimizers
η_sgd = 0.0198                            # workable but shows zig-zag/slow progress
η_rms = η_sgd                            # RMSProp base LR

ρ = 0.9                                  # EMA decay
ϵ = 1e-8

function run_sgd(θ0; T=160)
    θ = copy(θ0); path = [copy(θ)]; loss = Float64[]
    for t in 1:T
        push!(loss, f(θ))
        θ -= η_sgd * ∇f(θ)
        push!(path, copy(θ))
    end
    hcat(path...)', loss
end

function run_rmsprop(θ0; T=160)
    θ = copy(θ0); v = zeros(size(θ)); path = [copy(θ)]; loss = Float64[]
    for t in 1:T
        g = ∇f(θ)
        v .= ρ .* v .+ (1-ρ) .* (g .^ 2)            # EMA of squared grads
        θ .-= η_rms .* (g ./ sqrt.(v .+ ϵ))          # per-parameter step
        push!(loss, f(θ)); push!(path, copy(θ))
    end
    hcat(path...)', loss
end

# ----- Run
θ0 = [2.0, 2.0]
path_sgd, L_sgd = run_sgd(θ0, T = 125)
path_rms, L_rms = run_rmsprop(θ0, T = 125)

# ----- Contours + trajectories
θ1 = range(-2, 2; length=200); θ2 = range(-2, 2; length=200)
Z = [0.5*([x,y]'*A*[x,y]) for x in θ1, y in θ2]

plt_traj = contour(θ1, θ2, Z; levels=20, xlabel="θ₁", ylabel="θ₂", color = :grays, legend=:outerbottom,
    title="")
plot!(plt_traj, path_sgd[:,1], path_sgd[:,2], lw=1, label="Gradient descent",  alpha = 0.75, color = palette(:auto)[1], legend_column = 1, 
extra_kwargs = Dict(
  :subplot => Dict(
    :legend_style => "{at={(0.5,-0.12)},anchor=north,draw=none,fill=none}"  # tweak -0.24 as needed
  )
))
plot!(plt_traj, path_rms[:,1], path_rms[:,2], lw=3, label="RMSProp",  color = palette(:auto)[2], legend_column = 2)

# ----- Loss curves (log scale highlights speed difference)
plt_loss = plot(0:length(L_sgd)-1, L_sgd; yscale=:log10, lw=3, label="Gradient descent", legend = :topright,
    xlabel="iteration", ylabel="loss", title="", foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(plt_loss, 0:length(L_rms)-1, L_rms; yscale=:log10, lw=3, label="RMSProp")


############################################################
### Comparison of optimizers ###
############################################################

#  Anysotropic non-convex loss function
θ_opt   = find_zero(x -> 4x^3 - 6x + 1, -1.0)
ℓ(θ)    = θ^4 - 3θ^2 + θ
weights = range(1,100, length = 10) 
loss(θ) = dot(weights, ℓ.(θ))/sum(weights)

# Initialization
θ0a     = ones(10)* 2.0
θ0b     = -ones(10)* 3.0
η       = 1e-2
steps   = 50_000
tol     = 1e-6
β       = (0.9, 0.999)
ϵ       = 1e-8

function loss_optimiser(loss, θ0, opt; steps=1_000, tol=1e-8, σ = 0.5)
    θ = deepcopy(θ0)
    st  = Optimisers.setup(opt, θ)  # builds a “state tree” matching θ
    losses = Float64[]; gnorms = Float64[]
    for _ in 1:steps
        ℓ, back = Zygote.pullback(loss, θ)
        g = first(back(1.0)) + σ * randn(size(θ))
        push!(losses, ℓ); push!(gnorms, norm(g))
        if gnorms[end] ≤ tol; break; end
        st, θ = Optimisers.update(st, θ, g)
    end
    return θ, (losses=losses, grad_norms=gnorms)
end

# Running the optimizers
gd_θa, gd_statsa        = loss_optimiser(loss, θ0a, Optimisers.Descent(η), σ = 4.0, steps = 50_000)
mom_θa, mom_statsa      = loss_optimiser(loss, θ0a, Optimisers.Momentum(η, β[1]), σ = 4.0, steps = 50_000)
nes_θa, nes_statsa      = loss_optimiser(loss, θ0a, Optimisers.Nesterov(η, β[1]), σ = 4.0, steps = 50_000)
rms_θa, rms_statsa      = loss_optimiser(loss, θ0a, Optimisers.RMSProp(η, β[1], ϵ), σ = 4.0, steps = 50_000)
adam_θa, adam_statsa    = loss_optimiser(loss, θ0a, Optimisers.Adam(η, β, ϵ), σ = 4.0, steps = 50_000)
adamw_θa, adamw_statsa  = loss_optimiser(loss, θ0a, Optimisers.AdamW(η, β, 1e-2, ϵ), σ = 4.0, steps = 50_000)

gd_θb, gd_statsb        = loss_optimiser(loss, θ0b, Optimisers.Descent(η), σ = 0.0, steps = 1000)
mom_θb, mom_statsb      = loss_optimiser(loss, θ0b, Optimisers.Momentum(η, β[1]), σ = 0.0, steps = 1000)
nes_θb, nes_statsb      = loss_optimiser(loss, θ0b, Optimisers.Nesterov(η, β[1]), σ = 0.0, steps = 1000)
rms_θb, rms_statsb      = loss_optimiser(loss, θ0b, Optimisers.RMSProp(η, β[1], ϵ), σ = 0.0, steps = 1000)
adam_θb, adam_statsb    = loss_optimiser(loss, θ0b, Optimisers.Adam(η, β, ϵ), σ = 0.0, steps = 1000)
adamw_θb, adamw_statsb  = loss_optimiser(loss, θ0b, Optimisers.AdamW(η, β, 1e-2, ϵ), σ = 0.0, steps = 1000)

# Plot optimal solutions
p_optb = scatter(1:10, gd_θb, label="Gradient descent", xlabel=L"j", alpha = 0.75, markersize = 6,
    legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent,
    ylabel=L"\theta_j", title="", lw = 3, ylims = (-2.5, 1.5))
scatter!(1:10, mom_θb, label="Momentum", xlabel=L"j", alpha = 0.75, markersize = 6,
    legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent,
    ylabel=L"\theta_j", title="", lw = 3)
scatter!(1:10, nes_θb, label="Nesterov momentum", xlabel=L"j", alpha = 0.75, markersize = 6,
    legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent,
    ylabel=L"\theta_j", title="", lw = 3)
scatter!(1:10, rms_θb, label="RMSProp", xlabel=L"j", alpha = 0.75, markersize = 6,
    legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent,
    ylabel=L"\theta_j", title="", lw = 3)
scatter!(1:10, adam_θb, label="Adam", xlabel=L"j", alpha = 0.75, markersize = 6,
    legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent,
    ylabel=L"\theta_j", title="", lw = 3)
scatter!(1:10, adamw_θb, label="AdamW", xlabel=L"j", alpha = 0.75, markersize = 6,
    legend = :topleft, foreground_color_legend=:transparent, background_color_legend = :transparent,
    ylabel=L"\theta_j", title="", lw = 3)
hline!([θ_opt], label="Optimal θ", color = :black, linestyle = :dash, lw = 3)