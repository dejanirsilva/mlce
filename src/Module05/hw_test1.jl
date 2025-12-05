#########################################################
# Hennessy and Whited (2007) - Finite Difference Version
#########################################################
using Plots, LaTeXStrings, Distributions, LinearAlgebra
using Lux, Optimisers, Zygote, Random, ProgressMeter

#########################################################
### Model struct ###
#########################################################
@kwdef struct HennessyWhited
    α::Float64 = 0.55
    θ::Float64 = 0.26
    z̅::Float64 = -1.0
    σz::Float64 = 0.123
    δ::Float64 = 0.1
    χ::Float64 = 10.0
    λ::Float64 = 0.059
    ρ::Float64 = 0.04
end

#########################################################
### Neural Networks ###
#########################################################
function create_networks(rng)
    v_core = Chain(
        Dense(2, 64, Lux.swish),
        Dense(64, 32, Lux.swish),
        Dense(32, 16, Lux.swish),
        Dense(16, 1)
    )
    
    i_core = Chain(
        Dense(2, 64, Lux.relu),
        Dense(64, 32, Lux.relu),
        Dense(32, 32, Lux.relu),
        Dense(32, 1)
    )
    
    θᵥ, stᵥ = Lux.setup(rng, v_core) |> Lux.f64
    θᵢ, stᵢ = Lux.setup(rng, i_core) |> Lux.f64
    
    return v_core, i_core, θᵥ, stᵥ, θᵢ, stᵢ
end

# Value function with boundary condition V(0,z) = 0
function v_net(s, θᵥ, v_core, stᵥ, α)
    k = s[1:1, :]
    return k.^α .* v_core(s, θᵥ, stᵥ)[1]
end

# Investment policy
function i_net(s, θᵢ, i_core, stᵢ)
    return i_core(s, θᵢ, stᵢ)[1]
end

#########################################################
### Drift and Diffusion ###
#########################################################
function μₛ(m::HennessyWhited, s, i̅)
    k = s[1:1, :]
    z = s[2:2, :]
    μk = (i̅ .- m.δ) .* k
    μz = -m.θ .* (z .- m.z̅)
    return vcat(μk, μz)
end

function σₛ(m::HennessyWhited, s)
    n = size(s, 2)
    # Only z has volatility, k is deterministic given i
    return vcat(zeros(1, n), m.σz * ones(1, n))
end

#########################################################
### Finite Difference Drift Approximation ###
#########################################################
function drift_fd(s, m::HennessyWhited, θᵥ, θᵢ, v_core, stᵥ, i_core, stᵢ; h=1e-3)
    i̅ = i_net(s, θᵢ, i_core, stᵢ)
    μ = μₛ(m, s, i̅)
    σ = σₛ(m, s)
    
    # Evaluate V at perturbed points using 5-point stencil for accuracy
    V(x) = v_net(x, θᵥ, v_core, stᵥ, m.α)
    
    # Perturbation direction: σ * ε + μ * ε²/2
    # For second derivative, use: F''(0) where F(ε) = V(s + σ*ε/√2 + μ*ε²/(2m))
    # With m=1 shock, this simplifies
    
    # 5-point central difference for second derivative
    F(ϵ) = V(s .+ σ .* (ϵ / sqrt(2.0)) .+ μ .* (ϵ^2 / 2.0))
    
    drift = (-F(2h) .+ 16.0 .* F(h) .- 30.0 .* F(0.0) .+ 16.0 .* F(-h) .- F(-2h)) ./ (12.0 * h^2)
    
    return drift
end

#########################################################
### Dividends ###
#########################################################
function dividends(s, m::HennessyWhited, i̅)
    k = s[1:1, :]
    z = s[2:2, :]
    revenue = exp.(z) .* k.^m.α
    D_star = revenue .- (i̅ .+ 0.5 .* m.χ .* i̅.^2) .* k
    D = D_star .* (1.0 .+ m.λ .* (D_star .< 0))
    return D, D_star
end

#########################################################
### HJB Residual ###
#########################################################
function hjb_residual(s, m::HennessyWhited, θᵥ, θᵢ, v_core, stᵥ, i_core, stᵢ)
    i̅ = i_net(s, θᵢ, i_core, stᵢ)
    D, _ = dividends(s, m, i̅)
    drift = drift_fd(s, m, θᵥ, θᵢ, v_core, stᵥ, i_core, stᵢ)
    V_val = v_net(s, θᵥ, v_core, stᵥ, m.α)
    
    return D .+ drift .- m.ρ .* V_val
end

#########################################################
### Loss Functions ###
#########################################################
function loss_v(s, m, θᵥ, θᵢ, v_core, stᵥ, i_core, stᵢ)
    hjb = hjb_residual(s, m, θᵥ, θᵢ, v_core, stᵥ, i_core, stᵢ)
    return mean(abs2, hjb)
end

function loss_i(s, m, θᵥ, θᵢ, v_core, stᵥ, i_core, stᵢ)
    hjb = hjb_residual(s, m, θᵥ, θᵢ, v_core, stᵥ, i_core, stᵢ)
    return -mean(hjb)
end

#########################################################
### Training Loop ###
#########################################################
function train_model(; max_iter=50_000)
    m = HennessyWhited(z̅=0.0, λ=0.059, σz=0.123, θ=0.26, χ=0.10)
    
    rng = Xoshiro(1234)
    v_core, i_core, θᵥ, stᵥ, θᵢ, stᵢ = create_networks(rng)
    
    optᵥ = Optimisers.Adam(1e-3)
    optᵢ = Optimisers.Adam(1e-4)
    osᵥ = Optimisers.setup(optᵥ, θᵥ)
    osᵢ = Optimisers.setup(optᵢ, θᵢ)
    
    kmin, kmax = 5, 35.0
    dk = Uniform(kmin, kmax)
    d_z = Normal(m.z̅, m.σz / sqrt(2.0 * m.θ))
    
    loss_history_v = Float64[]
    loss_history_i = Float64[]
    
    p = Progress(max_iter; desc="Training...", dt=1.0)
    
    for it in 1:max_iter
        # Sample mini-batch
        k_batch = rand(rng, dk, 128)'
        z_batch = rand(rng, d_z, 128)'
        s_batch = vcat(k_batch, z_batch)
        
        # Policy evaluation step (update value function)
        lossᵥ, backᵥ = Zygote.pullback(θᵥ) do p
            loss_v(s_batch, m, p, θᵢ, v_core, stᵥ, i_core, stᵢ)
        end
        gradᵥ = first(backᵥ(1.0))
        osᵥ, θᵥ = Optimisers.update(osᵥ, θᵥ, gradᵥ)
        
        # Policy improvement step (update policy)
        lossᵢ, backᵢ = Zygote.pullback(θᵢ) do p
            loss_i(s_batch, m, θᵥ, p, v_core, stᵥ, i_core, stᵢ)
        end
        gradᵢ = first(backᵢ(1.0))
        osᵢ, θᵢ = Optimisers.update(osᵢ, θᵢ, gradᵢ)
        
        push!(loss_history_v, lossᵥ)
        push!(loss_history_i, abs(lossᵢ))
        
        next!(p, showvalues=[(:iter, it), ("Loss_v", lossᵥ), ("Loss_i", abs(lossᵢ))])
    end
    
    return m, v_core, i_core, θᵥ, stᵥ, θᵢ, stᵢ, loss_history_v, loss_history_i
end

#########################################################
### Run and Plot ###
#########################################################
m, v_core, i_core, θᵥ, stᵥ, θᵢ, stᵢ, loss_v_hist, loss_i_hist = train_model(max_iter=50_000)

# Create test points
kmin, kmax = 5.0, 35.0
k_test = collect(range(kmin, kmax, length=100))'
z_levels = [m.z̅ - 0.15, m.z̅, m.z̅ + 0.15]
s_test = [vcat(k_test, z * ones(1, 100)) for z in z_levels]

# Plot results
p1 = plot(title="Value Function V(k,z)", xlabel="k", ylabel="V")
p2 = plot(title="Investment Policy i(k,z)", xlabel="k", ylabel="i")
p3 = plot(title="Dividends D(k,z)", xlabel="k", ylabel="D")

labels = ["z = z̄ - 0.15", "z = z̄", "z = z̄ + 0.15"]
colors = [:blue, :green, :red]

for (idx, s) in enumerate(s_test)
    V_vals = v_net(s, θᵥ, v_core, stᵥ, m.α)[:]
    i_vals = i_net(s, θᵢ, i_core, stᵢ)[:]
    D_vals, _ = dividends(s, m, i_net(s, θᵢ, i_core, stᵢ))
    
    plot!(p1, k_test[:], V_vals, label=labels[idx], lw=2, color=colors[idx])
    plot!(p2, k_test[:], i_vals, label=labels[idx], lw=2, color=colors[idx])
    plot!(p3, k_test[:], D_vals[:], label=labels[idx], lw=2, color=colors[idx])
end

# Plot loss history
p4 = plot(loss_v_hist, yscale=:log10, label="Value Loss", xlabel="Iteration", ylabel="Loss", title="Training Loss")
plot!(p4, loss_i_hist, label="Policy Loss")

plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))