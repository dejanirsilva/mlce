#########################################################
# Hennessy and Whited (2007) - Deep Policy Iteration
# Corrected version matching Python implementation
#########################################################

using Lux, Optimisers, Zygote, Random, Statistics
using Distributions, ProgressMeter
using Plots
using Functors: fmap

#########################################################
### Fixed Model Parameters (matching Python exactly) ###
#########################################################

const δ = 0.1        # Depreciation
const α = 0.55       # Returns to scale  
const λ = 0.059      # Equity issuance cost
const θ = 0.26       # Mean reversion of productivity
const σz = 0.123     # Volatility of productivity
const χ = 0.1        # Capital adjustment cost (from Python __init__.py)
const β = 0.96       # Discount factor
const ρ = -log(β)    # Continuous-time discount rate ≈ 0.0408

# State space bounds (matching Python sampling)
const std_z = σz / sqrt(2 * θ)
const z_min = exp(-2 * std_z)
const z_max = exp(2 * std_z)
const k_min = (α * z_min * β / (1 - (1 - δ) * β))^(1 / (1 - α))
const k_max = (α * z_max * β / (1 - (1 - δ) * β))^(1 / (1 - α))

# Normalization for network inputs (matching Python normalize_state)
const state_min = [0.1, -2.0]
const state_max = [300.0, 2.0]

#########################################################
### Neural Networks (matching Python architecture) ###
#########################################################

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

v_net(state, θᵥ) = v_core(state, θᵥ, stᵥ)[1]
i_net(state, θᵢ) = i_core(state, θᵢ, stᵢ)[1] * 1e-3 .+ δ

indicator_fn(x) = 1.0 ./ (1.0 .+ exp.(30.0 .* x))

#########################################################
### MDP (matching Python mdp_fn exactly) ###
#########################################################

function hjb_residuals(state::AbstractMatrix, i_vals::AbstractMatrix)
    K = state[1:1, :]      # Capital
    logz = state[2:2, :]   # Log productivity
    
    z = exp.(logz)                              # Productivity
    y = z .* K.^α                               # Profits: z * K^α
    adj_cost = χ .* K .* i_vals.^2 ./ 2         # Adjustment cost: χ*K*i²/2
    E_star = y .- i_vals .* K .- adj_cost       # Pre-issuance: y - i*K - adj_cost
    
    E = (1.0 .+ λ .* indicator_fn(E_star)) .* E_star
    
    μK = K .* (i_vals .- δ)
    μz = -θ .* logz

    

    return E, μK, μz
end

#########################################################
### Drift computation with finite differences ###
#########################################################

function compute_drift(state::AbstractMatrix, θᵥ, v_core, stᵥ, θπ, π_core, stπ)
    h = 1e-4
    
    # Get policy
    i_vals = π_net(state, θπ, π_core, stπ)
    
    # Get MDP components
    E, μK, μz = compute_mdp(state, i_vals)
    
    # Value at current state
    V = v_net(state, θᵥ, v_core, stᵥ)
    
    # Derivatives using central differences
    # ∂V/∂K
    V_Kp = v_net(state .+ [h; 0.0], θᵥ, v_core, stᵥ)
    V_Km = v_net(state .- [h; 0.0], θᵥ, v_core, stᵥ)
    dVdK = (V_Kp .- V_Km) ./ (2 * h)
    
    # ∂V/∂z  
    V_zp = v_net(state .+ [0.0; h], θᵥ, v_core, stᵥ)
    V_zm = v_net(state .- [0.0; h], θᵥ, v_core, stᵥ)
    dVdz = (V_zp .- V_zm) ./ (2 * h)
    
    # ∂²V/∂z²
    d2Vdz2 = (V_zp .- 2.0 .* V .+ V_zm) ./ (h^2)
    
    # Infinitesimal generator: LV = μK*∂V/∂K + μz*∂V/∂z + (1/2)*σz²*∂²V/∂z²
    LV = μK .* dVdK .+ μz .* dVdz .+ 0.5 * σz^2 .* d2Vdz2
    
    return V, LV, E, i_vals
end

#########################################################
### HJB Residual ###
#########################################################

function compute_hjb(state::AbstractMatrix, θᵥ, v_core, stᵥ, θπ, π_core, stπ)
    V, LV, E, i_vals = compute_drift(state, θᵥ, v_core, stᵥ, θπ, π_core, stπ)
    
    # HJB: 0 = E + LV - ρV  =>  hjb = E + LV - ρV
    hjb = E .+ LV .- ρ .* V
    
    return hjb, V, i_vals, E
end

#########################################################
### Loss Functions (matching Python) ###
#########################################################

# Policy evaluation: minimize |HJB|
function loss_value(state, θᵥ, v_core, stᵥ, θπ, π_core, stπ)
    hjb, _, _, _ = compute_hjb(state, θᵥ, v_core, stᵥ, θπ, π_core, stπ)
    return mean(abs, hjb)  # Python uses abs, not abs2
end

# Policy improvement: maximize HJB (minimize -HJB)
function loss_policy(state, θᵥ, v_core, stᵥ, θπ, π_core, stπ)
    hjb, _, _, _ = compute_hjb(state, θᵥ, v_core, stᵥ, θπ, π_core, stπ)
    return -mean(hjb)
end

#########################################################
### State Sampling (matching Python sample_fn) ###
#########################################################

function sample_states(rng, n::Int)
    # Sample K uniformly between k_min and k_max
    K = k_min .+ (k_max - k_min) .* rand(rng, 1, n)
    # Sample logz from stationary distribution N(0, std_z²)
    logz = std_z .* randn(rng, 1, n)
    return vcat(K, logz)
end

#########################################################
### Training (matching Python hyperparameters) ###
#########################################################

function train(;
    max_iter::Int = 100_000,
    batch_size::Int = 2048,
    lr_v::Float64 = 1e-3,
    lr_π::Float64 = 1e-4,
    seed::Int = 0
)
    rng = Xoshiro(seed)
    
    v_core = create_value_network()
    π_core = create_policy_network()
    
    θᵥ, stᵥ = Lux.setup(rng, v_core)
    θπ, stπ = Lux.setup(rng, π_core)
    
    # Convert to Float64 for consistency
    θᵥ = fmap(x -> x isa AbstractArray ? Float64.(x) : x, θᵥ)
    θπ = fmap(x -> x isa AbstractArray ? Float64.(x) : x, θπ)
    
    # Use Adam optimizer
    osᵥ = Optimisers.setup(Optimisers.Adam(lr_v), θᵥ)
    osπ = Optimisers.setup(Optimisers.Adam(lr_π), θπ)
    
    loss_v_hist = Float64[]
    loss_π_hist = Float64[]
    
    p = Progress(max_iter; desc="Training...", showspeed=true)
    
    for iter in 1:max_iter
        state = sample_states(rng, batch_size)
        
        # Policy evaluation step
        lv, gv = Zygote.withgradient(θᵥ) do p
            loss_value(state, p, v_core, stᵥ, θπ, π_core, stπ)
        end
        
        # Gradient clipping (matching Python: clip to [-1, 1])
        gv_clipped = fmap(x -> x isa AbstractArray ? clamp.(x, -1.0, 1.0) : x, first(gv))
        osᵥ, θᵥ = Optimisers.update(osᵥ, θᵥ, gv_clipped)
        
        # Policy improvement step
        lp, gp = Zygote.withgradient(θπ) do p
            loss_policy(state, θᵥ, v_core, stᵥ, p, π_core, stπ)
        end
        
        gp_clipped = fmap(x -> x isa AbstractArray ? clamp.(x, -1.0, 1.0) : x, first(gp))
        osπ, θπ = Optimisers.update(osπ, θπ, gp_clipped)
        
        push!(loss_v_hist, lv)
        push!(loss_π_hist, abs(lp))
        
        next!(p, showvalues=[(:iter, iter), (:loss_v, round(lv, digits=6))])
    end
    
    return (v_core=v_core, π_core=π_core, θᵥ=θᵥ, θπ=θπ, stᵥ=stᵥ, stπ=stπ,
            loss_v=loss_v_hist, loss_π=loss_π_hist)
end

#########################################################
### Plotting (matching Python figure style) ###
#########################################################

function plot_results(model; kmin_plot=7.5, kmax_plot=40.0)
    k_grid = collect(range(kmin_plot, kmax_plot, length=200))
    z_vals = [0.87, 1.0, 1.15]  # Matching Python
    
    p1 = plot(title="Investment Rate", xlabel="Capital", ylabel="Investment Rate")
    p2 = plot(title="Dividends", xlabel="Capital", ylabel="Dividends")
    p3 = plot(title="Value V(k,z)", xlabel="Capital", ylabel="V")
    
    styles = [:dot, :solid, :dashdot]
    colors = [:blue, :orange, :green]
    
    for (j, z) in enumerate(z_vals)
        logz = log(z)
        state = vcat(k_grid', logz .* ones(1, length(k_grid)))
        
        i_vals = vec(π_net(state, model.θπ, model.π_core, model.stπ))
        V_vals = vec(v_net(state, model.θᵥ, model.v_core, model.stᵥ))
        
        # Compute dividends normalized by K (E/K as in Python)
        K = state[1:1, :]
        _, _, _, E = compute_hjb(state, model.θᵥ, model.v_core, model.stᵥ,
                                  model.θπ, model.π_core, model.stπ)
        E_over_K = vec(E ./ K)
        
        plot!(p1, k_grid, i_vals, label="z=$(z)", lw=3, ls=styles[j], color=colors[j])
        plot!(p2, k_grid, E_over_K, label="z=$(z)", lw=3, ls=styles[j], color=colors[j])
        plot!(p3, k_grid, V_vals, label="z=$(z)", lw=2, ls=styles[j], color=colors[j])
    end
    
    # Set axis limits to match Python plots
    plot!(p1, ylim=(-0.75, 1.5), xlim=(5, 41))
    plot!(p2, ylim=(-1.25, 0.75), xlim=(5, 41))
    
    # Loss plot
    idx = min(100, length(model.loss_v))
    p4 = plot(model.loss_v[idx:end], yscale=:log10, label="Value Loss", 
              xlabel="Iteration", ylabel="Loss", title="Training Loss", lw=1.5)
    plot!(p4, model.loss_π[idx:end], label="Policy Loss", lw=1.5)
    
    plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
end

function diagnose(model)
    println("\n" * "="^60)
    println("DIAGNOSTICS")
    println("="^60)
    println("Parameters: δ=$δ, α=$α, λ=$λ, θ=$θ, σz=$σz, χ=$χ, ρ=$(round(ρ,digits=4))")
    println("State bounds: K ∈ [$(round(k_min,digits=2)), $(round(k_max,digits=2))], std_z=$(round(std_z,digits=4))")
    println()
    
    test_points = [(15.0, log(0.87)), (20.0, 0.0), (25.0, log(1.15)), (30.0, 0.0)]
    
    for (k, lz) in test_points
        state = reshape([k, lz], 2, 1)
        hjb, V, i, E = compute_hjb(state, model.θᵥ, model.v_core, model.stᵥ,
                                    model.θπ, model.π_core, model.stπ)
        z = exp(lz)
        println("K=$(round(k,digits=1)), z=$(round(z,digits=2)):")
        println("  i=$(round(i[1],digits=4)), V=$(round(V[1],digits=2)), E/K=$(round(E[1]/k,digits=4)), HJB=$(round(hjb[1],digits=6))")
    end
    
    println("\nFinal losses:")
    println("  Value loss: $(round(model.loss_v[end], digits=6))")
    println("  Policy loss: $(round(model.loss_π[end], digits=6))")
end

#########################################################
### Main ###
#########################################################

function main()
    println("Hennessy-Whited (2007) - DPI (Corrected)")
    println("="^60)
    
    model = train(max_iter=100_000, batch_size=2048, lr_v=1e-3, lr_π=1e-4)
    
    diagnose(model)
    
    plt = plot_results(model)
    savefig(plt, "hw_dpi.png")
    println("\nSaved: hw_dpi.png")
    
    return model
end


main()
l