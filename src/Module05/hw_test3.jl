using LinearAlgebra
using ForwardDiff      # For HJB derivatives (∇V and H)
using Lux              # For the neural network architecture
using Random           # For RNG management
using Zygote           # For the outer optimization gradient (and Zygote.ignore_derivatives)
using Optimisers       # For Adam optimizer and updates
using ComponentArrays  # For parameter handling (like JAX/PyTree)
using Statistics       # For mean()
using NNlib            # For sigmoid function
using ProgressMeter    # New addition for progress tracking

# --- 1. Parameters and Configuration ---

const DType = Float32

# Model Parameters (now Float32)
const δ = DType(0.1)
const α = DType(0.55)
const λ = DType(0.059)
const θ = DType(0.26)
const σz = DType(0.123)
const χ = DType(0.1)
const ρ = DType(-log(0.96))

# Training Configuration
const BATCH_SIZE = 512 * 4  # 2048
const LR = DType(1e-3)
const LR_DECAY = DType(0.99)
const EPOCHS = 5
const ITERATIONS_PER_EPOCH = 15000
const LOG_INTERVAL = ITERATIONS_PER_EPOCH ÷ 100 # Calculate/log loss 100 times per epoch

# Swish activation function (silu in JAX)
swish(x) = x * sigmoid(x)

# ----------------------------------------------------------------------
# --- 2. Neural Network Structure (LayerNorm Removed for Stability) ---
# ----------------------------------------------------------------------

function mlp(rng::AbstractRNG, n_output::Int)
    model = Lux.Chain(
        # Layer 1
        Lux.Dense(2, 256, swish), 
        
        # Layer 2
        Lux.Dense(256, 128, swish), 
        
        # Layer 3
        Lux.Dense(128, 64, swish),
        
        # Output Layer
        Lux.Dense(64, n_output)
    )
    
    p_untyped, st = Lux.setup(rng, model)
    ps = ComponentArrays.ComponentArray(p_untyped)
    
    return model, ps, st
end

# ----------------------------------------------------------------------
# --- 3. Model Dynamics (Shared Functions) ---
# ----------------------------------------------------------------------

# The state vector s is [k, z]
function μ(s::AbstractVector, i::Number)
    k, z = s[1], s[2]
    μ_k = (i - δ) * k
    μ_z = -θ * (z - exp(DType(0.0))) 
    return [μ_k, μ_z]
end

function σ(s::AbstractVector)
    return [DType(0.0); σz]
end

function dividends(s::AbstractVector, i::Number)
    k, z = s[1], s[2]
    Y = z * k^α
    
    cash_flow = Y - i * k - χ * (i - δ)^2 * k
    
    issuance_cost = λ * max(DType(0.0), -cash_flow)
    
    return cash_flow - issuance_cost
end

# ----------------------------------------------------------------------
# --- 4. HJB Operator (Ito's Lemma - Corrected for Nested AD) ---
# ----------------------------------------------------------------------

function EdV_dt(V_nn, ps, st, s::AbstractVector, i::Number)
    
    # Detach parameters (ps, st) from Zygote tracking for the inner ForwardDiff call
    ps_ignore = Zygote.ignore_derivatives(ps)
    st_ignore = Zygote.ignore_derivatives(st)
    
    # V(s_val) closure uses the detached parameters
    V(s_val) = V_nn(reshape(s_val, 2, 1), ps_ignore, st_ignore)[1][1] 

    μ_s = μ(s, i)
    σ_s = σ(s)

    # First-order term: μ(s) ⋅ ∇V(s)
    ∇V = ForwardDiff.gradient(V, s)
    first_order_term = dot(μ_s, ∇V)

    # Second-order term: 0.5 * Tr(ΣΣᵀ * H(s))
    H = ForwardDiff.hessian(V, s)
    ΣΣᵀ = σ_s * transpose(σ_s)
    second_order_term = DType(0.5) * tr(ΣΣᵀ * H)

    return first_order_term + second_order_term
end

# ----------------------------------------------------------------------
# --- 5. Loss Functions (Unchanged) ---
# ----------------------------------------------------------------------

function hjb_loss(V_nn, ps_V, st_V, π_nn, ps_π, st_π, s::AbstractVector)
    # Reshape 1D vector 's' to 2x1 matrix 's_mat' for Lux (features x batch)
    s_mat = reshape(s, 2, 1)

    # Policy network evaluation
    i = π_nn(s_mat, ps_π, st_π)[1][1] 

    EdV = EdV_dt(V_nn, ps_V, st_V, s, i)
    d = dividends(s, i)
    
    # Value network evaluation
    V = V_nn(s_mat, ps_V, st_V)[1][1]

    HJB_RHS = d + EdV
    hjb_residual = ρ * V - HJB_RHS
    
    return hjb_residual^2
end

function policy_loss(V_nn, ps_V, st_V, π_nn, ps_π, st_π, s::AbstractVector)
    # Reshape 1D vector 's' to 2x1 matrix 's_mat' for Lux (features x batch)
    s_mat = reshape(s, 2, 1)

    i = π_nn(s_mat, ps_π, st_π)[1][1]

    EdV = EdV_dt(V_nn, ps_V, st_V, s, i)
    d = dividends(s, i)
    
    objective = d + EdV
    
    return -objective # Minimize negative objective
end

# ----------------------------------------------------------------------
# --- 6. Initialization and Sampling (Unchanged) ---
# ----------------------------------------------------------------------

function initialize_training(seed::Int=42)
    rng = Random.default_rng() 
    Random.seed!(rng, seed)
    
    # Initialize Networks
    V_model, ps_V, st_V = mlp(rng, 1)
    π_model, ps_π, st_π = mlp(rng, 1)

    # Initialize Optimizers
    opt = Optimisers.Adam(LR)
    opt_st_V = Optimisers.setup(opt, ps_V)
    opt_st_π = Optimisers.setup(opt, ps_π)
    
    rng_sample = rng 

    return V_model, ps_V, st_V, opt_st_V, π_model, ps_π, st_π, opt_st_π, rng_sample
end

function sample_state(rng::AbstractRNG, n::Int)
    
    # State Bounds
    k_min, k_max = DType(0.5), DType(30.0)
    logz_std_approx = σz / sqrt(DType(2.0) * θ)
    logz_min = -DType(3.0) * logz_std_approx
    logz_max = DType(3.0) * logz_std_approx
    
    # Uniform sampling, explicitly using DType
    k_batch = rand(rng, DType, 1, n) .* (k_max - k_min) .+ k_min
    logz_batch = rand(rng, DType, 1, n) .* (logz_max - logz_min) .+ logz_min
    
    z_batch = exp.(logz_batch)

    # Stack the state vectors: s = [k; z] (2 x n matrix)
    s_batch = vcat(k_batch, z_batch)
    
    return s_batch, rng 
end

# ----------------------------------------------------------------------
# --- 7. Training Step (Unchanged) ---
# ----------------------------------------------------------------------

function train_step(V_model, ps_V, st_V, opt_st_V, π_model, ps_π, st_π, opt_st_π, s_batch::AbstractMatrix)
    
    # 1. Train the Value Network (Policy Evaluation)
    function loss_V(p)
        mean([hjb_loss(V_model, p, st_V, π_model, ps_π, st_π, s) for s in eachcol(s_batch)])
    end
    
    V_loss, V_grads = Zygote.withgradient(loss_V, ps_V)
    opt_st_V, ps_V_new = Optimisers.update!(opt_st_V, ps_V, V_grads[1])
    
    # 2. Train the Policy Network (Policy Improvement)
    function loss_π(p)
        mean([policy_loss(V_model, ps_V_new, st_V, π_model, p, st_π, s) for s in eachcol(s_batch)])
    end
    
    π_loss, π_grads = Zygote.withgradient(loss_π, ps_π)
    opt_st_π, ps_π_new = Optimisers.update!(opt_st_π, ps_π, π_grads[1])
    
    return ps_V_new, opt_st_V, ps_π_new, opt_st_π
end

# ----------------------------------------------------------------------
# --- 8. Main Training Loop (FIXED: ProgressMeter Added) ---
# ----------------------------------------------------------------------

function run_training()
    V_model, ps_V, st_V, opt_st_V, π_model, ps_π, st_π, opt_st_π, rng_sample = initialize_training(42)
    
    println("Starting DPI Training for $(EPOCHS) epochs...")
    
    # Training Loop
    for epoch in 1:EPOCHS
        # Apply Learning Rate Decay
        current_lr = LR * (LR_DECAY)^(epoch - 1)
        
        # Re-initialize optimizers with the new learning rate
        V_opt_init = Optimisers.Adam(current_lr)
        π_opt_init = Optimisers.Adam(current_lr)
        
        opt_st_V = Optimisers.setup(V_opt_init, ps_V)
        opt_st_π = Optimisers.setup(π_opt_init, ps_π)
        
        println("\n  Epoch $epoch: LR = $(round(current_lr; digits=5))")
        
        # Initialize Progress Bar
        p = Progress(ITERATIONS_PER_EPOCH, "  Training Epoch $epoch:")
        
        for iter in 1:ITERATIONS_PER_EPOCH
            # 1. Sample a new batch of states
            s_batch, rng_sample = sample_state(rng_sample, BATCH_SIZE)
            
            # 2. Perform the training step
            ps_V, opt_st_V, ps_π, opt_st_π = train_step(
                V_model, ps_V, st_V, opt_st_V, 
                π_model, ps_π, st_π, opt_st_π, 
                s_batch
            )

            # Logging and Progress Bar Update
            if iter % LOG_INTERVAL == 0
                # Calculate current losses over the batch
                V_loss_curr = mean([hjb_loss(V_model, ps_V, st_V, π_model, ps_π, st_π, s) for s in eachcol(s_batch)])
                π_loss_curr = mean([policy_loss(V_model, ps_V, st_V, π_model, ps_π, st_π, s) for s in eachcol(s_batch)])
                
                # Update progress bar with loss values
                next!(p; showvalues=[
                    (:V_Loss, round(V_loss_curr, digits=5)),
                    (:π_Loss, round(π_loss_curr, digits=5))
                ])
            else
                # Advance the progress bar iteration count
                next!(p) 
            end
        end
        
        finish!(p) # Ensure the progress bar closes cleanly
        println("Epoch $epoch complete.")
    end
    
    println("\nTraining finished.")
    return V_model, ps_V, st_V, π_model, ps_π, st_π
end