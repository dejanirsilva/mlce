using ReverseDiff      # For the outer parameter gradient
using ForwardDiff      # For the inner spatial gradient
using Lux              # For the neural network
using ComponentArrays  # For parameter structure and flattening
using LinearAlgebra    # for dot()
using Random
using Statistics

# ----------------------------------------------------------------------
# 1. Setup: Define a Simple Mock Network and State
# ----------------------------------------------------------------------
rng = Random.default_rng()
# Simple network: 2 inputs, 1 output, tanh activation
model = Lux.Chain(
    Lux.Dense(2, 32, gelu),
    Lux.Dense(32, 32, gelu),
    Lux.Dense(32, 1, identity)
)
ps_test, st = Lux.setup(rng, model)
ps_test_ca = ComponentArray(ps_test)
X = [1 2; 3 4]

NN_model       = Lux.Dense(2, 1, tanh) 
ps_untyped, st = Lux.setup(rng, NN_model)
ps_init        = ComponentArrays.ComponentArray(ps_untyped)
collect(ps_untyped)

# Mock state variable 's' (our HJB state)
s = Float32[1.0, 0.5]

# Function to UNRAVEL ComponentArray to its flat vector
# We will use this to feed parameters into ReverseDiff.
ps_vector, ps_reconstruct = ComponentArrays.getdata(ps_init)


# ----------------------------------------------------------------------
# 2. Inner Derivative Calculation (ForwardDiff)
# ----------------------------------------------------------------------
"""
    compute_inner_term_forward(ps_vec, s, st, NN_model, ps_reconstruct)

Computes the inner term (s ⋅ ∇s V). The input 'ps_vec' is unwrapped 
to its value before ForwardDiff is called, but ReverseDiff's Tape 
tracks the dependency.
"""
function compute_inner_term_forward(ps_vec, s, st, NN_model, ps_reconstruct)
    # 1. Reconstruct the ComponentArray structure from the flat vector
    # This step is critical for Lux to use the parameters.
    ps_structured = ps_reconstruct(ps_vec)
    
    # 2. Define the V(s) closure using the structured parameters
    V(s_val) = NN_model(s_val, ps_structured, st)[1][1]
    
    # 3. ForwardDiff computes the gradient w.r.t. the state 's'
    # ReverseDiff is robust enough to handle this nested ForwardDiff call
    # when the parameters 'ps_vec' are tracked by its Tape.
    ∇V = ForwardDiff.gradient(V, s)
    
    return dot(s, ∇V) 
end

# ----------------------------------------------------------------------
# 3. Loss Functions and Tape Setup
# ----------------------------------------------------------------------

# The loss function that ReverseDiff will tape and differentiate
function loss_function(ps_vec)
    # 1. Calculate the inner term (our mock HJB residual part)
    inner_term = compute_inner_term_forward(ps_vec, s, st, NN_model, ps_reconstruct)
    
    # 2. Calculate the raw network output V(s)
    ps_structured = ps_reconstruct(ps_vec)
    V_output = NN_model(s, ps_structured, st)[1][1]

    # 3. Combine them into a simple loss (Loss = V^2 + (InnerTerm)^2)
    return V_output^2 + inner_term^2
end

# The V-only loss function for comparison
function loss_function_V_only(ps_vec)
    ps_structured = ps_reconstruct(ps_vec)
    V_output = NN_model(s, ps_structured, st)[1][1]
    return V_output^2 
end

# 4. Tape Preparation
# ReverseDiff tapes the function once for efficiency.
tape_full = ReverseDiff.GradientTape(loss_function, (ps_vector,))
tape_V_only = ReverseDiff.GradientTape(loss_function_V_only, (ps_vector,))

# Compile the tapes for maximum performance (optional but recommended)
compiled_tape_full = ReverseDiff.compile(tape_full)
compiled_tape_V_only = ReverseDiff.compile(tape_V_only)

# ----------------------------------------------------------------------
# 5. Verification: Compute and Compare Gradients
# ----------------------------------------------------------------------

# Compute the gradient for Loss 1 (full, correct)
∇_params_loss_1_vec = ReverseDiff.gradient!(compiled_tape_full, (ps_vector,))[1]
∇_params_loss_1 = ps_reconstruct(∇_params_loss_1_vec) # Convert back to structured form

# Compute the gradient for Loss 2 (V-only)
∇_params_loss_2_vec = ReverseDiff.gradient!(compiled_tape_V_only, (ps_vector,))[1]
∇_params_loss_2 = ps_reconstruct(∇_params_loss_2_vec) # Convert back to structured form

# Check the difference for a key parameter (e.g., the first weight)
grad_diff = abs(∇_params_loss_1.weight[1] - ∇_params_loss_2.weight[1])

println("--- ReverseDiff Mixed-Mode AD Verification ---")
println("Loss 1 (Full) Gradient (Weight 1): $(round(∇_params_loss_1.weight[1]; digits=5))")
println("Loss 2 (V-only) Gradient (Weight 1): $(round(∇_params_loss_2.weight[1]; digits=5))")
println("\nAbsolute Difference in Gradients: $(round(grad_diff; digits=5))")

is_differentiated = grad_diff > 1e-6

println("AD Flow Status: Gradient of InnerTerm (InnerTerm^2) is correctly traced? $(is_differentiated ? "✅ Yes, they are different!" : "❌ No, they are the same.")")



c = (a = 2, b = [1, 2]);
c
x = ComponentArray(a = 1.0, b = [2, 1, 4], c = c)
x.c.a = 400;
x
x[5]
collect(x)
typeof(similar(x, Int32)) === typeof(ComponentVector{Int32}(a = 1, b = [2, 1, 4], c = c))