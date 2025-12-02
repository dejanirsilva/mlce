# Initialize the parameters 
parameters, layer_states = Lux.setup(rng, model)

# Use Adam for optimization
learning_rate = 1e-3
opt = Adam(learning_rate)
opt_state = Optimisers.setup(opt, parameters)