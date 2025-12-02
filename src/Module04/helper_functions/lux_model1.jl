# Load packages
using Lux, Random

# Define the network architecture
model = Chain(
    Dense(1 => 2, Lux.relu), # first hidden layer
    Dense(2 => 1, identity)  # output layer
)

# Initialize the parameters
rng = Random.Xoshiro(123)
parameters, state = Lux.setup(rng, model)