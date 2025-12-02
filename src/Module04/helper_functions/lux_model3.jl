# Define the network architecture
model = Chain(
    Dense(1 => 2, Lux.relu), # first hidden layer
    Dense(2 => 2, Lux.relu), # second hidden layer
    Dense(2 => 2, Lux.relu), # third hidden layer
    Dense(2 => 1, identity)  # output layer
)