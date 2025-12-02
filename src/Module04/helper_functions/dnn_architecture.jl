# Archictecture
layers = [n_states, 32, 32, 1]
model = Chain(
    Dense(layers[1] => layers[2], Lux.gelu),
    Dense(layers[2] => layers[3], Lux.gelu),
    Dense(layers[3] => layers[4], identity)
)