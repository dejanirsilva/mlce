# Example usage: Gradient descent
opt = Optimisers.Descent(η)
θ_opt, stats = loss_optimiser(loss, θ0, opt, steps = 1000)