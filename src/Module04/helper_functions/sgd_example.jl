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