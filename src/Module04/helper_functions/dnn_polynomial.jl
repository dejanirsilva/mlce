# Constructing function f
rng  = Xoshiro(0)           # pseudo random number generator
roots = randn(rng, 5)       # polynomial roots
p(x) = prod(x .- roots)     # univariate polynomial
f(x) = mean(p.(x))          # multivariate version

# Random samples
n_states, sample_size = 10, 100_000
x_samples = rand(rng, Uniform(-1,1), (n_states,sample_size))
y_samples = [f(x_samples[:,i]) for i = 1:sample_size]'