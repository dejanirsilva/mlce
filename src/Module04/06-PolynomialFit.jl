using Random, Plots, LaTeXStrings, Statistics, LinearAlgebra
using Optimisers, Zygote, Roots, Lux, Distributions, ProgressMeter
pgfplotsx()
default(legend_font_halign=:left)

############################################################
# Polynomial fit with DNN
############################################################

# Constructing function f
rng         = Xoshiro(0)           # pseudo random number Generator
roots       = randn(rng, 5)        # polynomial roots
p(x)        = prod(x .- roots)     # univariate polynomial
f(x)        = mean(p.(x))          # multivariate version

# Random samples
n_states    = 10
sample_size = 100_000
x_samples   = rand(rng, Uniform(-1,1), (n_states,sample_size))
y_samples   = [f(x_samples[:,i]) for i = 1:sample_size]'

# Visualization
x_range     = -1:0.01:1
plot(x_range, [f(repeat([x],n_states)) for x in x_range], 
    linewidth = 4, label = "exact")

# Archictecture
layers = [n_states, 32, 32, 1]
model = Chain(
    Dense(layers[1] => layers[2], Lux.gelu),
    Dense(layers[2] => layers[3], Lux.gelu),
    Dense(layers[3] => layers[4], identity)
)

# Initialize the parameters 
parameters, layer_states = Lux.setup(rng, model) |> f64

# Checking the initial prediction
y_initial_prediction, _ = model(repeat(x_range',n_states), parameters, layer_states)
plot!(x_range, y_initial_prediction', linewidth = 4, label = "initial prediction")

# Loss function
batch_size = 128
function loss_fn(p, ls; batch_size = 128)
    ind = rand(rng,1:sample_size,batch_size)
    y_prediction, new_ls = model(x_samples[:,ind], p, ls)
    loss = 0.5 * mean( (y_prediction-y_samples[:,ind]).^2 )
    return loss, new_ls
end
ind = rand(rng,1:sample_size,batch_size)
y_prediction, new_ls = model(x_samples[:,ind], parameters, layer_states)

# Use Adam for optimization
learning_rate = 1e-3
opt = Adam(learning_rate)
opt_state = Optimisers.setup(opt, parameters)

# train loop
loss_history = []
max_iter = 300_000

pbar = Progress(max_iter; desc="Training...", dt=1.0) #progress bar
for i = 1:max_iter
    loss, layer_states = loss_fn(parameters, layer_states)
    grad = gradient(p->loss_fn(p, layer_states)[1], parameters)[1]
    opt_state, parameters = Optimisers.update(opt_state, parameters, grad)
    push!(loss_history, loss)
    next!(pbar, showvalues = [(:iter, i),("Loss", loss)])
end

# Training history
plot(loss_history, yscale = :log10, linewidth = 1.5)

# Scatter plot with prediction
y_prediction, layer_states = model(x_samples, parameters, layer_states) 
ind = rand(rng,1:sample_size,batch_size*2)
p1 = plot_scatter = scatter(x_samples[1,ind], y_samples[1,ind],label="data", xlabel = L"x_1", ylabel = L"y", legend = :topright,
    foreground_color_legend=:transparent, background_color_legend = :transparent, alpha = 0.85)
scatter!(x_samples[1,ind],y_prediction[1,ind],label="final prediction", alpha = 0.85)

# Function
y_exact = [f(repeat([x],n_states)) for x in x_range]
y_model = [model(repeat([x],n_states), parameters, layer_states)[1][1] for x in x_range]

p2 = plot_function = plot(x_range[5:end-5], y_exact[5:end-5], linewidth = 4, label = "exact", xlabel = L"x_1", ylabel = L"y", legend = :topright,
    foreground_color_legend=:transparent, background_color_legend = :transparent, alpha = 1.0)
plot!(x_range[5:end-5], y_model[5:end-5], linewidth = 3, linestyle = :dash, label = "neural network", alpha = 1.0)

plot(p1, p2, layout = (1,2), size = (800, 300))