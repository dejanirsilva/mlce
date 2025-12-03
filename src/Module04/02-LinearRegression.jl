using Random, Plots, LaTeXStrings, Printf, LinearAlgebra, Statistics
pgfplotsx()
default(legend_font_halign=:left)

############################################################
### Include helper functions ###
############################################################
for f in ("ls_grad_descent.jl",)
    include(joinpath(@__DIR__,"helper_functions/", f))
end

############################################################
### Train a linear regression model ###
############################################################

@kwdef struct LinearModel
    weights::AbstractVector
    bias::Float64
    f::Function = x-> dot(weights, x) + bias
end

# Random number generator
rnd      = Random.MersenneTwister(5)

# Simulate data
m0       = LinearModel(weights=[1.46], bias=0.39)
x_sim    = randn(rnd, 100)
y_sim    = m0.f.(x_sim) + randn(rnd, 100) * 0.2    

# Loss function
loss(m::LinearModel) = 0.5*(m.f.(x_sim) - y_sim).^2 |> mean 
f(w, b)  = 0.5*(w * x_sim .+ b - y_sim).^2 |> mean
∇f(w, b) = [sum((w * x_sim .+ b - y_sim) .* x_sim), sum(w * x_sim .+ b - y_sim)]

wrange   = range(0.0,3.0,length=50)
brange   = range(-0.5,1.5,length=50)

losses   = f.(wrange, brange')
w_path, b_path = ls_grad_descent(y_sim, x_sim, [2.75, 1.35]; learning_rate=1.0, max_iter=10)

# base
gr()
colors = cgrad(:matter, 5, categorical = true)
heatmap(wrange, brange, losses; color=:ice, colorbar=true, size=(500*1.5, 500*1.5))
contour!(wrange, brange, losses; levels=12, linecolor=:grey, linewidth=1.2, alpha = 0.3)
xlims!(extrema(wrange)); ylims!(extrema(brange))
scatter!(w_path, b_path; color=colors[1], markerstrokecolor=:black, markerstrokewidth=0.8, markersize=7, label="", xlabel=L"w", ylabel=L"b", alpha = 0.9)
plot!(w_path, b_path; color=colors[1], markerstrokecolor=:black, markerstrokewidth=0.8, markersize=7, label="", xlabel=L"w", ylabel=L"b", alpha = 0.9)
scatter!([m0.weights], [m0.bias]; color=colors[3], markerstrokecolor=:black, markerstrokewidth=0.8, markersize=7, label="", xlabel=L"w", ylabel=L"b", alpha = 0.9)
pgfplotsx()

X     = [x_sim ones(length(x_sim))]
θ_hat = inv(X' * X) * X' * y_sim
hcat(θ_hat, [w_path[end], b_path[end]])

############################################################
### Train a linear regression model ###
############################################################

f(x)            = 1/(1+25*x^2) # true function
rng             = Random.MersenneTwister(123)
x_grid          = range(-1, 1, length=2000)
y_grid          = f.(x_grid) + randn(rng, length(x_grid)) * 0.1
x_range         = 1:250:length(x_grid) # training sample

train_sample    = x_grid[x_range]
test_sample     = x_grid[setdiff(1:length(x_grid), x_range)] # test sample

y_train_sample  = y_grid[x_range]
y_test_sample   = y_grid[setdiff(1:length(x_grid), x_range)]

I               = length(train_sample)
k_range         = 0:7 # polynomial degree range
X               = [x.^k for x in x_grid, k in k_range]
X_train         = [x.^k for x in train_sample, k in k_range]
X_test          = [x.^k for x in test_sample, k in k_range]

βs = [ inv(X_train[:, 1:k]' * X_train[:, 1:k]) * X_train[:, 1:k]' * y_train_sample for k in 2:(k_range[end]+1)]
y_hat_train = [X_train[:,1:k] * βs[k-1] for k in 2:(k_range[end]+1)]
y_hat_test = [X_test[:,1:k] * βs[k-1] for k in 2:(k_range[end]+1)]
train_losses = [0.5*mean((y_train_sample - y_hat_train[k]).^2) for k in 1:k_range[end]]
test_losses = [0.5*mean((y_test_sample - y_hat_test[k]).^2) for k in 1:k_range[end]]

j = 0
p1 = plot(k_range[2:end-j].+1, train_losses[1:end-j],
    label = "",
    xlabel = "Number of parameters",
    ylabel = "Training Loss",
    color = palette(:auto)[1],
    linewidth = 3,
    legend = false,
    foreground_color_legend = :transparent,
    background_color_legend = :transparent)
    annotate!(p1, 3.4, 0.029, Plots.text(L"\textbf{Training loss}"; color=palette(:auto)[1], halign=:left, pointsize=14))

p2 = twinx(p1)
plot!(p2,
    k_range[2:end-j].+1, test_losses[1:end-j],
    label = "",
    ylabel = "Test Loss",
    color = palette(:auto)[2],
    linewidth = 3,
    linestyle = :dash, 
    legend = false,
    foreground_color_legend = :transparent,
    background_color_legend = :transparent)  # disable the second legend
# Inline annotations matching series colors
xvals = k_range[2:end-j]
x_ann = xvals[end] - 0.3
y_ann_train = last(train_losses[1:end-j])
y_ann_test  = last(test_losses[1:end-j])
annotate!(p2, 4.0, 0.07, Plots.text(L"\textbf{Test loss}"; color=palette(:auto)[2], halign=:left, pointsize=14))