using Random, Plots, LaTeXStrings, Printf, LinearAlgebra, Statistics

############################################################
# Shallow neural networks
############################################################

# Activation functions
x = range(-3, 3, length=100)
relu(x) = max(0, x)
sigmoid(x) = 1/(1 + exp(-x))
silu(x) = x * sigmoid(x)
d = Normal(0, 1)
gelu(x) = x * cdf(d, x)

p1 = plot(x, relu.(x), label="", xlabel="x", ylabel="Activation function", linewidth=3, ylims = (-1.25,3), color = palette(:auto)[1],
    ls=[:solid :solid], foreground_color_legend=:transparent, background_color_legend = :transparent, legend = :topleft)
savefig("lecture_notes/figures/chapter04/activation_functions1.pdf")
p2 = plot(x, gelu.(x), label="", xlabel="x", ylabel="Activation function", linewidth=3, ylims = (-1.25,3), color = palette(:auto)[2],
    ls=[:solid :solid], foreground_color_legend=:transparent, background_color_legend = :transparent, legend = :topleft)
savefig("lecture_notes/figures/chapter04/activation_functions2.pdf")
p3 = plot(x, tanh.(x), label="", xlabel="x", ylabel="Activation function", linewidth=3, ylims = (-1.25,3), color = palette(:auto)[3],
    ls=[:solid :solid], foreground_color_legend=:transparent,  background_color_legend = :transparent, legend = :topleft)
savefig("lecture_notes/figures/chapter04/activation_functions3.pdf")
p4 = plot(x, silu.(x), label="", xlabel="x", ylabel="Activation function", linewidth=3, ylims = (-1.25,3), color = palette(:auto)[4],
    ls=[:solid :solid], foreground_color_legend=:transparent, background_color_legend = :transparent, legend = :topleft)
savefig("lecture_notes/figures/chapter04/activation_functions4.pdf")

# A shallow network
include("listings/shallow_nn.jl")

# Test the multiple input shallow network
W, b = ones(4, 2), randn(4)
wₙ, bₙ = randn(4), randn(1)[1]
shallow_nn([0.1, 0.9], W, b, wₙ, bₙ)

# Example of a shallow network
x̂ = [0.0,0.25,0.5,0.75] # breakpoints

rnd = Random.MersenneTwister(4) # random seed
wₙ, bₙ = randn(rnd, length(x̂)), randn(rnd, 1)[1]
W, b = ones(length(wₙ)), collect(-x̂)
shallow_nn(0.5, W, b, wₙ, bₙ)

# Evaluate once to get y-limits
xgrid = range(0, 1.0, length=100)
ygrid = shallow_nn.(xgrid, [W], [b], [wₙ], [bₙ])
plot(xgrid, ygrid, label="", xlabel=L"x", ylabel=L"f(x)", linewidth=3, color = palette(:auto)[1],
    ls=[:solid :solid], foreground_color_legend=:transparent, background_color_legend = :transparent, legend = :topleft)
vspan!([x̂[2], x̂[3]]; color = :gray, alpha = 0.18, label = "")
vspan!([last(x̂), 1.0]; color = :gray, alpha = 0.18, label = "")
savefig("lecture_notes/figures/chapter04/shallow_network.pdf")

