using Random, Plots, LaTeXStrings, Printf, LinearAlgebra, Statistics
using Distributions, Lux, Optimisers, Zygote
pgfplotsx()
default(legend_font_halign=:left)

############################################################
### Include helper functions ###
############################################################
for f in ("shallow_nn.jl",)
    include(joinpath(@__DIR__,"helper_functions/", f))
end

############################################################
# Activation functions
############################################################

# Activation functions
x           = range(-3, 3, length=100)
relu(x)     = max(0, x)
sigmoid(x)  = 1/(1 + exp(-x))
silu(x)     = x * sigmoid(x)
d           = Normal(0, 1)
gelu(x)     = x * cdf(d, x)

p1 = plot(x, relu.(x), label="", xlabel="x", ylabel="Activation function", linewidth=3, ylims = (-1.25,3), color = palette(:auto)[1],
    ls=[:solid :solid], title="ReLU", foreground_color_legend=:transparent, background_color_legend = :transparent, legend = :topleft)
p2 = plot(x, gelu.(x), label="", xlabel="x", ylabel="Activation function", linewidth=3, ylims = (-1.25,3), color = palette(:auto)[2],
    ls=[:solid :solid], title="GELU", foreground_color_legend=:transparent, background_color_legend = :transparent, legend = :topleft)
p3 = plot(x, tanh.(x), label="", xlabel="x", ylabel="Activation function", linewidth=3, ylims = (-1.25,3), color = palette(:auto)[3],
    ls=[:solid :solid], title="tanh", foreground_color_legend=:transparent,  background_color_legend = :transparent, legend = :topleft)
p4 = plot(x, silu.(x), label="", xlabel="x", ylabel="Activation function", linewidth=3, ylims = (-1.25,3), color = palette(:auto)[4],
    ls=[:solid :solid], title="SiLU", foreground_color_legend=:transparent, background_color_legend = :transparent, legend = :topleft)
plot(p1, p2, p3, p4, layout = (2, 2), size = (800, 600))

############################################################
# Shallow neural networks
############################################################

# Example of a shallow network
x̂ = [0.0,0.25,0.5,0.75] # breakpoints

rnd    = Random.MersenneTwister(4) # random seed 
wₙ, bₙ  = randn(rnd, length(x̂)), randn(rnd, 1)[1]
W, b   = ones(length(wₙ)), collect(-x̂)
shallow_nn(0.5, W, b, wₙ, bₙ)

# Evaluate once to get y-limits
xgrid = range(0, 1.0, length=100)
ygrid = shallow_nn.(xgrid, [W], [b], [wₙ], [bₙ])
plot(xgrid, ygrid, label="", xlabel=L"x", ylabel=L"f(x)", linewidth=3, color = palette(:auto)[1],
        ls=[:solid :solid], title="Shallow network", foreground_color_legend=:transparent, background_color_legend = :transparent, legend = :topleft)
    vspan!([x̂[2], x̂[3]]; color = :gray, alpha = 0.18, label = "")
    vspan!([last(x̂), 1.0]; color = :gray, alpha = 0.18, label = "")

############################################################
# Adaptive Choice of Breakpoints
############################################################

# 1) Data: a skewed / nonlinear target on [0,1]
α, β = 2, 10.0              # choose something quite nonlinear near 0
dist = Beta(α, β)
N = 512
x = range(0.0, 1.0; length=N) |> collect
y = pdf.(dist, x)
plot(x, y, line = 3.0, xlabel = L"x", ylabel = L"y", 
    label = "Beta pdf (α=$(α), β=$(β))", legend = :topright, 
    foreground_color_legend=:transparent, background_color_legend = :transparent)

# 2) Training function
function train_snn(model::Chain, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; 
        rng::Random.AbstractRNG = Random.default_rng(), opt = Optimisers.Adam(1e-2), 
        nepochs::Int64 = 4000)
    # 3) Init params & state
    ps, st = Lux.setup(rng, model)

    # 4) Loss (MSE). Model expects inputs as (in, batch)
    xmat = reshape(x, 1, :)
    ymat = reshape(y, 1, :)    
    # Helper functions
    function loss(ps, st, xmat, ymat)
        ŷ, _ = model(xmat, ps, st)
        mean(abs2, ŷ .- ymat)
    end
    # 5) Optimizer and training loop
    opt_state = Optimisers.setup(opt, ps)

    for epoch in 1:nepochs
        grads = Zygote.gradient(p -> loss(p, st, xmat, ymat), ps)
        opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        # (st is unused/unchanged here; no BN/RNN)
    end
    return ps, st, xmat, ymat
end

# 3) Extract breakpoints and "kink strengths"
function breakpoints_and_strengths(ps)
    # For the first Dense layer:
    # weight: H×1, bias: H, second-layer weight: 1×H
    W1 = ps.layer_1.weight[:, 1]             # length H
    b1 = ps.layer_1.bias                     # length H
    W2 = vec(ps.layer_2.weight)              # length H (since it's 1×H)

    # Breakpoints (where w1*x + b1 = 0)
    bp = -b1 ./ W1

    # Strength of slope jump at the kink contributed by neuron j
    Δslope = W2 .* W1

    # Keep finite, in-domain breakpoints
    mask = isfinite.(bp) .& (bp .>= 0) .& (bp .<= 1)
    return bp[mask], Δslope[mask[:]]
end

# 4) Data for plots
function data_for_plots(H::Int64, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; 
    rng::Random.AbstractRNG = Random.default_rng(), nepochs::Int64 = 20000)
    model = Chain(
        Dense(1 => H, relu),     # hidden kinks live here
        Dense(H => 1)
    )

    ps, st, xmat, ymat = train_snn(model, x, y, rng = rng, nepochs = nepochs)
    ŷ, _ = model(xmat, ps, st)
    ŷ = vec(ŷ)
    bp, Δs = breakpoints_and_strengths(ps)
    return ŷ, bp, model
end

# 5) Plots
rng = Random.MersenneTwister(49)
ŷ1, bp1, model1 = data_for_plots(15, x, y, rng = rng, nepochs = 20_000)
ŷ2, bp2, model2 = data_for_plots(30, x, y, rng = rng, nepochs = 20_000)
ŷ3, bp3, model3 = data_for_plots(100, x, y, rng = rng, nepochs = 20_000)

plt1a = plot(x, y, lw=3, label="Beta pdf (α=$(α), β=$(β))",
            title="Fit of 1-hidden-layer ReLU net to Beta pdf", 
            legend = :topright, xlabel = L"x", ylabel = L"y",
            foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(x, ŷ1, lw=2.0, label="SNN fit (H=15)", linestyle = :dash)

plt1b = histogram(bp1, bins=0:0.025:1,
                 xlabel=L"x", ylabel="count", alpha = 0.8,
                 title="Histogram of ReLU breakpoints in [0,1]",
                 label="breakpoints", legend = :topright,
                 foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(twinx(plt1b), x, y, line = 3.0, label="", ylabel = "Beta pdf", linestyle = :dash, color = palette(:auto)[2])

plt2a = plot(x, y, lw=3, label="Beta pdf (α=$(α), β=$(β))",
            legend = :topright, xlabel = L"x", ylabel = L"y",
            foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(x, ŷ2, lw=2.0, label="SNN fit (H=30)", linestyle = :dash)

plt2b = histogram(bp2, bins=0:0.025:1,
                 xlabel=L"x", ylabel="count", alpha = 0.8,
                 label="breakpoints", legend = :topright,
                 foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(twinx(plt2b), x, y, line = 3.0, label="", ylabel = "Beta pdf", linestyle = :dash, color = palette(:auto)[2])

plt3a = plot(x, y, lw=3, label="Beta pdf (α=$(α), β=$(β))",
            legend = :topright, xlabel = L"x", ylabel = L"y",
            foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(x, ŷ3, lw=2.0, label="SNN fit (H=100)", linestyle = :dash)

plt3b = histogram(bp3, bins=0:0.025:1,
                 xlabel=L"x", ylabel="count", alpha = 0.8,
                 label="breakpoints", legend = :topright,
                 foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(twinx(plt3b), x, y, line = 3.0, label="", ylabel = "Beta pdf", linestyle = :dash, color = palette(:auto)[2])


plot(plt1a, plt1b, plt2a, plt2b, plt3a, plt3b, layout = (3, 2), size = (1200, 1200))

