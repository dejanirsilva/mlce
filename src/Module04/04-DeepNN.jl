using Plots, LaTeXStrings, Lux, Random
pgfplotsx()
default(legend_font_halign=:left)

############################################################
# Deep neural networks as composition of SNN
############################################################
# Triangular map
τ(x) = 2 * max(0, x) -  4 * max(0, x -  0.5) 
# Compose τ L times: gives ~ 2^L oscillations (exponential pieces)
function triangle_comp(x::AbstractVector, L::Int)
    y = copy(x)
    @inbounds for _ in 1:L
        y .= τ.(y)
    end
    return y
end
xrange = collect(range(0.0, 1.0; length=1000))
p_triangle_comp = plot(xrange, triangle_comp(xrange, 1), line = 3, label = L"f_1(x)", xlabel = "x", ylabel = "y", title = "", foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(xrange, triangle_comp(xrange, 2), line = 3, label = L"f_2(x)",linestyle = :dash, alpha = 0.5)
plot!(xrange, triangle_comp(xrange, 3), line = 3, label = L"f_3(x)", linestyle = :dash, alpha = 0.5)

############################################################
# Deep neural networks in Lux
############################################################

# Shallow neural network in Lux
model = Chain(
    Dense(1 => 2, Lux.relu), # single input, two hidden units
    Dense(2 => 1, identity)  # two hidden units, one output
)

rng = Random.Xoshiro(123)
parameters, state = Lux.setup(rng, model)
parameters

parameters = (layer_1 = (weight = [1.0; 1.0;;], bias = [0.0, -0.5]), 
              layer_2 = (weight = [2.0 -4.0], bias = [0.0]))

xgrid = collect(range(0.0, 1.0, length=9))
ygrid = model(xgrid', parameters, state)[1]

plot(xgrid, ygrid', line = 3, xlabel = L"x", ylabel = L"f_1(x)", 
     title = "Shallow Neural Network", size = (400, 275), label = "")

# Deep neural network in Lux
model = Chain(
    Dense(1 => 2, Lux.relu),
    Dense(2 => 2, Lux.relu),
    Dense(2 => 1, identity)
)

parameters, state = Lux.setup(rng, model)
parameters = (layer_1 = (weight = [1.0; 1.0;;], bias = [0.0, -0.5]), 
              layer_2 = (weight = [2.0 -4.0; 2.0 -4.0], bias = [0.0, -0.5]),
              layer_3 = (weight = [2.0 -4.0], bias = [0.0]))
xgrid = collect(range(0.0, 1.0, length=9))
ygrid = model(xgrid', parameters, state)[1]'
plot(xgrid, ygrid, line = 3, xlabel = L"x", ylabel = L"f_2(x)", title = "Deep Neural Network", size = (400, 275), label = "")