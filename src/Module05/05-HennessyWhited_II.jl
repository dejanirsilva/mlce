#########################################################
# Hennessy and Whited (2007)
#########################################################
using Plots, LaTeXStrings, Distributions, LinearAlgebra, BenchmarkTools
using Lux, Optimisers, ForwardDiff, Zygote, Random, ProgressMeter
pgfplotsx()
default(legend_font_halign=:left)

#########################################################
### Model struct ###
#########################################################
@kwdef struct HennessyWhited
    α::Float64 = 0.55
    θ::Float64 = 0.26
    z̅::Float64 = 0.0
    σz::Float64 = 0.123
    δ::Float64 = 0.1
    χ::Float64 = 0.1
    λ::Float64 = 0.059
    ρ::Float64 = -log(0.96)
    μₛ::Function = (s,i) -> vcat((i .- δ) .* s[1,:]', -θ .* (s[2,:] .- z̅)')
    σₛ::Function = (s,i) -> vcat(zeros(1,size(s,2)), σz*ones(1,size(s,2)))
end;

#########################################################
### Define the neural network ###
#########################################################
v_core = Chain(
    Dense(2, 64, Lux.swish),
    Dense(64, 32, Lux.swish),
    Dense(32, 16, Lux.swish),
    Dense(16, 1)
)
v_net(s, θᵥ) =  v_core(s, θᵥ, stᵥ)[1]
i_core = Chain(
    Dense(2, 64, Lux.relu),
    Dense(64, 32, Lux.relu),
    Dense(32, 32, Lux.relu),
    Dense(32, 1)
)

i_net(s, θᵢ) = i_core(s, θᵢ, stᵢ)[1]

#########################################################
function hjb_residuals(s, m, θᵥ, θᵢ; h = 1e-3)
    i̅           = i_net(s, θᵢ)
    k           = @view s[1,:]
    z           = @view s[2,:]
    revenue     = (exp.(z) .* k.^m.α)'         
    D_star      = revenue .- (i̅ + 0.5 * m.χ * (i̅ ).^2).* k'
    D           = D_star .* (1 .+ m.λ * (D_star .< 0))
    μₛ, σₛ       = m.μₛ(s,i̅), m.σₛ(s,i̅)
    F(ϵ)        = v_net(s .+ σₛ .* (ϵ / sqrt(2.0)) .+ μₛ .* (ϵ^2 / 2.0), θᵥ)
	# drift       = (F(h) - 2.0 * F(0.0) + F(-h)) / (h*h)
    drift       = (-F(2*h) + 16*F(h)- 30.0 * F(0.0)+16*F(-h) - F(-2*h)) / (12*h*h)
    return D + drift - m.ρ * v_net(s, θᵥ)
end

# ϵ  = 1e-3
# ek = vcat(ϵ * ones(1, 100), zeros(1, 100))
# ez = vcat(zeros(1, 100), ϵ * ones(1, 100))

# h = ϵ
# # v_net(s, θᵥ) = sum(s.^2/2, dims = 1)
# s = s_test[1]
# i̅ = i_net(s, θᵢ)
# μₛ = m.μₛ(s,i̅)
# σₛ = m.σₛ(s,i̅)
# F(ϵ)        = v_net(s .+ σₛ .* (ϵ / sqrt(2.0)) .+ μₛ .* (ϵ^2 / 2.0), θᵥ)
# drift       = (F(h) - 2.0 * F(0.0) + F(-h)) / (h*h)
# drift2      = (-F(2*h) + 16*F(h)- 30.0 * F(0.0)+16*F(-h) - F(-2*h)) / (12*h*h)
# drift3      = (2*F(-3*h) - 27*F(-2*h) + 270*F(-h) - 490*F(0.0) + 270*F(h) - 27*F(2*h) + 2*F(3*h)) / (180*h*h)

# function drift_ad(s_batch, θᵥ, θᵢ)
#     i̅       = i_net(s_batch, θᵢ)
#     μₛ_batch = m.μₛ(s,i̅)
#     σₛ_batch = m.σₛ(s,i̅)
#     B = size(s_batch, 2)
#     drift = zeros(B)
#     for b in 1:B
#         s = s_batch[:, b]
#         μₛ = μₛ_batch[:, b]
#         σₛ = σₛ_batch[:, b]
#         Vs  = ForwardDiff.gradient(S->v_net(S,θᵥ)[1], s)
#         Vss = ForwardDiff.hessian(S->v_net(S,θᵥ)[1], s)
#         drift[b] = Vs' * μₛ + 0.5 * σₛ' * Vss * σₛ
#     end
#     return drift'
# end

# s = s_test[1]
# drift_exact =  drift_ad(s, θᵥ, θᵢ)

# maximum(abs.(drift - drift_exact))
# maximum(abs.(drift3 - drift_exact))
# maximum(abs.(drift2 - drift_exact))

function dividends(s, m, θᵢ)
    i̅           = i_net(s, θᵢ)
    k           = @view s[1,:]
    z           = @view s[2,:]
    revenue     = (exp.(z) .* k.^m.α)'         
    D_star      = revenue .- (i̅ + 0.5 * m.χ * i̅.^2).* k'
    D           = D_star .* (1 .+ m.λ * (D_star .< 0))
    return D
end

loss_v(s, m, θᵥ, θᵢ) = mean(abs2, hjb_residuals(s, m, θᵥ, θᵢ))
loss_i(s, m, θᵥ, θᵢ) = -mean(hjb_residuals(s, m, θᵥ, θᵢ))

#########################################################
### Training

m                    = HennessyWhited(z̅ = 0.0)
rng                  = Xoshiro(1234)
θᵥ, stᵥ              = Lux.setup(rng, v_core) |> Lux.f64
θᵢ, stᵢ              = Lux.setup(rng, i_core) |> Lux.f64
optᵥ, optᵢ           = Optimisers.Adam(1e-3), Optimisers.Adam(1e-4)
osᵥ, osᵢ             = Optimisers.setup(optᵥ, θᵥ), Optimisers.setup(optᵢ, θᵢ)

max_iter             = 100_000
kmin, kmax           = 5.0, 40.0
dk                   = Uniform(kmin, kmax)
d_z                  = Normal(m.z̅, m.σz/sqrt(2.0*m.θ))
eval_steps, improve_steps = 5, 1

p = Progress(max_iter; desc = "Training...", dt = 1.0)
it = 0
while it <= max_iter
    k_batch = rand(rng, dk, 100)'
    z_batch = rand(rng, d_z, 100)'
    s_batch = vcat(k_batch, z_batch)
    lossᵥ, lossᵢ = zero(Float64), zero(Float64)

    # Policy evaluation step
    for _ = 1:eval_steps
        lossᵥ, backᵥ = Zygote.pullback(p -> loss_v(s_batch, m, p, θᵢ), θᵥ)
        gradᵥ        = first(backᵥ(1.0))
        osᵥ, θᵥ      = Optimisers.update(osᵥ, θᵥ, gradᵥ)
    end
    
    # Policy improvement step
    for _ = 1:improve_steps
        lossᵢ, backᵢ = Zygote.pullback(p -> loss_i(s_batch, m, θᵥ, p), θᵢ)
        gradᵢ        = first(backᵢ(1.0))
        osᵢ, θᵢ      = Optimisers.update(osᵢ, θᵢ, gradᵢ)
    end

    next!(p, showvalues = [(:iter, it),("Loss_v", lossᵥ),("Loss_i", abs(lossᵢ))])
    it += 1
end

# Plotting the results

s_test   = [vcat(collect(range(kmin, kmax, length=100))', (z+m.z̅)*ones(1, 100)) for z in [log(0.87), log(1.0),log(1.15)]]

# Dividends
p1 = plot(s_test[1][1,:], [dividends(s_test[1], m, θᵢ)[:], dividends(s_test[2], m, θᵢ)[:], dividends(s_test[3], m, θᵢ)[:]], label=["Dividends" "Dividends" "Dividends"], l = 3, alpha = 0.5, legend = :topright, foreground_color_legend=:transparent, background_color_legend = :transparent)
p2 = plot(s_test[1][1,:], [i_net(s_test[1], θᵢ)[:], i_net(s_test[2],θᵢ)[:], i_net(s_test[3],θᵢ)[:]], label=["i(k, z)" "i(k, z)" "i(k, z)"], l = 3, alpha = 0.5, legend = :topright, foreground_color_legend=:transparent, background_color_legend = :transparent)
p3 = plot(s_test[1][1,:], [v_net(s_test[1], θᵥ)[:], v_net(s_test[2],θᵥ)[:], v_net(s_test[3],θᵥ)[:]], label=["V(k, z)" "V(k, z)" "V(k, z)"], l = 3, alpha = 0.5, legend = :topright, foreground_color_legend=:transparent, background_color_legend = :transparent)
plot(p1, p2, layout = (1, 2), size = (1200, 700))
