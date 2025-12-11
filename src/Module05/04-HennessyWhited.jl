#########################################################
# Hennessy and Whited (2007)
#########################################################
using Plots, LaTeXStrings, Distributions, LinearAlgebra, Interpolations
using Lux, Optimisers, Zygote, Random, ProgressMeter, ForwardDiff
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
### Training special case: policy evaluation ###

# HJB residual for the special case
function hjb_special_case(m, s, θᵥ)
    (; α, λ, ρ, δ) = m
    k, z  = s[1,:]', s[2,:]'
    D_star     = exp.(z) .* k.^α - δ * k
    D     = D_star .* (1 .+ λ * (D_star .< 0))
    hjb   = D - ρ * v_net(s, θᵥ)
    return hjb
end

# Exact solution for the value function
function v_special_case(m, s)
    (; α, λ, ρ, δ) = m
    k, z  = s[1,:]', s[2,:]'
    D_star     = exp.(z) .* k.^α - δ * k
    D     = D_star .* (1 .+ λ * (D_star .< 0))
    return D'/ρ
end

# Loss function for the special case
function loss_v_special_case(m, s, θᵥ)
    hjb = hjb_special_case(m, s, θᵥ)
    return mean(abs2, hjb)
end

# Value function network
v_core = Chain(
    Dense(2, 32, Lux.swish),
    Dense(32, 24, Lux.swish),
    Dense(24, 1)
)
v_net(s, θᵥ) =  (s[1,:].^m.α)' .* v_core(s, θᵥ, stᵥ)[1]

# Initialize the parameters and optimizer
rng                  = Xoshiro(1234)
θᵥ, stᵥ              = Lux.setup(rng, v_core) |> Lux.f64
optᵥ                 = Optimisers.Adam(1e-3)
osᵥ                  = Optimisers.setup(optᵥ, θᵥ)

# Model parameters
m = HennessyWhited(χ = 10.0, λ = 1.50, θ = 8.0, z̅ = -1.50, σz = 0.05)

# Training parameters
max_iter             = 300_000
kmin, kmax           = 0.0, 10.0
dk                   = Uniform(kmin, kmax)
dz                   = Normal(m.z̅, 0.10)

p = Progress(max_iter; desc = "Training...", dt = 1.0)
loss_history_special_case = Float64[]
it = 0
while it <= max_iter
    s_batch = vcat(rand(rng, dk, 150)', rand(rng, dz, 150)')
    lossᵥ, lossᵢ = zero(Float64), zero(Float64)

    # Policy evaluation step
    lossᵥ, backᵥ = Zygote.pullback(p -> loss_v_special_case(m, s_batch, p), θᵥ)
    gradᵥ        = first(backᵥ(1.0))
    osᵥ, θᵥ      = Optimisers.update(osᵥ, θᵥ, gradᵥ)
    push!(loss_history_special_case, lossᵥ)

    next!(p, showvalues = [(:iter, it),("Loss_v", lossᵥ)])
    if lossᵥ < 1e-6
        println("Iteration ", it, "| Loss_v = ", lossᵥ)
        break
    end
    it += 1
end

# Plotting the loss history for the special case
p_loss_special_case = plot((1:length(loss_history_special_case))/1e3, loss_history_special_case, l = 3.0, yaxis = :log10, alpha = 0.75,
    label = "", ylabel = "Loss", xlabel = "Iteration (in thousands)",
    foreground_color_legend=:transparent, background_color_legend = :transparent)    

# Plotting the value function for the special case
s_test1 = vcat(collect(range(kmin, kmax, length=250))', m.z̅*ones(1, 250))
p_special_case = plot(s_test1[1,:], v_net(s_test1, θᵥ)', l = 3.0, color = palette(:auto)[1], alpha = 0.75, 
    xlabel = L"k", ylabel = L"V(k, z)", ylims = (-20.0, 5.0),
    legend = :bottomleft, label = L"\mathrm{DNN:} z = \overline{z}", foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(s_test1[1,:], v_special_case(m, s_test1), l = 1.5, ls = :dash, color = palette(:auto)[1], alpha = 1.0, label = L"\mathrm{Exact:} z = \overline{z}")
s_test2 = vcat(collect(range(kmin, kmax, length=150))', (m.z̅-0.10)*ones(1, 150))
plot!(s_test2[1,:], v_net(s_test2, θᵥ)', l = 3.0, color = palette(:auto)[2], alpha = 0.75, 
 label = L"\mathrm{DNN:} z = \overline{z} - 0.10")
plot!(s_test2[1,:], v_special_case(m, s_test2), l = 1.5, ls = :dash, color = palette(:auto)[2], alpha = 1.0,
    label = L"\mathrm{Exact:} z = \overline{z} - 0.10")
s_test3 = vcat(collect(range(kmin, kmax, length=150))', (m.z̅+0.10)*ones(1, 150))
plot!(s_test3[1,:], v_net(s_test3, θᵥ)', l = 3.0, color = palette(:auto)[3], alpha = 0.75,
    label = L"\mathrm{DNN:} z = \overline{z} + 0.10")
plot!(s_test3[1,:], v_special_case(m, s_test3), l = 1.5, ls = :dash, color = palette(:auto)[3], alpha = 1.0,
    label = L"\mathrm{Exact:} z = \overline{z} + 0.10")

#########################################################
### HJB residual ###

function second_derivative_FD(F::Function, h::Float64; stencil::Symbol = :three)
    if stencil == :nine
        return (-9.0 .* F(-4h) .+ 128.0 .* F(-3h) .- 1008.0 .* F(-2h) .+ 8064.0 .* F(-h) .- 14350.0 .* F(0.0) .+ 8064.0 .* F(h) .- 1008.0 .* F(2h) .+ 128.0 .* F(3h) .- 9.0 .* F(4h)) ./ (5040.0 * h^2)
    elseif stencil == :seven
        return (2.0 .* F(-3h) .- 27.0 .* F(-2h) .+ 270.0 .* F(-h) .- 490.0 .* F(0.0) .+ 270.0 .* F(h) .- 27.0 .* F(2h) .+ 2.0 .* F(3h)) ./ (180.0 * h^2)
    elseif stencil == :five
        return (-F(2h) .+ 16.0 .* F(h) .- 30.0 .* F(0.0) .+ 16.0 .* F(-h) .- F(-2h)) ./ (12.0 * h^2)
    else 
        return (F(h) - 2.0 * F(0.0) + F(-h)) / (h*h) # Three point stencil
    end
end

function hjb_residual(m, s, θᵥ, θᵢ; h = 5e-2, stencil::Symbol = :three)
    (; α, λ, ρ, δ, χ, θ, z̅, σz) = m
    k, z  = s[1,:]', s[2,:]'
    i     = i_net(s, θᵢ)
    D_star     = exp.(z) .* k.^α - (i + 0.5*χ*i.^2).*k
    D     = D_star .* (1 .+ λ * (D_star .< 0))
    μk    = (i .- δ) .* k
    μz    = -θ .* (z .- z̅)
    μₛ    = vcat(μk, μz)
    σₛ    = vcat(zeros(1, size(s,2)), σz*ones(1, size(s,2)))
    F(ϵ)  = v_net(s .+ σₛ .* (ϵ / sqrt(2.0)) .+ μₛ * (ϵ^2 / 2.0), θᵥ)
    drift = second_derivative_FD(F, h, stencil = stencil)
    hjb   = D + drift - m.ρ * v_net(s, θᵥ)
    return hjb
end
 
function loss_v(m, s, θᵥ, θᵢ; h = 5e-2, stencil::Symbol = :nine)
    hjb = hjb_residual(m, s, θᵥ, θᵢ; h = h, stencil = stencil)
    return mean(abs2, hjb)
end

function loss_i(m, s, θᵥ, θᵢ; h = 5e-2, stencil::Symbol = :nine)
    hjb = hjb_residual(m, s, θᵥ, θᵢ; h = h, stencil = stencil)
    return -mean(hjb)
end

derivative_v_net(s, θᵥ) = map(x-> ForwardDiff.derivative(X-> v_net([X;s[2,2]], θᵥ), x)[1], s[1,:])'

#########################################################
### Training ###

# Value function network
v_core = Chain(
    Dense(2, 32, Lux.swish),
    Dense(32, 24, Lux.swish),
    Dense(24, 1)
)
v_net(s, θᵥ) =  (s[1,:].^m.α)' .* v_core(s, θᵥ, stᵥ)[1]

# Initialize the parameters and optimizer
rng                  = Xoshiro(1234)
θᵥ, stᵥ              = Lux.setup(rng, v_core) |> Lux.f64
optᵥ                 = Optimisers.Adam(1e-3)
osᵥ                  = Optimisers.setup(optᵥ, θᵥ)

### Training
i_core = Chain(
    Dense(2, 32, Lux.gelu),
    Dense(32, 24, Lux.gelu),
    Dense(24, 1)
)

i_net(s, θᵢ) = i_core(s, θᵢ, stᵢ)[1] 

# Initialize the parameters and optimizer
θᵢ, stᵢ              = Lux.setup(rng, i_core) |> Lux.f64
optᵢ                 = Optimisers.Adam(1e-3)
osᵢ                  = Optimisers.setup(optᵢ, θᵢ)

m = HennessyWhited(χ = 10.0, λ = 0.0, θ = 8.0, z̅ = -0.20, σz = 0.05)

max_iter             = 150_000
kmin, kmax           = 1.0, 8.0
dk                   = Uniform(kmin, kmax)
dz                   = Normal(m.z̅, m.σz/sqrt(2.0*m.θ))

p = Progress(max_iter; desc = "Training...", dt = 1.0)
loss_history_v = Float64[]
loss_history_i = Float64[]
it = 0
nsteps_v, nsteps_i = 10, 1
while it <= max_iter
    s_batch = vcat(rand(rng, dk, 150)', rand(rng, dz, 150)')
    lossᵥ, lossᵢ = zero(Float64), zero(Float64)

    # Policy evaluation step
    for _ = 1:nsteps_v
        lossᵥ, backᵥ = Zygote.pullback(p -> loss_v(m, s_batch, p, θᵢ, stencil = :nine), θᵥ)
        gradᵥ        = first(backᵥ(1.0))
        osᵥ, θᵥ      = Optimisers.update(osᵥ, θᵥ, gradᵥ)
        # Compute loss with updated θᵥ to track actual training progress
        lossᵥ = loss_v(m, s_batch, θᵥ, θᵢ)
    end

    # Policy improvement step
    for _ = 1:nsteps_i
        lossᵢ, backᵢ = Zygote.pullback(p -> loss_i(m, s_batch, θᵥ, p), θᵢ)
        gradᵢ        = first(backᵢ(1.0))
        osᵢ, θᵢ      = Optimisers.update(osᵢ, θᵢ, gradᵢ)
    end

    # Compute loss with current values of both parameters
    lossᵥ = loss_v(m, s_batch, θᵥ, θᵢ)
    lossᵢ = loss_i(m, s_batch, θᵥ, θᵢ)
    push!(loss_history_v, lossᵥ)
    push!(loss_history_i, lossᵢ)

    next!(p, showvalues = [(:iter, it),("Loss_v", lossᵥ),("Loss_i", lossᵢ)])
    if max(lossᵥ, abs(lossᵢ)) < 1e-6
        println("Iteration ", it, "| Loss_v = ", lossᵥ, "| Loss_i = ", lossᵢ)
        break
    end
    it += 1
end

# Plot loss history
p_loss_v = plot((1:length(loss_history_v))/1e3, loss_history_v, l = 3.0, yaxis = :log10, legend = :topright, 
    alpha = 0.75, label = L"\mathrm{Loss}: v", ylabel = "Loss", xlabel = "Iteration (in thousands)",
    foreground_color_legend=:transparent, background_color_legend = :transparent)
 plot!((1:length(loss_history_v))/1e3, abs.(loss_history_i), l = 2.0, yaxis = :log10, alpha = 0.75,
    label = L"\mathrm{Loss}: i", foreground_color_legend=:transparent, background_color_legend = :transparent)

s_test = vcat(collect(range(kmin, kmax, length=150))', m.z̅*ones(1, 150))

p_i = plot(s_test[1,:], i_net(s_test, θᵢ)', l = 3.0, ylims = (0.0, 0.4), xlabel = L"k", ylabel = L"i(k,z)",
    legend = :topright, label = L"\mathrm{DNN:} i(k,z)", foreground_color_legend=:transparent, background_color_legend = :transparent)
plot!(s_test[1,:], (derivative_v_net(s_test, θᵥ)' .- 1.0)/m.χ, l = 2.0, ls = :dash,
    label = L"\mathrm{Exact:} (v_k(k,z)-1)/\chi")
