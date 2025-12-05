#########################################################
# Hennessy and Whited (2007)
#########################################################
using Plots, LaTeXStrings, Distributions, LinearAlgebra
using Lux, Optimisers, Zygote, Random, ProgressMeter
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
v_net(s, θᵥ) =  s[1,:]' .* v_core(s, θᵥ, stᵥ)[1]
i_core = Chain(
    Dense(2, 64, Lux.relu),
    Dense(64, 32, Lux.relu),
    Dense(32, 32, Lux.relu),
    Dense(32, 1)
)
i_net(s, θᵢ) = i_core(s, θᵢ, stᵢ)[1]

#########################################################
### Special case ###
#########################################################
function hjb_residuals_special_case(s, m, θᵥ; h = 1e-3)
    k           = @view s[1,:]
    z           = @view s[2,:]
    i̅           = ones(1, length(k)) * m.δ
    revenue     = (exp.(z) .* k.^m.α)'         
    D_star      = revenue .- m.δ.* k'
    D           = D_star .* (1 .+ m.λ * (D_star .< 0))
    μₛ, σₛ       = m.μₛ(s,i̅), m.σₛ(s,i̅)
    F(ϵ)        = v_net(s .+ σₛ .* (ϵ / sqrt(2.0)) .+ μₛ .* (ϵ^2 / 2.0), θᵥ)
	drift       = (F(h) - 2.0 * F(0.0) + F(-h)) / (h*h)
    return D + drift - m.ρ * v_net(s, θᵥ)
end

loss_v_special_case(s, m, θᵥ) = mean(abs2, hjb_residuals_special_case(s, m, θᵥ))

# Exact solution for the value function when χ -> ∞ and θ=σz=0
function v_exact(s::AbstractMatrix, m::HennessyWhited)
    (; α, λ, ρ, δ) = m
    k = @view s[1,:]
    z = @view s[2,:]
    D_star = exp.(z) .* k.^α  - δ * k
    D = D_star .* (1 .+ λ * (D_star .< 0))
    return D'/ρ
end

#########################################################
### Training

m                    = HennessyWhited(σz = 0.0, θ = 0.0, z̅ = -1.0, λ = 0.0)                            
rng                  = Xoshiro(1234)
θᵥ, stᵥ              = Lux.setup(rng, v_core) |> Lux.f64
optᵥ                 = Optimisers.Adam(1e-3)
osᵥ                  = Optimisers.setup(optᵥ, θᵥ)

max_iter             = 150_000
kmin, kmax           = 0.0, 30.0
dk                   = Uniform(kmin, kmax)
d_z                  = Normal(m.z̅, 0.17)

p = Progress(max_iter; desc = "Training...", dt = 1.0)
loss_history = Float64[]
it = 0
while it <= max_iter
    s_batch = vcat(rand(rng, dk, 150)', rand(rng, d_z, 150)')
    lossᵥ = zero(Float64)
    
    # Policy evaluation step
    lossᵥ, backᵥ = Zygote.pullback(p -> loss_v_special_case(s_batch, m, p), θᵥ)
    gradᵥ        = first(backᵥ(1.0))
    osᵥ, θᵥ      = Optimisers.update(osᵥ, θᵥ, gradᵥ)
    push!(loss_history, lossᵥ)
    next!(p, showvalues = [(:iter, it),("Loss_v", lossᵥ)])
    if lossᵥ < 5e-6
        break
    end
    it += 1
end

Δz_range = [log(0.87), log(1.0),log(1.15)]
s_test   = [vcat(collect(range(0.0, 30, length=100))', (z+m.z̅)*ones(1, 100)) for z in [log(0.87), log(1.0),log(1.15)]]

# Loss history
p1 = plot(loss_history, yscale=:log10, label="", l = 3, 
            xlabel="Iteration", ylabel="Loss", title="")

# Value function
p2 = plot(s_test[1][1,:], [v_net(s_test[1], θᵥ)[:], v_net(s_test[2],θᵥ)[:], v_net(s_test[3],θᵥ)[:]], 
        label=reshape([LaTeXString("DNN: \$z - \\bar{z} = $(round(Δz, digits=2))\$") for Δz in Δz_range], 1, 3), 
        l = 3, alpha = 0.5, xlabel = L"k", ylabel = L"v(k, z)", legend = :bottomleft, foreground_color_legend=:transparent, background_color_legend = :transparent)
     plot!(s_test[1][1,:], [v_exact(s_test[1], m)[:], v_exact(s_test[2], m)[:], v_exact(s_test[3], m)[:]], 
        label=reshape([LaTeXString("Exact: \$z - \\bar{z} = $(round(Δz, digits=2))\$") for Δz in Δz_range], 1, 3),
        l = 3, alpha = 0.5, foreground_color_legend=:transparent, background_color_legend = :transparent, color = palette(:auto)[1:3]', ls =:dash)