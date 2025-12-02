@kwdef struct IncomeFluctuationsModel
    ρ::Float64 = 0.05
    r::Float64 = 0.03
    γ::Float64 = 2.0
    Y::Vector{Float64} = [0.1, 0.2]
    λ::Vector{Float64} = [0.02, 0.03]
    Wmin::Float64 = -0.02
    Wmax::Float64 = 2.0
    N::Int64 = 500
    Wgrid::LinRange{Float64} = range(Wmin, Wmax, length = N)
end