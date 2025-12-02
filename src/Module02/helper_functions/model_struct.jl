## Model struct
Base.@kwdef struct ConsumptionSavingsDT
    γ::Float64 = 2.0        # CRRA coefficient
    ρ::Float64 = 0.05       # discount rate
    A::Float64 = 1.00       # terminal value function parameter
    R::Float64 = exp(ρ)     # interest rate
    σ::Float64 = 0.25       # standard deviation of log income
    Z::NamedTuple = tauchen(9, 0.0, σ) # income process
    Y::Vector{Float64} = exp.(Z.z)  # income levels
    N::Int64  = 11          # number of grid points
    α::Float64 = 0.0        # grid spacing parameter
    Mgrid::Vector{Float64} = make_grid(0.0, 2.5, N; α = α)
    agrid::Vector{Float64} = make_grid(0.0, 1.0, N; α = α)
end