@kwdef struct BlackScholesModel
    σS::Float64 = 0.20
    r::Float64 = 0.05
    K::Float64 = 1.0
    T::Float64 = 1.0
    Ns::Int64 = 150
    Nt::Int64 = 300
    sgrid::LinRange{Float64} = range(-1.5, 1.5, length=Ns)
end