function check_HJB(m::IncomeFluctuationsModel, v::Matrix{Float64}, vW::Matrix{Float64}, c::Matrix{Float64}, s::Matrix{Float64})
    (; ρ, r, γ, Y, λ, Wgrid) = m # unpack model parameters
    return @. c.^(1-γ) / (1-γ) + vW * (r * Wgrid .+ Y' - c) + λ' * ([v[:,2] v[:,1]]- v) - ρ * v
end