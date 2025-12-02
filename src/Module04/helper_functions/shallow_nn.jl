function shallow_nn(x::AbstractVector{<:Real}, 
    W::AbstractMatrix{<:Real}, b::AbstractVector{<:Real},
    wₙ::AbstractVector{<:Real}, bₙ::Real; σ::Function = x->max(0,x))
    @assert size(W,2) == length(x)  # ncols of W = length of x
    @assert size(W,1) == length(wₙ) # nrows of W = length of wₙ
    @assert length(b) == size(W,1)  # biases for the hidden units
    return wₙ' * σ.(W * x .+ b)+ bₙ
end
# Convenience: scalar input (d = 1); w is the column of W
function shallow_nn(x::Real, 
    w::AbstractVector{<:Real}, b::AbstractVector{<:Real},
    wₙ::AbstractVector{<:Real}, bₙ::Real; σ::Function = x->max(0,x))
    return shallow_nn([x], reshape(w,length(w),1), b, wₙ, bₙ, σ = σ)
end