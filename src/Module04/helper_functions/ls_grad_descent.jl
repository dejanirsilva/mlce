function ls_grad_descent(y::AbstractVector{<:Real}, 
    x::AbstractVector{<:Real}, θ₀::AbstractVector{<:Real}; 
    learning_rate::Real=0.01, max_iter::Integer=100, ε::Real = 1e-4)
    @assert length(x) == length(y)
    I = length(x)
    X = hcat(x, ones(I)) # design matrix
    ∇f(θ) = X' * (X * θ - y) / I # gradient
    w_path, b_path =Float64[θ₀[1]], Float64[θ₀[2]] # initial values
    for _ in 1:max_iter
        g = ∇f([w_path[end], b_path[end]])
        push!(w_path, w_path[end] - learning_rate * g[1])
        push!(b_path, b_path[end] - learning_rate * g[2])
        if norm(g) < ε
            break
        end
    end
    return (;w = w_path, b = b_path)
end