# Compute derivative
function derivative(f::Function, x::Real)
    u = D(x, 1.0) # construct the dual number u = x_0 + 1.0 * ϵ
    return f(u).d # extract the dual part of the result
end