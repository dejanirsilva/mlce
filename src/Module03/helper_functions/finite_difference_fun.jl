"""
    finite_difference(f, x0, accuracy; scheme=:forward, Δx=0.01)

Approximate f'(x0) by differentiating a local interpolating 
polynomial built on a stencil that achieves the requested accuracy.
"""
function finite_difference(f::Function, x0::Float64, accuracy::Int;
                    scheme::Symbol = :forward, Δx::Float64 = 0.01)
    @assert accuracy > 0 "accuracy order must be positive"
    steps = zeros(Int, accuracy+1)
    if iseven(accuracy)
        m = accuracy ÷ 2
        steps = collect(-m:m)
    else
        q = accuracy
        steps = scheme === :forward ? collect(0:q) : collect(-q:0)
    end
    x_points = x0 .+ Δx .* steps
    p    = Polynomials.fit(x_points, f.(x_points), length(x_points) - 1)
    return Polynomials.derivative(p)(x0)
end