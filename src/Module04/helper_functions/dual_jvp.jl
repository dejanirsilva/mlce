# Test function
f(x; n = 1) = [exp(-mean(sqrt.(x)))+i for i = 1:n]

# Compute JVP
x, v = [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]
xdual = ForwardDiff.Dual{Float64}.(x, v) # vector of dual numbers
ydual = f(xdual; n = 2) # evaluate function at dual numbers
jvp = ForwardDiff.partials.(ydual) # jvp

# Compute Jacobian
jac = ForwardDiff.jacobian(x->f(x; n = 2), x)