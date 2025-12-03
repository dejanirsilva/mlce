using ForwardDiff, LinearAlgebra

############################################################
### Hyper-dual approach ###
############################################################

# Define the function
V(s)         = sum(s.^2) # example function
n, m         = 100, 1 # number of state variables and shocks
s0, f, g     = ones(n), ones(n), ones(n,m) # example values

# Exact drift
∇f, H        = 2*s0, Matrix(2.0*I, n,n) # gradient and Hessian
drift_exact  = ∇f'*f + 0.5*tr(g'*H*g) # exact drift

# Hyper-dual approach
F(ϵ)         = sum([V(s0 + g[:,i]*ϵ/sqrt(2) + f/(2m)*ϵ^2) for i = 1:m])
drift_hyper  = ForwardDiff.derivative(ϵ -> ForwardDiff.derivative(F, ϵ), 0.0)

drift_exact, drift_hyper
