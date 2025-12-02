ℓ(θ) = θ^4 - 3θ^2 + θ   # fourth-order polynomial
weights = range(1,100, length = 10)  # weights 
loss(θ) = dot(weights, ℓ.(θ))/sum(weights) # loss function