# Loss optimisation function
function loss_optimiser(loss, θ0, opt; steps=1_000, tol=1e-8)
    θ = deepcopy(θ0) # make a copy to avoid in-place modifications
    st  = Optimisers.setup(opt, θ)  # builds a “state tree”
    losses = Float64[]; gnorms = Float64[] 
    for _ in 1:steps
        ℓ, back = Zygote.pullback(loss, θ) # pullback of the loss
        g = first(back(1.0)) # compute gradient
        push!(losses, ℓ); push!(gnorms, norm(g))
        if gnorms[end] ≤ tol; break; end 
        st, θ = Optimisers.update(st, θ, g) # update state/params
    end
    return θ, (losses=losses, grad_norms=gnorms)
end