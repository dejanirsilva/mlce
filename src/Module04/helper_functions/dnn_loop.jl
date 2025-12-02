# train loop
loss_history = []
n_steps = 300_000
for _ in 1:n_steps
    loss, layer_states = loss_fn(parameters, layer_states)
    grad = gradient(p->loss_fn(p, layer_states)[1], parameters)[1]
    opt_state, parameters = Optimisers.update(opt_state, parameters, grad)
    push!(loss_history, loss)
    if epoch % 5000 == 0
        println("Epoch: $epoch, Loss: $loss")
    end
end