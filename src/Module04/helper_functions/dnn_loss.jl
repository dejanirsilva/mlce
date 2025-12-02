# Loss function
function loss_fn(p, ls; batch_size = 128)
    ind = rand(rng,1:sample_size,batch_size)
    y_prediction, new_ls = model(x_samples[:,ind], p, ls)
    loss = 0.5 * mean( (y_prediction-y_samples[:,ind]).^2 )
    return loss, new_ls
end