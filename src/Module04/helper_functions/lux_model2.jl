# Defining the parameters
parameters = (layer_1 = (weight = [1.0;1.0;;], bias = [0.0,-0.5]),
             layer_2 = (weight = [2.0 -4.0], bias = [0.0]))

# Checking the model prediction
y_pred, state = model([0.0,0.5,1.0]', parameters, state) # returns [0.0, 1.0, 0.0]
