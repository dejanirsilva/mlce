using Images

### Load the gazelle image
gazelle_path = joinpath(@__DIR__, "helper_functions", "n02423022_gazelle.JPEG")
gazelle_img = load(gazelle_path)
channels = gazelle_img |> channelview

### Image channels
# Red channel
channels[1,:,:]
# Green channel
channels[2,:,:]
# Blue channel
channels[3,:,:]

# Image size
nx, ny = size(channels[1,:,:])

### Plot the channels
img0 = load(gazelle_path)
img1 = reshape(RGB.(channels[1,:,:][:], [0.0],[0.0]), nx, ny)
img2 = reshape(RGB.([0.0],channels[2,:,:][:], [0.0]), nx, ny)
img3 = reshape(RGB.([0.0], [0.0],channels[3,:,:][:]), nx, ny)
