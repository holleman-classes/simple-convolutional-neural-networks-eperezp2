import numpy as np

# Define the network architecture specifications for Convolutional and Dense layers
# Convolutional layer tuple format: (number of filters, kernel size, stride, input channels)
# Dense layer tuple format: (number of output units, number of input units [to be calculated later])
conv_layers = [
    (32, 3, 2, 3),    # First Conv2D layer with 32 filters, 3x3 kernel, stride of 2, and 3 input channels (RGB image)
    (64, 3, 2, 32),   # Second Conv2D layer with 64 filters, input channels from the previous layer's filters
    (128, 3, 2, 64),  # Third Conv2D layer
    # Repeating Conv2D layers with 128 filters, a 3x3 kernel, and stride of 1. These have the same input/output channels.
    (128, 3, 1, 128), 
    (128, 3, 1, 128),
    (128, 3, 1, 128),
    (128, 3, 1, 128)
]

dense_layers = [
    (128, 0),  # First Dense layer, input units will be calculated after flattening the output of the last Conv layer
    (10, 128)  # Second Dense layer, for CIFAR-10 classification (10 output classes)
]

# Initialize lists to store the calculations for each layer
output_sizes = [32]  # Starting with the initial input image size of 32x32 pixels
params = []
macs = []
layer_names = []

# Calculate parameters and MACs for Conv2D layers
for i, (filters, kernel_size, stride, input_channels) in enumerate(conv_layers):
    layer_name = f"Conv2D-{filters}f-{kernel_size}x{kernel_size}-s{stride}"
    layer_names.append(layer_name)
    # Calculate the number of parameters in the Conv layer (including bias for each filter)
    params_conv = (kernel_size * kernel_size * input_channels + 1) * filters
    params.append(params_conv)
    
    # Calculate the output size of the layer considering the stride and input size
    output_size = np.ceil(output_sizes[-1] / stride)
    output_sizes.append(output_size)
    # Calculate MACs considering the kernel size, input channels, number of filters, and the output size
    macs_conv = kernel_size * kernel_size * input_channels * filters * output_size * output_size
    macs.append(macs_conv)

# Add parameters for BatchNorm layers after each Conv2D and Dense layer
# Note: BatchNorm layers adjust the activation, not contributing traditional MACs
for filters in [layer[0] for layer in conv_layers]:
    layer_names.append(f"BatchNorm-{filters}")
    params_bn = 2 * filters  # For gamma and beta parameters in BatchNorm
    params.append(params_bn)
    macs.append(0)  # No traditional MACs for BatchNorm

# Handle MaxPooling layer separately
layer_names.append("MaxPooling")
params.append(0)  # No learnable parameters in MaxPooling
output_size = np.ceil(output_sizes[-1] / 4)  # Considering a 4x4 pooling size with stride 4
output_sizes.append(output_size)
macs.append(0)  # No MACs for MaxPooling

# Flatten the output of the last Conv/Pooling layer to feed into Dense layers
layer_names.append("Flatten")
params.append(0)  # Flattening does not involve learnable parameters
macs.append(0)  # No MACs for flattening
flattened_size = output_size * output_size * conv_layers[-1][0]  # Calculate flattened size for the first Dense layer
output_sizes.append(flattened_size)

# Calculate parameters and MACs for Dense layers
for i, (output_units, input_units) in enumerate(dense_layers):
    # Update the input units for the first dense layer based on the flattened size
    if input_units == 0:  
        input_units = int(flattened_size)
        dense_layers[i] = (output_units, input_units)
    layer_name = f"Dense-{output_units}u"
    layer_names.append(layer_name)
    # Calculate parameters (weights + bias) for Dense layers
    params_dense = (input_units + 1) * output_units
    params.append(params_dense)
    # Calculate MACs for Dense layers (simply the product of input and output units)
    macs_dense = input_units * output_units
    macs

