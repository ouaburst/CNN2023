# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 00:11:57 2023

@author: Budokan
"""

import numpy as np
import matplotlib.pyplot as plt
import mnist


def conv2D(image, kernel, filter=None, stride=(1, 1), padding='valid', pooling=None, pool_size=(2, 2)):
    # Check input dimensions
    assert len(image.shape) in [2, 3], "Input image must be 2D or 3D array"
    assert len(kernel.shape) == 2, "Kernel must be a 2D array"
    assert isinstance(stride, tuple) and len(stride) == 2, "Stride must be a tuple of two integers"
    if pooling:
        assert pooling == 'max', "Pooling must be 'max'"

    # Add padding if required
    if padding == 'same':
        pad_h = int(((image.shape[0]-1)*stride[0]+kernel.shape[0]-image.shape[0])/2)
        pad_w = int(((image.shape[1]-1)*stride[1]+kernel.shape[1]-image.shape[1])/2)
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    elif padding != 'valid':
        raise ValueError("Padding must be either 'valid' or 'same'")

    # Compute output shape
    out_h = int((image.shape[0]-kernel.shape[0])/stride[0] + 1)
    out_w = int((image.shape[1]-kernel.shape[1])/stride[1] + 1)
    output = np.zeros((out_h, out_w))

    # Perform convolution
    for i in range(0, image.shape[0]-kernel.shape[0]+1, stride[0]):
        for j in range(0, image.shape[1]-kernel.shape[1]+1, stride[1]):
            if filter is None or filter[i//stride[0], j//stride[1]]:
                output[int(i/stride[0]), int(j/stride[1])] = np.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)

    # Apply pooling if required
    if pooling:
        pool_out_h = int(out_h / pool_size[0])
        pool_out_w = int(out_w / pool_size[1])
        pool_output = np.zeros((pool_out_h, pool_out_w))
        for i in range(0, out_h, pool_size[0]):
            for j in range(0, out_w, pool_size[1]):
                pool_output[int(i/pool_size[0]), int(j/pool_size[1])] = np.max(output[i:i+pool_size[0], j:j+pool_size[1]])
        output = pool_output

    return output

def conv_net(image):
    # Define the convolution kernels
    kernel1 = np.random.randn(5, 5)
    kernel2 = np.random.randn(5, 5)

    # Perform the first convolution with 6 filters and max pooling
    conv1 = conv2D(image, kernel1, filter=None, stride=(1, 1), padding='valid', pooling='max', pool_size=(2, 2))
    conv1 = np.maximum(conv1, 0)  # Apply ReLU activation function
    pool1 = conv2D(conv1, kernel=None, filter=None, stride=(2, 2), padding='valid', pooling='max', pool_size=(2, 2))

    # Perform the second convolution with 6 filters and max pooling
    conv2 = conv2D(pool1, kernel2, filter=None, stride=(1, 1), padding='valid', pooling='max', pool_size=(2, 2))
    conv2 = np.maximum(conv2, 0)  # Apply ReLU activation function
    pool2 = conv2D(conv2, kernel=None, filter=None, stride=(2, 2), padding='valid', pooling='max', pool_size=(2, 2))

    # Flatten the output of the second pooling layer
    flattened = pool2.flatten()

    return flattened

# Load the MNIST dataset
train_images = mnist.train_images()
image = train_images[0]

# Apply the conv_net function to the image
output = conv_net(image)

# Plot the original image and the output of the convolutional network
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Input Image')
ax[1].plot(output)
ax[1].set_title('Convolutional Network Output')
plt.show()

