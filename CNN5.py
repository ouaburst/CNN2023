# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 23:08:31 2023

@author: Budokan
"""

import numpy as np
import matplotlib.pyplot as plt
import mnist

def conv2D(image, kernel, stride=(1, 1), padding='valid', pooling=None, pool_size=(2, 2)):
    """
    Applies 2D convolution on an image using a kernel.

    Args:
        image (ndarray): The input image as a 2D or 3D NumPy array.
        kernel (ndarray): The kernel to be used for the convolution as a 2D NumPy array.
        stride (tuple): The stride of the convolution in the (height, width) direction. Defaults to (1, 1).
        padding (str): The type of padding to be applied. Can be 'valid' or 'same'. Defaults to 'valid'.
        pooling (str): The type of pooling to be applied. Can be 'max' or None. Defaults to None.
        pool_size (tuple): The size of the pooling window in the (height, width) direction. Defaults to (2, 2).

    Returns:
        ndarray: The output image as a 2D NumPy array.
    """

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
    # Define the kernels
    kernel1 = np.random.rand(5, 5)
    kernel2 = np.random.rand(5, 5)

    # Perform the first convolution
    conv1 = conv2D(image, kernel1, padding='same', pooling='max', pool_size=(2, 2))
    
    # Perform the second convolution
    conv2 = conv2D(conv1, kernel2, padding='same', pooling='max', pool_size=(2, 2))
    
    # Flatten the output
    output = conv2.flatten()

    return output


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



'''
# Load an example image from the MNIST dataset
train_images = mnist.train_images()
example_image = train_images[0]

# Apply the convolutional neural network to the example image
flattened = conv_net(example_image)

# Plot the original image and the flattened output
fig, ax = plt.subplots(1, 2)
ax[0].imshow(example_image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].plot(flattened)
ax[1].set_title('Flattened Output')
plt.show()
'''


'''
# Define the architecture
def conv_net(image):
    # First convolution layer
    conv1 = conv2D(image, np.ones((5, 5))/25, padding='same', pooling='max', pool_size=(2, 2))
    # Second convolution layer
    conv2 = conv2D(conv1, np.ones((5, 5))/25, padding='same', pooling='max', pool_size=(2, 2))
    # Flatten the output
    flat = conv2.flatten()
    return flat

# Load MNIST dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()

# Normalize the images
train_images = (train_images / 255.0) - 0.5

# Get the first image in the dataset
image = train_images[0]

# Pass the image through the convolutional neural network
output = conv_net(image)

# Print the output shape
print(output.shape)

# Plot the image
plt.imshow(image, cmap='gray')
plt.show()
'''
