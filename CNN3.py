# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:53:05 2023

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


# Load MNIST dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()

# Define kernel
kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

print("++++++++++ ",len(kernel.shape))

# Define sample image
image = train_images[0]

# Apply convolution
output = conv2D(image, kernel, stride=(1, 1), padding='same', pooling='max', pool_size=(2, 2))

# Print output
#print(output)

# Plot original image and output
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(output, cmap='gray')
ax2.set_title('Convolved Image')
plt.show()

'''
# Define input image and kernel
image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
kernel = np.array([[1, 0], [0, 1], [1, 1]])

# Perform convolution with padding and max pooling
output = conv2D(image, kernel, stride=(1, 1), padding='same')
#pooled_output = max_pool2D(output, pool_size=(2, 2))

# Print results
print("Input image:")
print(image)
print("Convolved output:")
print(output)
#print("Max-pooled output:")
#print(pooled_output)
'''