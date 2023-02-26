# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 21:48:49 2023

@author: Budokan
"""
import numpy as np

def conv2D(image, kernel, stride=(1, 1), padding='valid'):
    """
    Applies 2D convolution on an image using a kernel.

    Args:
        image (ndarray): The input image as a 2D or 3D NumPy array.
        kernel (ndarray): The kernel to be used for the convolution as a 2D NumPy array.
        stride (tuple): The stride of the convolution in the (height, width) direction. Defaults to (1, 1).
        padding (str): The type of padding to be applied. Can be 'valid' or 'same'. Defaults to 'valid'.

    Returns:
        ndarray: The output image as a 2D NumPy array.
    """

    # Check input dimensions
    assert len(image.shape) in [2, 3], "Input image must be 2D or 3D array"
    assert len(kernel.shape) == 2, "Kernel must be a 2D array"
    assert isinstance(stride, tuple) and len(stride) == 2, "Stride must be a tuple of two integers"

    # Add padding if required
    if padding == 'same':
        pad_h = int(((image.shape[0]-1)*stride[0]+kernel.shape[0]-image.shape[0])/2)
        pad_w = int(((image.shape[1]-1)*stride[1]+kernel.shape[1]-image.shape[1])/2)
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
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

    return output

# Test the conv2D function
image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
kernel = np.array([[1, 0], [0, 1], [1, 1]])

output = conv2D(image, kernel, stride=(1, 1), padding='same')
print(output)
