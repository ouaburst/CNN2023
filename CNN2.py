# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 23:19:43 2023

@author: Budokan
"""

import numpy as np

def conv2d(image, kernel):
    # Get the dimensions of the input image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Compute the output dimensions
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Create an empty output array
    output = np.zeros((output_height, output_width))

    # Perform the convolution
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output
