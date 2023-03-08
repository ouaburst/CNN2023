# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:42:27 2023

@author: Oualid Burström
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
        pool_out_h = int(np.ceil(out_h / pool_size[0]))
        pool_out_w = int(np.ceil(out_w / pool_size[1]))
        pool_output = np.zeros((pool_out_h, pool_out_w))
        for i in range(0, out_h, pool_size[0]):
            for j in range(0, out_w, pool_size[1]):
                pool_window = output[i:i+pool_size[0], j:j+pool_size[1]]
                pool_output[int(i/pool_size[0]), int(j/pool_size[1])] = np.max(pool_window[:min(pool_size[0], pool_window.shape[0]), :min(pool_size[1], pool_window.shape[1])])
    output = pool_output


    return output


# Define the convolutional neural network architecture
def conv_net(image):
    # Define the kernel and filter size
    kernel_size = (5, 5)
    
    # Create six different kernels
    #kernels = [np.random.randn(*kernel_size) for _ in range(6)]
    kernels = [np.clip(np.random.randn(*kernel_size),-0.5, 0.5) for _ in range(6)]
    

    # First convolutional layer with six filters
    conv1 = [conv2D(image, kernel, stride=(1, 1), padding='valid', pooling='max', pool_size=(2, 2)) for kernel in kernels]
    
    # Second convolutional layer with six filters
    conv2 = [conv2D(conv, kernel, stride=(1, 1), padding='valid', pooling='max', pool_size=(2, 2)) for kernel, conv in zip(kernels, conv1)]

    # Flatten the output of the second convolutional layer
    flattened = np.concatenate(conv2).flatten()

    return flattened



# Load MNIST dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()

# Normalize the images
train_images = (train_images / 255.0) - 0.5

# Get the first image in the dataset
image = train_images[0]

# Pass the image through the convolutional neural network
#output = conv_net(image)

'''
# Print the output shape
print(output.shape)

# Plot the image
plt.imshow(image, cmap='gray')
plt.show()
'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Define the simple artificial neural network
def simple_ann(input_data):
    # Define the weights and biases for the network
    w1 = np.random.randn(input_data.shape[0], 64)
    b1 = np.zeros((1, 64))
    w2 = np.random.randn(64, 32)
    b2 = np.zeros((1, 32))
    w3 = np.random.randn(32, 128)
    b3 = np.zeros((1, 128))
    w4 = np.random.randn(128, 10)
    b4 = np.zeros((1, 10))

    # Feed the input through the network
    z1 = np.dot(input_data, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = np.tanh(z2)
    z3 = np.dot(a2, w3) + b3
    a3 = np.tanh(z3)
    z4 = np.dot(a3, w4) + b4
    output = softmax(z4)

    return output


# Load MNIST dataset
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images
test_images = (test_images / 255.0) - 0.5

# Get the first image in the test dataset
test_image = test_images[0]

# Pass the test image through the CNN
cnn_output = conv_net(test_image)

# Pass the CNN output through the simple ANN
ann_output = simple_ann(cnn_output)

# Get the predicted class label
predicted_label = np.argmax(ann_output)

# Print the predicted label and the true label
print("Predicted label:", predicted_label)
print("True label:", test_labels[0])
