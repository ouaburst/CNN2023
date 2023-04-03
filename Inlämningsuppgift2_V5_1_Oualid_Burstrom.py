# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:58:54 2023

@author: brobm
"""

import numpy as np
import mnist

# Load mnist dataset
x_train = mnist.train_images()
y_train = mnist.train_labels()
x_test = mnist.test_images()
y_test = mnist.test_labels()

# Reshape the input data
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical format
y_train = np.eye(10)[y_train.astype('int32')]
y_test = np.eye(10)[y_test.astype('int32')]

# -------------------- Functions -------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def convolution2d(image, kernel, mode='valid'):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    if mode == 'full':
        output_height = image_height + kernel_height - 1
        output_width = image_width + kernel_width - 1
        padded_image = np.pad(image, ((kernel_height - 1, kernel_height - 1), (kernel_width - 1, kernel_width - 1)))
    else:
        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1
        padded_image = image

    output = np.zeros((output_height, output_width))

    for y in range(output_height):
        for x in range(output_width):
            output[y, x] = np.sum(padded_image[y:y+kernel_height, x:x+kernel_width] * kernel)

    return output

# Forward Propagation
def conv_forward(input, weights, bias, padding=0):
    input_shape = input.shape
    input_depth = input_shape[2]
    kernel_shape = weights.shape[:2]
    layer_depth = weights.shape[3]
    output_shape = (input_shape[0] - kernel_shape[0] + 2 * padding + 1,
                    input_shape[1] - kernel_shape[1] + 2 * padding + 1,
                    layer_depth)

    if padding > 0:
        padded_input = np.pad(input, ((padding, padding),
                                       (padding, padding),
                                       (0, 0)), mode='constant', constant_values=0)
    else:
        padded_input = input

    output = np.zeros(output_shape)

    for k in range(layer_depth):
        for d in range(input_depth):
            kernel = weights[:, :, d, k]
            output[:, :, k] += convolution2d(padded_input[:, :, d], kernel) + bias[k]

    return output, padded_input

def maxpool_forward(input, pool_size=2, stride=2):
    input_shape = input.shape
    output_shape = (input_shape[0] // stride, input_shape[1] // stride, input_shape[2])

    output = np.zeros(output_shape)

    for y in range(0, input_shape[0], stride):
        for x in range(0, input_shape[1], stride):
            for z in range(input_shape[2]):
                output[y // stride, x // stride, z] = np.max(input[y:y + pool_size, x:x + pool_size, z])

    return output

def flatten(input):
    return input.flatten()

def dense_forward(input, weights, bias):
    return np.dot(input, weights) + bias

# Backward Propagation
def conv_backward(output_error, padded_input, weights, padding=0):
    input_shape = padded_input.shape
    input_depth = input_shape[2]
    kernel_shape = weights.shape[:2]
    layer_depth = weights.shape[3]
    output_error_shape = output_error.shape

    kernel_gradient = np.zeros(weights.shape)
    bias_gradient = np.zeros(weights.shape[3])
    input_gradient = np.zeros(padded_input.shape)

    for k in range(layer_depth):
        for d in range(input_depth):
            kernel_gradient[:, :, d, k] = convolution2d(padded_input[:, :, d], output_error[:, :, k], mode='valid')

    for k in range(layer_depth):
        bias_gradient[k] = np.sum(output_error[:, :, k])

    for y in range(output_error_shape[0]):
        for x in range(output_error_shape[1]):
            for k in range(layer_depth):
                kernel = weights[:, :, :, k]
                input_gradient[y:y + kernel_shape[0], x:x + kernel_shape[1], :] += kernel * output_error[y, x, k]

    if padding > 0:
        input_gradient = input_gradient[padding:-padding, padding:-padding, :]

    return input_gradient, kernel_gradient, bias_gradient

def dense_backward(output_error, input, weights):
    input_gradient = np.dot(output_error, weights.T)
    weights_gradient = np.outer(input, output_error)
    bias_gradient = output_error

    return input_gradient, weights_gradient, bias_gradient

# ------------------- Training -------------------

# Initialize weights and biases
conv_weights = np.random.randn(3, 3, 1, 8) * 0.1
conv_bias = np.zeros(8)

dense_weights = np.random.randn(13 * 13 * 8, 10) * 0.1
dense_bias = np.zeros(10)

# Training parameters
learning_rate = 0.01
epochs = 10
batch_size = 32

# Training loop
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Forward pass
        conv_output, padded_input = conv_forward(x_batch, conv_weights, conv_bias)
        conv_output_relu = np.maximum(conv_output, 0)
        maxpool_output = maxpool_forward(conv_output_relu)
        flat_output = np.array([flatten(maxpool_output[i]) for i in range(maxpool_output.shape[0])])
        dense_output = dense_forward(flat_output, dense_weights, dense_bias)
        y_pred = sigmoid(dense_output)

        # Calculate gradients
        mse_error = mse_derivative(y_batch, y_pred)
        sigmoid_error = mse_error * sigmoid_derivative(dense_output)
        flat_error, dense_weights_grad, dense_bias_grad = dense_backward(sigmoid_error, flat_output, dense_weights)
        maxpool_error = np.array([flat_error[i].reshape(13, 13, 8) for i in range(flat_error.shape[0])])
        conv_output_relu_error = maxpool_error.repeat(2, axis=1).repeat(2, axis=2)
        conv_output_error = conv_output_relu_error * (conv_output > 0)
        input_error, conv_weights_grad, conv_bias_grad = conv_backward(conv_output_error, padded_input, conv_weights)

        # Update weights and biases
        conv_weights -= learning_rate * conv_weights_grad
        conv_bias -= learning_rate * conv_bias_grad
        dense_weights -= learning_rate * dense_weights_grad
        dense_bias -= learning_rate * dense_bias_grad

    # Evaluate training accuracy
    conv_output, _ = conv_forward(x_train, conv_weights, conv_bias)
    conv_output_relu = np.maximum(conv_output, 0)
    maxpool_output = maxpool_forward(conv_output_relu)
    flat_output = np.array([flatten(maxpool_output[i]) for i in range(maxpool_output.shape[0])])
    dense_output = dense_forward(flat_output, dense_weights, dense_bias)
    y_pred = sigmoid(dense_output)
    training_accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_train, axis=1))

    print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {training_accuracy:.4f}")

# Evaluate test accuracy
conv_output, _ = conv_forward(x_test, conv_weights, conv_bias)
conv_output_relu = np.maximum(conv_output, 0)
maxpool_output = maxpool_forward(conv_output_relu)
flat_output = np.array([flatten(maxpool_output[i]) for i in range(maxpool_output.shape[0])])
dense_output = dense_forward(flat_output, dense_weights, dense_bias)
y_pred = sigmoid(dense_output)
test_accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))

print(f"Test Accuracy: {test_accuracy:.4f}")

