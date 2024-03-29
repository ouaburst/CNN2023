# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 17:18:58 2023

@author: Budokan
"""

import cupy as cp
import mnist
import matplotlib.pyplot as plt

# Load mnist dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

num_train_samples = 10000  # Set the desired number of training samples
num_test_samples = 5000    # Set the desired number of testing samples

# Reduce the number of training and test images
train_images = cp.array(train_images[:num_train_samples])
train_labels = cp.array(train_labels[:num_train_samples])
test_images = cp.array(test_images[:num_test_samples])
test_labels = cp.array(test_labels[:num_test_samples])

# Normalize the images
train_images = train_images / 255
test_images = test_images / 255

# Convolution operation
def convolution(image, kernel, stride):
    kernel_size = kernel.shape[0]
    output_size = (image.shape[0] - kernel_size) // stride + 1

    output = cp.zeros((output_size, output_size))

    for y in range(0, output_size):
        for x in range(0, output_size):
            img_region = image[y * stride : y * stride + kernel_size, x * stride : x * stride + kernel_size]
            output[y, x] = cp.sum(img_region * kernel)

    return output

# Max pooling operation
def max_pooling(image, pool_size, stride):
    output_size = (image.shape[0] - pool_size) // stride + 1

    output = cp.zeros((output_size, output_size))

    for y in range(0, output_size):
        for x in range(0, output_size):
            img_region = image[y * stride : y * stride + pool_size, x * stride : x * stride + pool_size]
            output[y, x] = cp.max(img_region)

    return output


# CNN implementation
def cnn(image, conv_filters, pool_size, pool_stride):
    feature_maps = [convolution(image, conv_filter, 1) for conv_filter in conv_filters]
    pooled_maps = [max_pooling(feature_map, pool_size, pool_stride) for feature_map in feature_maps]
    output = cp.concatenate([pooled_map.flatten() for pooled_map in pooled_maps])
    return output

# Activation function
def relu(x):
    return cp.maximum(x, 0)

# Softmax function
def softmax(x):
    exp_x = cp.exp(x - cp.max(x))
    return exp_x / cp.sum(exp_x, axis=1, keepdims=True)

# ANN implementation
def ann(input_data, weights, biases):
    input_layer = cp.dot(input_data, weights[0]) + biases[0]
    hidden_layer = relu(cp.dot(input_layer, weights[1]) + biases[1])
    output_layer = softmax(cp.dot(hidden_layer, weights[2]) + biases[2])
    return input_layer, hidden_layer, output_layer

# Initialize filters, weights, and biases
conv_filters = [cp.random.randn(5, 5) * 0.01 for _ in range(6)]
pool_size = 2
pool_stride = 2

input_size = 6 * 144  # 6 filters * (12x12) pooled feature maps
hidden_size = 64
output_size = 10

weights = [
    cp.random.randn(input_size, hidden_size) * 0.01,
    cp.random.randn(hidden_size, hidden_size) * 0.01,
    cp.random.randn(hidden_size, output_size) * 0.01,
]

biases = [
    cp.zeros((1, hidden_size)),
    cp.zeros((1, hidden_size)),
    cp.zeros((1, output_size)),
]

# Train the networks
learning_rate = 0.01
epochs = 5

for epoch in range(epochs):
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        cnn_output = cnn(image, conv_filters, pool_size, pool_stride)
        input_layer, hidden_layer, ann_output = ann(cnn_output.reshape(1, -1), weights, biases)

        # One-hot encode the label
        one_hot_label = cp.zeros((1, output_size))
        one_hot_label[0, label] = 1

        # Calculate loss and gradients
        loss = -cp.sum(one_hot_label * cp.log(ann_output))
        d_output = ann_output - one_hot_label
        d_hidden = cp.dot(d_output, weights[2].T) * (hidden_layer > 0)
        d_input = cp.dot(d_hidden, weights[1].T)

        # Update weights and biases
        weights[2] -= learning_rate * cp.dot(hidden_layer.T, d_output)
        biases[2] -= learning_rate * cp.sum(d_output, axis=0, keepdims=True)
        weights[1] -= learning_rate * cp.dot(input_layer.T, d_hidden)
        biases[1] -= learning_rate * cp.sum(d_hidden, axis=0, keepdims=True)
        weights[0] -= learning_rate * cp.dot(cnn_output.reshape(-1, 1), d_input)

        if i % 1000 == 0:
            print(f"Epoch: {epoch + 1}, Sample: {i}, Loss: {loss}")

# Evaluate the networks
test_cnn_outputs = cp.array([cnn(image, conv_filters, pool_size, pool_stride) for image in test_images])
test_ann_outputs = ann(test_cnn_outputs, weights, biases)

# Get only the output_layer from ann function
_, _, test_ann_outputs = ann(test_cnn_outputs, weights, biases)

accuracy = cp.mean(cp.argmax(test_ann_outputs, axis=1) == test_labels)
print(f"Test set accuracy: {accuracy * 100:.2f}%")
