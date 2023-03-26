# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:17:45 2023

@author: Budokan
"""

# Import necessary libraries
import numpy as np
import mnist
import matplotlib.pyplot as plt

# Load mnist dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Set the number of training and test samples to be used
num_train_samples = 60000  # Set the desired number of training samples
num_test_samples = 10000    # Set the desired number of testing samples

# Reduce the number of training and test images
train_images = train_images[:num_train_samples]
train_labels = train_labels[:num_train_samples]
test_images = test_images[:num_test_samples]
test_labels = test_labels[:num_test_samples]

# Normalize the pixel values of the images to be between 0 and 1
train_images = train_images / 255
test_images = test_images / 255

# Define the necessary functions

def initialize_filters_and_weights():
    # Filter shapes for convolution layers: (number of filters, channels, height, width)
    filter_shape1 = (6, 1, 5, 5)
    filter_shape2 = (16, 6, 5, 5)

    # Weight shapes for fully connected layers: (output nodes, input nodes)
    weight_shape1 = (120, 256)
    weight_shape2 = (84, 120)
    weight_shape3 = (10, 84)

    filters1 = np.random.randn(*filter_shape1) / np.sqrt(np.prod(filter_shape1))
    filters2 = np.random.randn(*filter_shape2) / np.sqrt(np.prod(filter_shape2))
    weights1 = np.random.randn(*weight_shape1) / np.sqrt(np.prod(weight_shape1))
    weights2 = np.random.randn(*weight_shape2) / np.sqrt(np.prod(weight_shape2))
    weights3 = np.random.randn(*weight_shape3) / np.sqrt(np.prod(weight_shape3))

    return filters1, filters2, weights1, weights2, weights3

def conv_layer(X, filters, stride=1, padding=0):
    (num_filters, channels, filter_height, filter_width) = filters.shape
    (batch_size, _, input_height, input_width) = X.shape
    output_height = int((input_height + 2 * padding - filter_height) / stride) + 1
    output_width = int((input_width + 2 * padding - filter_width) / stride) + 1

    output = np.zeros((batch_size, num_filters, output_height, output_width))

    for b in range(batch_size):
        for f in range(num_filters):
            for i in range(0, input_height - filter_height + 1, stride):
                for j in range(0, input_width - filter_width + 1, stride):
                    output[b, f, i, j] = np.sum(X[b, :, i:i + filter_height, j:j + filter_width] * filters[f])

    return output

def relu(X):
    return np.maximum(0, X)

def relu_derivative(X):
    return (X > 0).astype(float)

def max_pool(X, pool_size=2, stride=2):
    (batch_size, channels, input_height, input_width) = X.shape
    output_height = int((input_height - pool_size) / stride) + 1
    output_width = int((input_width - pool_size) / stride) + 1

    output = np.zeros((batch_size, channels, output_height, output_width))

    for b in range(batch_size):
        for c in range(channels):
            for i in range(0, input_height - pool_size + 1, stride):
                for j in range(0, input_width - pool_size + 1, stride):
                    output[b, c, i // stride, j // stride] = np.max(X[b, c, i:i + pool_size, j:j + pool_size])

    return output

def fully_connected(X, W):
    return np.dot(X, W.T)

def softmax(X):
    exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    num_samples = y_true.shape[0]
    clipped_y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    loss = -np.sum(y_true * np.log(clipped_y_pred)) / num_samples
    return loss

def one_hot_encoding(y, num_classes):
    encoded = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        encoded[i, label] = 1
    return encoded

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def forward_propagation(X, filters1, filters2, weights1, weights2, weights3):
    conv1 = conv_layer(X, filters1)
    relu1 = relu(conv1)
    pool1 = max_pool(relu1)
    conv2 = conv_layer(pool1, filters2)
    relu2 = relu(conv2)
    pool2 = max_pool(relu2)
    flattened = pool2.reshape(X.shape[0], -1)
    fc1 = fully_connected(flattened, weights1)
    relu3 = relu(fc1)
    fc2 = fully_connected(relu3, weights2)
    relu4 = relu(fc2)
    logits = fully_connected(relu4, weights3)
    probs = softmax(logits)
    return conv1, relu1, pool1, conv2, relu2, pool2, flattened, fc1, relu3, fc2, relu4, logits, probs

def conv_backward(d_output, X, filters, stride=1, padding=0):
    (batch_size, _, input_height, input_width) = X.shape
    (num_filters, channels, filter_height, filter_width) = filters.shape

    d_filters = np.zeros(filters.shape)
    d_X = np.zeros(X.shape)

    for b in range(batch_size):
        for f in range(num_filters):
            for i in range(0, input_height - filter_height + 1, stride):
                for j in range(0, input_width - filter_width + 1, stride):
                    d_filters[f] += d_output[b, f, i, j] * X[b, :, i:i + filter_height, j:j + filter_width]
                    d_X[b, :, i:i + filter_height, j:j + filter_width] += d_output[b, f, i, j] * filters[f]

    return d_X, d_filters

def max_pool_backward(d_output, X, pool_size=2, stride=2):
    (batch_size, channels, input_height, input_width) = X.shape

    d_X = np.zeros(X.shape)

    for b in range(batch_size):
        for c in range(channels):
            for i in range(0, input_height - pool_size + 1, stride):
                for j in range(0, input_width - pool_size + 1, stride):
                    window = X[b, c, i:i + pool_size, j:j + pool_size]
                    max_idx = np.unravel_index(np.argmax(window), window.shape)
                    d_X[b, c, i + max_idx[0], j + max_idx[1]] = d_output[b, c, i // stride, j // stride]

    return d_X

def train(X, y, filters1, filters2, weights1, weights2, weights3, epochs, batch_size, learning_rate):
    y_encoded = one_hot_encoding(y, 10)

    for epoch in range(epochs):
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y_encoded[i:i + batch_size]

            # Forward propagation
            conv1, relu1, pool1, conv2, relu2, pool2, flattened, fc1, relu3, fc2, relu4, logits, probs = forward_propagation(X_batch, filters1, filters2, weights1, weights2, weights3)

            # Compute loss
            loss = cross_entropy_loss(y_batch, probs)
            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, loss))

            # Backpropagation
            d_logits = probs - y_batch
            d_weights3 = np.dot(d_logits.T, relu4)
            d_relu4 = np.dot(d_logits, weights3)
            d_fc2 = d_relu4 * relu_derivative(fc2)
            d_weights2 = np.dot(d_fc2.T, relu3)
            d_relu3 = np.dot(d_fc2, weights2)
            d_fc1 = d_relu3 * relu_derivative(fc1)
            d_weights1 = np.dot(d_fc1.T, flattened)
            d_flattened = np.dot(d_fc1, weights1)
            d_pool2 = d_flattened.reshape(pool2.shape)
            d_relu2 = max_pool_backward(d_pool2, relu2)
            d_conv2 = d_relu2 * relu_derivative(conv2)
            d_pool1, d_filters2 = conv_backward(d_conv2, pool1, filters2)
            d_relu1 = max_pool_backward(d_pool1, relu1)
            d_conv1 = d_relu1 * relu_derivative(conv1)
            _, d_filters1 = conv_backward(d_conv1, X_batch, filters1)

            # Update weights and filters
            filters1 -= learning_rate * d_filters1
            filters2 -= learning_rate * d_filters2
            weights1 -= learning_rate * d_weights1
            weights2 -= learning_rate * d_weights2
            weights3 -= learning_rate * d_weights3

    return filters1, filters2, weights1, weights2, weights3            
            
filters1, filters2, weights1, weights2, weights3 = initialize_filters_and_weights()
train_images = train_images.reshape(-1, 1, 28, 28)
test_images = test_images.reshape(-1, 1, 28, 28)

epochs = 10
batch_size = 32
learning_rate = 0.001

filters1, filters2, weights1, weights2, weights3 = train(train_images, train_labels, filters1, filters2, weights1, weights2, weights3, epochs, batch_size, learning_rate)

_, _, _, _, _, _, _, _, _, _, _, _, test_probs = forward_propagation(test_images, filters1, filters2, weights1, weights2, weights3)
test_preds = np.argmax(test_probs, axis=1)

print("Test Accuracy: {:.4f}".format(accuracy(test_labels, test_preds)))

