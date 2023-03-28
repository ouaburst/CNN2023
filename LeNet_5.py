# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:04:06 2023

@author: Budokan
"""

import numpy as np

# CNN layers
conv1 = np.random.randn(6, 5, 5)  # 6 filters, 5x5 kernel
conv2 = np.random.randn(6, 5, 5)  # 6 filters, 5x5 kernel

# ANN layers
#ann1_weights = np.random.randn(150, 60)
ann1_weights = np.random.randn(96, 60)
ann2_weights = np.random.randn(60, 40)
ann3_weights = np.random.randn(40, 10)

def forward_pass_cnn(input_image, conv1, conv2):
    # Conv1
    conv1_output = np.zeros((6, 24, 24))
    for f in range(6):
        for i in range(24):
            for j in range(24):
                conv1_output[f, i, j] = np.sum(input_image[:, i:i+5, j:j+5] * conv1[f])

    # Max pooling 1
    max_pool1_output = np.zeros((6, 12, 12))
    for f in range(6):
        for i in range(0, 24, 2):
            for j in range(0, 24, 2):
                max_pool1_output[f, i//2, j//2] = np.max(conv1_output[f, i:i+2, j:j+2])

    # Conv2
    conv2_output = np.zeros((6, 8, 8))
    for f in range(6):
        for i in range(8):
            for j in range(8):
                conv2_output[f, i, j] = np.sum(max_pool1_output[:, i:i+5, j:j+5] * conv2[f])

    # Max pooling 2
    max_pool2_output = np.zeros((6, 4, 4))
    for f in range(6):
        for i in range(0, 8, 2):
            for j in range(0, 8, 2):
                max_pool2_output[f, i//2, j//2] = np.max(conv2_output[f, i:i+2, j:j+2])

    # Flatten
    flattened_output = max_pool2_output.flatten()
    return flattened_output, conv1_output, max_pool1_output, conv2_output


def forward_pass_ann(input_data, ann1_weights, ann2_weights, ann3_weights):
    hidden1 = np.dot(input_data, ann1_weights)
    hidden2 = np.dot(hidden1, ann2_weights)
    output = np.dot(hidden2, ann3_weights)
    return output, hidden1, hidden2

def backward_pass_ann(output_error, hidden2, hidden1, input_data, ann1_weights, ann2_weights, ann3_weights, learning_rate):
    # Update the weights using the chain rule and gradient descent
    ann3_delta = output_error
    ann2_delta = np.dot(ann3_delta, ann3_weights.T)
    ann1_delta = np.dot(ann2_delta, ann2_weights.T)

    ann3_weights -= learning_rate * np.outer(hidden2, ann3_delta)
    ann2_weights -= learning_rate * np.outer(hidden1, ann2_delta)
    ann1_weights -= learning_rate * np.outer(input_data, ann1_delta)
    ann1_input_delta = np.dot(ann1_delta, ann1_weights.T)
    
    return ann1_input_delta, hidden1, hidden2



def backward_pass_cnn(input_image, ann1_input_delta, conv1, conv2, max_pool1_output, conv1_output, conv2_output, learning_rate):
    # Reshape ann1_input_delta back to max_pool2_output shape
    max_pool2_delta = ann1_input_delta.reshape((6, 4, 4))

    # Max pooling 2 gradient
    conv2_delta = np.zeros((6, 8, 8))
    for f in range(6):
        for i in range(0, 8, 2):
            for j in range(0, 8, 2):
                idx = np.unravel_index(np.argmax(conv2_output[f, i:i+2, j:j+2]), (2, 2))
                conv2_delta[f, i+idx[0], j+idx[1]] = max_pool2_delta[f, i//2, j//2]

    # Conv2 gradient
    conv2_grad = np.zeros_like(conv2)
    for f in range(6):
        for i in range(5):
            for j in range(5):
                conv2_grad[f, i, j] = np.sum(conv2_delta[f] * max_pool1_output[:, i:i+8, j:j+8])
    conv2 -= learning_rate * conv2_grad

    # Max pooling 1 gradient
    conv1_delta = np.zeros((6, 24, 24))
    for f in range(6):
        for i in range(0, 24, 2):
            for j in range(0, 24, 2):
                idx = np.unravel_index(np.argmax(conv1_output[f, i:i+2, j:j+2]), (2, 2))
                conv1_delta[f, i+idx[0], j+idx[1]] = max_pool1_output[f, i//2, j//2]

    # Conv1 gradient
    conv1_grad = np.zeros_like(conv1)
    for f in range(6):
        for i in range(5):
            for j in range(5):
                conv1_grad[f, i, j] = np.sum(conv1_delta[f] * input_image[:, i:i+24, j:j+24])
    conv1 -= learning_rate * conv1_grad


def train_network(input_images, labels, num_epochs, learning_rate):
    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        for input_image, label in zip(input_images, labels):
            # Forward pass
            flattened_output, conv1_output, max_pool1_output, conv2_output = forward_pass_cnn(input_image, conv1, conv2)
            output, hidden1, hidden2 = forward_pass_ann(flattened_output, ann1_weights, ann2_weights, ann3_weights)

            # Calculate loss and accuracy
            loss = np.square(output - label).sum()
            total_loss += loss
            total_accuracy += int(np.argmax(output) == np.argmax(label))

            # Backward pass
            output_error = output - label
            ann1_input_delta, hidden1, hidden2 = backward_pass_ann(output_error, hidden1, hidden2, flattened_output, ann1_weights, ann2_weights, ann3_weights, learning_rate)
            backward_pass_cnn(input_image, ann1_input_delta, conv1, conv2, max_pool1_output, conv1_output, learning_rate)

        # Print epoch loss and accuracy
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}, Accuracy: {total_accuracy / len(input_images)}')




# Create dummy dataset
num_samples = 100
input_images = np.random.randn(num_samples, 1, 28, 28)  # 100 samples of 28x28 grayscale images
labels = np.eye(10)[np.random.randint(0, 10, num_samples)]  # One-hot encoded random labels

# Train the network
num_epochs = 10
learning_rate = 0.001
train_network(input_images, labels, num_epochs, learning_rate)

