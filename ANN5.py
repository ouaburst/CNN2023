# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 20:52:49 2023

@author: Budokan
"""

import numpy as np
import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset
train_images = mnist.train_images()[:100]
train_labels = mnist.train_labels()[:100]
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Initialize weights and biases for the neural network
input_size = 784 # 28x28 pixels
hidden_size = [128, 64, 32, 16, 8]
output_size = 10 # 0-9 digits
weights = [np.random.randn(input_size, hidden_size[0])]
biases = [np.zeros((1, hidden_size[0]))]
for i in range(1, 5):
    weights.append(np.random.randn(hidden_size[i-1], hidden_size[i]))
    biases.append(np.zeros((1, hidden_size[i])))
weights.append(np.random.randn(hidden_size[-1], output_size))
biases.append(np.zeros((1, output_size)))

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward pass function
def forward_pass(input_data, weights, biases):
    hidden_activations = []
    hidden_inputs = []
    output_activation = input_data
    for i in range(len(weights)):
        hidden_input = np.dot(output_activation, weights[i]) + biases[i]
        hidden_inputs.append(hidden_input)
        output_activation = sigmoid(hidden_input)
        hidden_activations.append(output_activation)
    return output_activation, hidden_activations, hidden_inputs

# Backpropagation algorithm
def backpropagation(input_data, target_output, weights, biases, learning_rate):
    output_activation, hidden_activations, hidden_inputs = forward_pass(input_data, weights, biases)
    error = output_activation - target_output
    for i in range(len(weights)-1, -1, -1):
        if i == len(weights)-1:
            delta = error * output_activation * (1 - output_activation)
        else:
            delta = np.dot(delta, weights[i+1].T) * hidden_activations[i] * (1 - hidden_activations[i])
        weights[i] -= learning_rate * np.dot(hidden_activations[i-1].T, delta) if i > 0 else learning_rate * np.dot(input_data.T, delta)
        biases[i] -= learning_rate * np.sum(delta, axis=0, keepdims=True)
    return weights, biases

# Train the neural network
num_epochs = 100
learning_rate = 0.1
for epoch in range(num_epochs):
    for image, label in zip(train_images, train_labels):
        input_data = image.reshape(1, -1) / 255.0
        target_output = np.zeros((1, output_size))
        target_output[0, label] = 1
        weights, biases = backpropagation(input_data, target_output, weights, biases, learning_rate)
    print(f"Epoch {epoch+1} completed")

# Test the neural network
num_correct = 0
for image, label in zip(test_images, test_labels):
    input_data = image.reshape(1, -1) / 255.0
    output_activation, _, _ = forward_pass(input_data, weights, biases)
    predicted_label = np.argmax(output_activation)
    if predicted_label == label:
        num_correct += 1
accuracy = num_correct / len(test_images)
print(f"Accuracy: {accuracy}")
