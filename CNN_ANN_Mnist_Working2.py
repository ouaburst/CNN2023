# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:05:28 2023

@author: Oualid Burström
"""

import numpy as np
from scipy import signal
import mnist

# -------------------------- Import Mnist ----------------------------

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

# -------------------------- Network ----------------------------

class Network:
    # Constructor: Initialize network properties
    def __init__(self):
        self.layers = []  # Initialize the list to store layers in the network
        self.loss = None  # Placeholder for the loss function
        self.loss_prime = None  # Placeholder for the derivative of the loss function

    # Add a layer to the network
    def add(self, layer):
        self.layers.append(layer)

    # Set the loss function and its derivative for the network
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # Generate predictions for the given input data
    def predict(self, input_data):
        samples = len(input_data)  # Number of input samples
        result = []  # Initialize the list to store prediction results

        # Iterate over all input samples
        for i in range(samples):
            # Forward propagation through the network
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)  # Add the output to the result list

        return result  # Return the list of predictions

    # Train the network using the given training data, number of epochs, and learning rate
    def train(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)  # Number of training samples

        # Iterate over the specified number of epochs
        for i in range(epochs):
            err = 0  # Initialize the error to 0 for each epoch

            # Iterate over all training samples
            for j in range(samples):
                output = x_train[j]
                # Forward propagation through the network
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y_train[j], output)  # Accumulate the error for the current sample

                # Backpropagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples  # Calculate the average error for the epoch
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))  # Print the epoch number and the average error


# -------------------------- Layer ----------------------------

# Base class for neural network layers
class Layer:
    # Constructor initializing input and output attributes
    def __init__(self):
        self.input = None
        self.output = None

    # Forward propagation method (to be implemented by subclasses)
    def forward_propagation(self, input):
        raise NotImplementedError

    # Backward propagation method (to be implemented by subclasses)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

        
# -------------------------- FCLayer ----------------------------

# This is a class definition for a fully connected layer in a neural network.

class FCLayer(Layer):

# This is the constructor method that initializes the weights and biases with random values.
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

# This method implements the forward propagation step for the fully connected layer.
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

# This method implements the backward propagation step for the fully connected layer.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
    
        # Update the weights and biases using the gradient descent algorithm.
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


# -------------------------- ConvLayer ----------------------------

# This class represents a Convolutional Layer in a Convolutional Neural Network.

class ConvLayer(Layer):
    # Initialize the ConvLayer with input shape, kernel shape, layer depth and optional padding.
    def __init__(self, input_shape, kernel_shape, layer_depth, padding=0):
        self.input_shape = input_shape
        self.input_depth = input_shape[2]
        self.kernel_shape = kernel_shape
        self.layer_depth = layer_depth
        self.padding = padding
        self.output_shape = (input_shape[0] - kernel_shape[0] + 2 * padding + 1,
                             input_shape[1] - kernel_shape[1] + 2 * padding + 1,
                             layer_depth)
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.input_depth, layer_depth) - 0.5
        self.bias = np.random.rand(layer_depth) - 0.5

    # Pad the input with zeros, if padding is specified.    
    def pad_input(self, input):
        if self.padding > 0:
            padded_input = np.pad(input, ((self.padding, self.padding),
                                           (self.padding, self.padding),
                                           (0, 0)), mode='constant', constant_values=0)
        else:
            padded_input = input
        return padded_input

    # Perform forward propagation for a given input and return the output.
    def forward_propagation(self, input):
        self.input = self.pad_input(input)
        self.output = np.zeros(self.output_shape)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                self.output[:,:,k] += signal.correlate2d(self.input[:,:,d], self.weights[:,:,d,k], 'valid') + self.bias[k]

        return self.output

    # Perform backward propagation, update the weights and bias, and return the input error.
    def backward_propagation(self, output_error, learning_rate):
        if self.padding > 0:
            padded_input = self.input
        else:
            padded_input = self.pad_input(self.input)
        in_error = np.zeros(padded_input.shape)
        dWeights = np.zeros((self.kernel_shape[0], self.kernel_shape[1], self.input_depth, self.layer_depth))
        dBias = np.zeros(self.layer_depth)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:,:,d] += signal.convolve2d(output_error[:,:,k], self.weights[:,:,d,k], 'full')
                dWeights[:,:,d,k] = signal.correlate2d(padded_input[:,:,d], output_error[:,:,k], 'valid')
            dBias[k] = self.layer_depth * np.sum(output_error[:,:,k])

        self.weights -= learning_rate*dWeights
        self.bias -= learning_rate*dBias

        if self.padding > 0:
            in_error = in_error[self.padding:-self.padding, self.padding:-self.padding, :]

        return in_error

# -------------------------- FlattenLayer ----------------------------

# This class represents a Flatten Layer in a Neural Network, which is used to convert multidimensional input data
# into a one-dimensional array.
class FlattenLayer(Layer):
    # Perform forward propagation by flattening the input data and reshaping it into a one-dimensional array.
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.flatten().reshape((1,-1))
        return self.output

    # Perform backward propagation by reshaping the output error back to the original input shape.
    def backward_propagation(self, output_error, learning_rate):
        return output_error.reshape(self.input.shape)

# -------------------------- ActivationLayer ----------------------------

# This class represents an Activation Layer in a Neural Network, which applies a given activation function
# and its derivative to the input data.
class ActivationLayer(Layer):
    # Initialize the ActivationLayer with the specified activation function and its derivative.
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # Perform forward propagation by applying the activation function to the input data.
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Perform backward propagation by applying the derivative of the activation function to the output error.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

# ---------- Activation function and its derivative ---------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# ---------- Loss function and its derivative ---------------------

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_derivative(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

# ---------- MaxPoolingLayer ---------------------

# This class represents a Max Pooling Layer in a Convolutional Neural Network, which is used to downsample
# the input data while preserving its most important features.
class MaxPoolingLayer(Layer):
    # Initialize the MaxPoolingLayer with the specified pool size and stride.
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    # Perform forward propagation by applying max pooling to the input data.
    def forward_propagation(self, input_data):
        self.input = input_data
        input_shape = input_data.shape
        output_shape = (input_shape[0] // self.stride, input_shape[1] // self.stride, input_shape[2])

        self.output = np.zeros(output_shape)
        for i in range(0, input_shape[0], self.stride):
            for j in range(0, input_shape[1], self.stride):
                self.output[i//self.stride, j//self.stride] = np.max(input_data[i:i+self.pool_size, j:j+self.pool_size], axis=(0, 1))

        return self.output

    # Perform backward propagation by calculating the input error using the output error and the input data.
    def backward_propagation(self, output_error, learning_rate):
        input_shape = self.input.shape
        input_error = np.zeros(input_shape)

        for i in range(0, input_shape[0], self.stride):
            for j in range(0, input_shape[1], self.stride):
                for k in range(input_shape[2]):
                    patch = self.input[i:i+self.pool_size, j:j+self.pool_size, k]
                    max_val_idx = np.unravel_index(np.argmax(patch, axis=None), patch.shape)
                    input_error[i+max_val_idx[0], j+max_val_idx[1], k] = output_error[i//self.stride, j//self.stride, k]

        return input_error

# -------------------- The Network -------------------------

net = Network()
net.add(ConvLayer((28, 28, 1), (5, 5), 6, padding=2))       # input_shape=(28, 28, 1)   ;   output_shape=(24, 24, 6)
net.add(MaxPoolingLayer(pool_size=2, stride=2))             # input_shape=(24, 24, 6)   ;   output_shape=(14, 14, 6)
net.add(ConvLayer((14, 14, 6), (5, 5), 6))                  # input_shape=(14, 14, 6)   ;   output_shape=(10, 10, 6)
net.add(MaxPoolingLayer())                                  # input_shape=(10, 10, 6)   ;   output_shape=(5, 5, 6)
net.add(FlattenLayer())                                     # input_shape=(5, 5, 6)     ;   output_shape=(1, 5*5*6)
net.add(FCLayer(5 * 5 * 6, 60))                             # input_shape=(1, 5*5*6)    ;   output_shape=(1, 60)
net.add(ActivationLayer(sigmoid, sigmoid_derivative))
net.add(FCLayer(60, 40))                                    # input_shape=(1, 60)       ;   output_shape=(1, 40)
net.add(ActivationLayer(sigmoid, sigmoid_derivative))
net.add(FCLayer(40, 10))                                    # input_shape=(1, 40)       ;   output_shape=(1, 10)
net.add(ActivationLayer(sigmoid, sigmoid_derivative))

# Set the loss function for the neural network using the mean squared error (mse) and its derivative (mse_derivative).
net.use(mse, mse_derivative)

# Train the neural network (net) using a subset of the training data (first 1000 samples) with 10 epochs and a learning rate of 0.1.
net.train(x_train[0:1000], y_train[0:1000], epochs=10, learning_rate=0.1)

# Test on 5 samples
out = net.predict(x_test[0:5])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:5])
