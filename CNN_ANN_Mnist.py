# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:05:28 2023

@author: brobm
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

# Set the number of training and test samples to be used
num_train_samples = 10000  # Set the desired number of training samples
num_test_samples = 2000    # Set the desired number of testing samples

# Reduce the number of training and test images
x_train = x_train[:num_train_samples]
y_train = y_train[:num_train_samples]
x_test = x_test[:num_test_samples]
y_test = y_test[:num_test_samples]

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
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

# -------------------------- FCLayer ----------------------------

# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
        
# -------------------------- FCLayer ----------------------------

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

# -------------------------- ConvLayer ----------------------------

# inherit from base class Layer
# This convolutional layer is always with stride 1
class ConvLayer(Layer):
    # input_shape = (i,j,d)
    # kernel_shape = (m,n)
    # layer_depth = output_depth
    def __init__(self, input_shape, kernel_shape, layer_depth):
        self.input_shape = input_shape
        self.input_depth = input_shape[2]
        self.kernel_shape = kernel_shape
        self.layer_depth = layer_depth
        self.output_shape = (input_shape[0]-kernel_shape[0]+1, input_shape[1]-kernel_shape[1]+1, layer_depth)
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.input_depth, layer_depth) - 0.5
        self.bias = np.random.rand(layer_depth) - 0.5

    # returns output for a given input
    def forward_propagation(self, input):
        self.input = input
        self.output = np.zeros(self.output_shape)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                self.output[:,:,k] += signal.correlate2d(self.input[:,:,d], self.weights[:,:,d,k], 'valid') + self.bias[k]

        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        in_error = np.zeros(self.input_shape)
        dWeights = np.zeros((self.kernel_shape[0], self.kernel_shape[1], self.input_depth, self.layer_depth))
        dBias = np.zeros(self.layer_depth)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:,:,d] += signal.convolve2d(output_error[:,:,k], self.weights[:,:,d,k], 'full')
                dWeights[:,:,d,k] = signal.correlate2d(self.input[:,:,d], output_error[:,:,k], 'valid')
            dBias[k] = self.layer_depth * np.sum(output_error[:,:,k])

        self.weights -= learning_rate*dWeights
        self.bias -= learning_rate*dBias
        return in_error

# -------------------------- FlattenLayer ----------------------------

# inherit from base class Layer
class FlattenLayer(Layer):
    # returns the flattened input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.flatten().reshape((1,-1))
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return output_error.reshape(self.input.shape)

# -------------------------- ActivationLayer ----------------------------

# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

# ---------- Activation function and its derivative ---------------------

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

# ---------- Loss function and its derivative ---------------------

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

# ---------- MaxPoolingLayer ---------------------

class MaxPoolingLayer(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward_propagation(self, input_data):
        self.input = input_data
        input_shape = input_data.shape
        output_shape = (input_shape[0] // self.stride, input_shape[1] // self.stride, input_shape[2])

        self.output = np.zeros(output_shape)
        for i in range(0, input_shape[0], self.stride):
            for j in range(0, input_shape[1], self.stride):
                self.output[i//self.stride, j//self.stride] = np.max(input_data[i:i+self.pool_size, j:j+self.pool_size], axis=(0, 1))

        return self.output

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


# ---------------------------------------------------



# Network
'''
net = Network()
net.add(ConvLayer((28, 28, 1), (3, 3), 1))  # input_shape=(28, 28, 1)   ;   output_shape=(26, 26, 1) 
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FlattenLayer())                     # input_shape=(26, 26, 1)   ;   output_shape=(1, 26*26*1)
net.add(FCLayer(26*26*1, 100))              # input_shape=(1, 26*26*1)  ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 10))                   # input_shape=(1, 100)      ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))
'''

# Network
net = Network()
net.add(ConvLayer((28, 28, 1), (5, 5), 6))  # input_shape=(28, 28, 1)   ;   output_shape=(24, 24, 6)
net.add(MaxPoolingLayer())                 # input_shape=(24, 24, 6)   ;   output_shape=(12, 12, 6)
net.add(ConvLayer((12, 12, 6), (5, 5), 6))  # input_shape=(12, 12, 6)   ;   output_shape=(8, 8, 6)
net.add(MaxPoolingLayer())                 # input_shape=(8, 8, 6)     ;   output_shape=(4, 4, 6)
net.add(FlattenLayer())                     # input_shape=(4, 4, 6)     ;   output_shape=(1, 4*4*6)
net.add(FCLayer(4 * 4 * 6, 60))             # input_shape=(1, 4*4*6)    ;   output_shape=(1, 60)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(60, 40))                    # input_shape=(1, 60)       ;   output_shape=(1, 40)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(40, 10))                    # input_shape=(1, 40)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))


# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=100, learning_rate=0.1)

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])