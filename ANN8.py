# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 20:32:02 2023

@author: Budokan
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import pyplot
import os

def normalise(X):
    mean = np.mean(X)
    s = np.std(X)
    if s != 0:
        return np.divide(np.subtract(X, mean), s)
    else:
        return X/255 - 0.5

def load_mnist():
    # load MNIST training data
    X_train = []
    y_train = []
    for i in range(10):
        path = os.path.join("mnist_jpg/", str(i))
        for filename in os.listdir(path):
            img = Image.open(os.path.join(path, filename)).convert('L')
            img.load()
            X = np.asarray(img)
            X = X.flatten()
            X = normalise(X)
            X_train.append(X)
            y_train.append(i)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # load MNIST test data
    X_test = []
    y_test = []
    path = os.path.join("mnist_jpg/", "test")
    for filename in os.listdir(path):
        img = Image.open(os.path.join(path, filename)).convert('L')
        img.load()
        X = np.asarray(img)
        X = X.flatten()
        X = normalise(X)
        X_test.append(X)
        y_test.append(int(filename[0]))
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return (X_train, y_train), (X_test, y_test)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def forward_pass(weight1, input, b1):
    Z = np.matmul(input, weight1) + b1
    return Z, sigmoid(Z)

def backward_init(target, out, hout, hidden_w):
    dEtot_dout_o = -(target - out)
    dout_o_dnet = sigmoid_derivative(out)
    dnet_dw = hout
    delta0 = dEtot_dout_o * dout_o_dnet
    diff = np.matmul(dnet_dw.T, delta0)
    hidden_w = hidden_w - 0.5 * diff
    return hidden_w

# load MNIST data
(X_train, y_train), (X_test, y_test) = load_mnist()

# set up labels
nr_training = X_train.shape[0]
nr_tests = X_test.shape[0]
nr_labels = 10
y_train_onehot = np.zeros((nr_training, nr_labels))
y_train_onehot[np.arange(nr_training), y_train] = 1
y_test_onehot = np.zeros((nr_tests, nr_labels))
y_test_onehot[np.arange(nr_tests), y_test] = 1

# set up neural network
input_w = np.random.randn(X_train.shape[1], 5) - 0.5
b1 = 0.35
b2 = 0.6
Etot = 0.1

# set up training parameters
learning_rate = 0.1
num_epochs = 1000

hidden_w = np.random.randn(5, nr_labels) - 0.5

# training loop
for epoch in range(num_epochs):
    # perform forward propagation on training data
    hidden_input, hidden_output = forward_pass(input_w, X_train, b1)
    output_input, output = forward_pass(hidden_w, hidden_output, b2)

    # calculate error and update Etot
    error = np.mean(np.abs(y_train_onehot - output))
    Etot += error

    # perform backpropagation to update weights and biases
    delta1 = np.multiply((y_train_onehot - output), sigmoid_derivative(output))
    dweight2 = np.matmul(hidden_output.T, delta1)
    dhidden_output = np.matmul(delta1, hidden_w.T)
    delta0 = np.multiply(dhidden_output, sigmoid_derivative(hidden_output))
    dweight1 = np.matmul(X_train.T, delta0)

    hidden_w += learning_rate * dweight2
    input_w += learning_rate * dweight1
    

    if epoch % 100 == 0:
        print("Epoch: %d, Error: %f" % (epoch, error))

# perform forward propagation on test data
hidden_input, hidden_output = forward_pass(input_w, X_test, b1)
output_input, output = forward_pass(hidden_w, hidden_output, b2)

# calculate error on test data
test_error = np.mean(np.abs(y_test_onehot - output))
print("Test Error: %f" % test_error)