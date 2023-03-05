# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 17:35:42 2023

@author: Budokan
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import pyplot

def normalize(X):
    mean = np.mean(X)
    s = np.std(X)
    if s != 0:
        return np.divide(np.subtract(X, mean), s)
    else:
        return X/255 - 0.5

# Set up variables
nr_letters = 5
nr_training = 10
nr_tests = 5
correct = np.zeros((nr_tests))

# Labels for training and testing data
train_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
test_labels = [0, 1, 2, 3, 4]

#Initialize arrays for training and testing data
y_train = np.zeros((nr_training, nr_letters))
y_test = np.zeros((nr_tests, nr_letters))
X_train = np.zeros((nr_training, 25))
X_test = np.zeros((nr_tests, 25))

#Load and preprocess training data
for i in range(nr_training):
    filename = f"{i+1}t.jpg"
    img = Image.open(filename).convert('L')
    img.load()
    X = np.asarray(img)
    X = X.flatten()
    X = np.array(X)
    X = normalize(X)
    X_train[i] = X
    y = train_labels[i]
    y_train[i, y] = 1

# Load and preprocess testing data
for i in range(nr_tests):
    filename = f"{i+1}.jpg"
    img = Image.open(filename).convert('L')
    img.load()
    y = test_labels[i]
    y_test[i, y] = 1
    X = np.asarray(img)
    X = X.flatten()
    X = np.array(X)
    X = normalize(X)
    X_test[i] = X
    pyplot.imshow(img)
    plt.show()

# Set up ANN variables
input = X_train
target = y_train
b1 = 0.35
b2 = 0.6
Etot = .1
size_data = 25
size_hidden = 5
input_w = (np.random.randn(size_data, size_hidden)) - 0.5

#Define functions for forward and backward pass
def forward_pass(weight1, input, b1):
    Z = np.matmul(input, weight1) + b1
    return Z, sigmoid(Z)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1.0 - sigmoid(z))

def backward_init(target, out, hout, hidden_w):
    dEtot_dout_o = -(target - out)
    dout_o_dnet = sigmoid_derivative(out)
    dnet_dw = hout
    delta0 = dEtot_dout_o * dout_o_dnet
    diff = np.matmul(dnet_dw.T, delta0)
    diff_b1 = np.mean(dEtot_dout_o * dout_o_dnet, axis=0)
    delta1 = np.matmul(delta0, hidden_w.T) * sigmoid_derivative(hout)
    diff_w = np.matmul(X_train.T, delta1)
    diff_b2 = np.mean(delta0, axis=0)
    return diff_w, diff_b2, diff, diff_b1

# Train ANN
for i in range(nr_training):
    input_data = input[i:i+1, :]
    target_data = target[i:i+1, :]
    Z1, output = forward_pass(input_w, input_data, b1)




