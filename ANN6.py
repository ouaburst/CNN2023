# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:59:07 2023

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

# train neural network
for i in range(0, 100):
    Etot_old = Etot
    for j in range(0, nr_training):
        [net1b, out] = forward_pass(input_w, np.array([X_train[j]]), b2)
        Etot = np.sum(0.5 * np.power((y_train_onehot[j] - out), 2))
        input_wo = input_w
        input_w = backward_init(y_train_onehot[j], out, np.array([X_train[j]]), input_wo)
    print(Etot)
    plt.plot([i - 1, i], [Etot_old, Etot], 'b')

# evaluate neural network on test data
confusionmatrix = np.zeros((nr_labels, nr_labels))
for j in range(0, nr_tests):
    [net1b, out] = forward_pass(input_w, np.array([X_test[j]]), b2)
    predicted_label = np.argmax(out)
    true_label = y_test[j]
    confusionmatrix[true_label, predicted_label] += 1

print(confusionmatrix)

