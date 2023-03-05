# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:40:52 2023

@author: Budokan
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define a function to normalize the pixel values of an image array
def normalize(X):
    mean = np.mean(X)
    s = np.std(X)
    if s != 0:
        X = (X - mean) / s
    else:
        X = X / 255 - 0.5
    return X

# Load the training and test data from image files
train_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] 
test_labels = [0, 1, 2, 3, 4]       

nr_training = len(train_labels)
nr_tests = len(test_labels)

X_train = np.zeros((nr_training, 25)) 
y_train = np.zeros((nr_training, 5))

X_test = np.zeros((nr_tests, 25))
y_test = np.zeros((nr_tests, 5))

for i in range(nr_training):
    filename = f"data/{i+1}t.jpg"
    img = Image.open(filename).convert('L')
    X = np.asarray(img).flatten()
    X_train[i] = normalize(X) 
    y_train[i, train_labels[i]] = 1   

for i in range(nr_tests):
    filename = f"data/{i+1}.jpg"
    img = Image.open(filename).convert('L')
    X = np.asarray(img).flatten()
    X_test[i] = normalize(X)
    y_test[i, test_labels[i]] = 1  

# Define the neural network architecture
size_data = 25
size_hidden = 5

input_w = np.random.randn(size_data, size_hidden) - 0.5
b1 = 0.35
b2 = 0.6

# Define helper functions for the neural network
def sigmoid(z):  
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z)) 

# Define functions for the forward and backward pass of the neural network
def forward_pass(weight1, input, bias):
    z = np.matmul(input, weight1) + bias
    return z, sigmoid(z)

def backward_init(target, out, hout, hidden_w):
    dEtot_dout_o = -(target - out)
    dout_o_dnet = sigmoid_derivative(out)
    dnet_dw = hout
    delta0 = dEtot_dout_o * dout_o_dnet
    diff = np.matmul(dnet_dw.T, delta0)
    hidden_w -= 0.5 * diff
    return hidden_w

# Train the neural network using the backpropagation algorithm
Etot = 0
for i in range(100):
    Etot_old = Etot
    for j in range(nr_training):
        net1b, out = forward_pass(input_w, np.array([X_train[j]]), b2)
        Etot = np.sum(0.5 * np.power((y_train[j] - out), 2))
        input_wo = input_w
        input_w = backward_init(y_train[j], out, np.array([X_train[j]]), input_wo)
    print(Etot)
    plt.plot([i-1, i], [Etot_old, Etot], 'b')

# Test the neural network on the test data and display the confusion matrix
confusion_matrix = []
for i in range(nr_tests):
    net1b, out = forward_pass(input_w, np.array([X_test[i]]), b2)
    confusion_matrix.append(out)

confusion_matrix = np.array(confusion_matrix)
print(np.round(confusion_matrix, 2))
