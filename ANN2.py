# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:30:50 2023

@author: Budokan
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define a function to normalize the input data
def normalize(X):
    mean = np.mean(X)
    std = np.std(X)
    if std != 0:
        return (X - mean) / std
    else:
        return X / 255 - 0.5

# Define the number of samples and classes
n_train_samples = 10
n_test_samples = 5
n_classes = 5

# Define the labels for the training and testing data
train_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
test_labels = [0, 1, 2, 3, 4]

# Initialize arrays for the training and testing data and labels
X_train = np.zeros((n_train_samples, 25))
y_train = np.zeros((n_train_samples, n_classes))
X_test = np.zeros((n_test_samples, 25))
y_test = np.zeros((n_test_samples, n_classes))

# Load and preprocess the training data
for i in range(n_train_samples):
    filename = f"{i+1}t.jpg"
    img = Image.open(filename).convert("L")
    X = np.asarray(img).flatten()
    X = normalize(X)
    X_train[i] = X
    y_train[i, train_labels[i]] = 1

# Load and preprocess the testing data
for i in range(n_test_samples):
    filename = f"{i+1}.jpg"
    img = Image.open(filename).convert("L")
    X = np.asarray(img).flatten()
    X = normalize(X)
    X_test[i] = X
    y_test[i, test_labels[i]] = 1

# Define the network architecture
n_input = 25
n_hidden = 5
n_output = n_classes

# Initialize the network weights and biases
W1 = np.random.randn(n_input, n_hidden) - 0.5
b1 = 0.35
W2 = np.random.randn(n_hidden, n_output) - 0.5
b2 = 0.6

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Train the network using backpropagation
learning_rate = 0.5
n_epochs = 100

for epoch in range(n_epochs):
    # Initialize the total error for this epoch
    total_error = 0
    
    # Iterate over the training samples
    for i in range(n_train_samples):
        # Forward pass
        hidden_input = np.matmul(X_train[i], W1) + b1
        hidden_output = sigmoid(hidden_input)
        output = np.matmul(hidden_output, W2) + b2
        
        # Compute the error
        error = y_train[i] - output
        total_error += np.sum(error**2)
        
        # Backward pass
        delta_output = error * sigmoid_derivative(output)
        delta_hidden = np.matmul(delta_output, W2.T) * sigmoid_derivative(hidden_output)
        W2 += learning_rate * np.outer(hidden_output, delta_output)
        b2 += learning_rate * delta_output
        W1 += learning_rate * np.outer(X_train[i], delta_hidden)
        b1 += learning_rate * delta_hidden
    
    # Print the total error for this epoch
    print(f"Epoch {epoch+1}: error = {total_error}")

'''    
    # Plot the learning curve
    plt.plot(epoch, total_error, "bo")
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.show()
'''
    
# Make predictions on the testing data
confusion_matrix = np.zeros((n_classes, n_classes))

for i in range(n_test_samples):
    # Forward pass
    hidden_input = np.matmul(X_test[i], W1) + b1
    hidden_output = sigmoid(hidden_input)
    output = np.matmul(hidden_output, W2) + b2
    
    # Print the predicted class probabilities for this sample
    print(f"Sample {i+1}: {output}")
    
    # Update the confusion matrix
    predicted_class = np.argmax(output)
    true_class = np.argmax(y_test[i])
    confusion_matrix[true_class, predicted_class] += 1

# Print the confusion matrix
print("Confusion matrix:")
print(confusion_matrix)

    
