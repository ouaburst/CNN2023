import numpy as np
import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset
train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:10]
test_labels = mnist.test_labels()[:10]

# Define neural network architecture
input_size = 784
hidden_sizes = [128, 64, 32, 16, 8]
output_size = 10

# Initialize weights and biases
weights = []
biases = []
for i in range(len(hidden_sizes)+1):
    if i == 0:
        w = np.random.randn(input_size, hidden_sizes[0]) / np.sqrt(input_size)
        b = np.zeros((1, hidden_sizes[0]))
    elif i == len(hidden_sizes):
        w = np.random.randn(hidden_sizes[-1], output_size) / np.sqrt(hidden_sizes[-1])
        b = np.zeros((1, output_size))
    else:
        w = np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) / np.sqrt(hidden_sizes[i-1])
        b = np.zeros((1, hidden_sizes[i]))
    weights.append(w)
    biases.append(b)

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x/100))


# Define derivative of sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define forward pass function
def forward_pass(x):
    activations = []
    x = x.reshape(1, input_size)
    activations.append(x)
    for i in range(len(hidden_sizes)+1):
        z = np.dot(x, weights[i]) + biases[i]
        x = sigmoid(z)
        activations.append(x)
    return activations

# Define backpropagation algorithm
def backpropagation(x, y, activations):
    deltas = []
    delta = (activations[-1] - y) * sigmoid_derivative(np.dot(activations[-2], weights[-1]) + biases[-1])
    deltas.append(delta)
    for i in range(len(hidden_sizes)-1, -1, -1):
        delta = np.dot(deltas[-1], weights[i+1].T) * sigmoid_derivative(np.dot(activations[i], weights[i]) + biases[i])
        deltas.append(delta)
    deltas.reverse()
    for i in range(len(weights)):
        weights[i] -= np.dot(activations[i].T, deltas[i])
        biases[i] -= deltas[i]

# Train neural network
epochs = 50
learning_rate = 0.1
for e in range(epochs):
    for i in range(len(train_images)):
        x = train_images[i]
        y = np.zeros(output_size)
        y[train_labels[i]] = 1
        activations = forward_pass(x)
        backpropagation(x, y, activations)
    print("Epoch: ", e+1)

# Evaluate neural network
correct = 0
for i in range(len(test_images)):
    x = test_images[i]
    y = test_labels[i]
    activations = forward_pass(x)
    prediction = np.argmax(activations[-1])
    if prediction == y:
        correct += 1
accuracy = correct / len(test_images)
print("Accuracy: ", accuracy)
