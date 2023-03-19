# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 22:05:24 2023

@author: Oualid Burstrom
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import mnist

# Applies 2D convolution on an image using a kernel with specified stride, padding, and pooling
# The input image can be 2D or 3D, and the kernel must be a 2D array
# If padding is 'same', the function pads the input image to ensure that the output has the same spatial dimensions as the input
# If pooling is 'max', the function applies max pooling with a specified pool size to the output of the convolution
# Returns the output image as a 2D NumPy array

def conv2D(image, kernel, stride=(1, 1), padding='valid', pooling='max', pool_size=(2, 2)):
    # Convert image and kernel to CuPy arrays if they are not already
    image = cp.array(image)
    kernel = cp.array(kernel)

    if padding == 'same':
        pad = ((kernel.shape[0] - 1) // 2, (kernel.shape[1] - 1) // 2)
        image = cp.pad(image, ((pad[0], pad[0]), (pad[1], pad[1])), 'constant')

    output_shape = ((image.shape[0] - kernel.shape[0]) // stride[0] + 1,
                    (image.shape[1] - kernel.shape[1]) // stride[1] + 1)
    output = cp.zeros(output_shape)

    for i in range(0, image.shape[0] - kernel.shape[0] + 1, stride[0]):
        for j in range(0, image.shape[1] - kernel.shape[1] + 1, stride[1]):
            output[int(i/stride[0]), int(j/stride[1])] = cp.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)

    if pooling == 'max':
        pooled_output = cp.zeros((output_shape[0] // pool_size[0], output_shape[1] // pool_size[1]))
        for i in range(0, output_shape[0], pool_size[0]):
            for j in range(0, output_shape[1], pool_size[1]):
                pooled_output[int(i/pool_size[0]), int(j/pool_size[1])] = cp.max(output[i:i+pool_size[0], j:j+pool_size[1]])
        return pooled_output
    else:
        return output



# Define a convolutional neural network with two convolutional layers
# Each convolutional layer has six filters and uses max pooling
# The input image is convolved with six different 5x5 kernels in the first layer
# The output of the second convolutional layer is flattened and returned

def conv_net(image):
    # Define the kernel and filter size
    kernel_size = (5, 5)
    
    # Create six different kernels
    kernels = [cp.clip(cp.random.randn(*kernel_size),-0.5, 0.5) for _ in range(6)]  

    # First convolutional layer with six filters
    conv1 = [conv2D(image, kernel, stride=(1, 1), padding='valid', pooling='max', pool_size=(2, 2)) for kernel in kernels]
    
    # Second convolutional layer with six filters
    conv2 = [conv2D(conv, kernel, stride=(1, 1), padding='valid', pooling='max', pool_size=(2, 2)) for kernel, conv in zip(kernels, conv1)]

    # Flatten the output of the second convolutional layer
    flattened = cp.concatenate(conv2).flatten()

    return flattened

num_train_samples = 30000  # Set the desired number of training samples
num_test_samples =  15000    # Set the desired number of testing samples

# Load the MNIST dataset and limit the number of samples for train and test
train_images = mnist.train_images()[:num_train_samples]
train_labels = mnist.train_labels()[:num_train_samples]
test_images = mnist.test_images()[:num_test_samples]
test_labels = mnist.test_labels()[:num_test_samples]

# Flatten the input images
#train_images = train_images.reshape(-1, 784)
#test_images = test_images.reshape(-1, 784)

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Apply conv_net to the train and test images
train_images = cp.array([conv_net(img) for img in train_images])
test_images = cp.array([conv_net(img) for img in test_images])

# Convert the data to CuPy arrays
train_images = cp.asarray(train_images)
test_images = cp.asarray(test_images)
train_labels = cp.asarray(train_labels)
test_labels = cp.asarray(test_labels)

# Define the activation function (ReLU) and its derivative
def relu(x):
    return cp.maximum(0, x)

def relu_derivative(x):
    return cp.where(x > 0, 1, 0)


# Define the activation function (sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

input_size = train_images.shape[1]  # The output size of conv_net function
hidden_size = 128
output_size = 10


# Define the network architecture
#input_size = 784  # 28x28
#hidden_size = 128
hidden_size1 = 128
hidden_size2 = 64
#output_size = 10

hidden_layer_1 = 128
hidden_layer_2 = 64

# Initialize the weights and biases using Xavier initialization when using ReLU
# Initialization of weights and biases
w1 = cp.random.randn(input_size, hidden_layer_1) * cp.sqrt(2 / input_size)
b1 = cp.zeros((1, hidden_layer_1))

w2 = cp.random.randn(hidden_layer_1, hidden_layer_2) * cp.sqrt(2 / hidden_layer_1)
b2 = cp.zeros((1, hidden_layer_2))

w3 = cp.random.randn(hidden_layer_2, output_size) * cp.sqrt(2 / hidden_layer_2)
b3 = cp.zeros((1, output_size))
'''
# Initialize the weights and biases randomly when using sigmoid
w1 = cp.random.randn(input_size, hidden_size1)
b1 = cp.random.randn(hidden_size1)
w2 = cp.random.randn(hidden_size1, hidden_size2)
b2 = cp.random.randn(hidden_size2)
w3 = cp.random.randn(hidden_size2, output_size)
b3 = cp.random.randn(output_size)
'''

# Define the hyperparameters
# Use ReLU or sigmoid
#learning_rate = 1.0 # sigmoid
learning_rate = 0.1 # ReLU
num_epochs = 50

# Initialize lists to store the loss and accuracy for each epoch
train_loss = []
train_accuracy = []

# Trains a neural network with three layers (one input layer, one hidden layer, and one output layer)
# Uses stochastic gradient descent with backpropagation to update the weights and biases
# Computes the loss and accuracy for each epoch and appends them to lists for plotting
# Prints the loss and accuracy every 10 epochs
for epoch in range(num_epochs):
    # Forward pass
    z1 = cp.dot(train_images, w1) + b1
    a1 = relu(z1)
    #a1 = sigmoid(z1)
    z2 = cp.dot(a1, w2) + b2
    a2 = relu(z2)
    #a2 = sigmoid(z2)
    z3 = cp.dot(a2, w3) + b3
    output = cp.exp(z3) / cp.sum(cp.exp(z3), axis=1, keepdims=True)

    # Compute the loss and accuracy
    loss = -cp.sum(cp.log(output[range(len(train_labels)), train_labels])) / len(train_labels)
    predicted_labels = cp.argmax(output, axis=1)
    accuracy = cp.mean(predicted_labels == train_labels)

    # Backward pass
    dz3 = output
    dz3[range(len(train_labels)), train_labels] -= 1
    dw3 = cp.dot(a2.T, dz3) / len(train_labels)
    db3 = cp.sum(dz3, axis=0) / len(train_labels)
    da2 = cp.dot(dz3, w3.T)
    dz2 = da2 * relu_derivative(z2)
    #dz2 = da2 * sigmoid_derivative(z2)
    dw2 = cp.dot(a1.T, dz2) / len(train_labels)
    db2 = cp.sum(dz2, axis=0) / len(train_labels)
    da1 = cp.dot(dz2, w2.T)
    dz1 = da1 * relu_derivative(z1)
    #dz1 = da1 * sigmoid_derivative(z1)
    dw1 = cp.dot(train_images.T, dz1) / len(train_labels)
    db1 = cp.sum(dz1, axis=0) / len(train_labels)

    # Update the weights and biases
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    
    # Append the loss and accuracy to the lists
    train_loss.append(loss)
    train_accuracy.append(accuracy)    

    # Print the loss and accuracy every 10 epochs
    if epoch % 10 == 0:
        #print("Epoch {}/{} - loss: {:.4f} - accuracy: {:.4f}".format(epoch+1, num_epochs, loss, accuracy))
        print("Epoch {}/{} - loss: {:.4f} - accuracy: {:.4f}".format(epoch+1, num_epochs, cp.asnumpy(loss), cp.asnumpy(accuracy)))

# Evaluate the network on the test set
z1 = cp.dot(test_images, w1) + b1
a1 = relu(z1)
#a1 = sigmoid(z1)
z2 = cp.dot(a1, w2) + b2
a2 = relu(z2)
#a2 = sigmoid(z2)
z3 = cp.dot(a2, w3) + b3
output = cp.exp(z3) / cp.sum(cp.exp(z3), axis=1, keepdims=True)
predicted_labels = cp.argmax(output, axis=1)
accuracy = cp.mean(predicted_labels == test_labels)
#print("Test accuracy: {:.4f}".format(accuracy))
print("Test accuracy: {:.4f}".format(cp.asnumpy(accuracy))) 


# Evaluate the network on the test set
num_correct = 0
for i in range(len(test_images)):
    # Forward pass
    z1 = cp.dot(test_images[i], w1) + b1
    a1 = relu(z1)
    #a1 = sigmoid(z1)
    z2 = cp.dot(a1, w2) + b2
    a2 = relu(z2)
    #a2 = sigmoid(z2)
    z3 = cp.dot(a2, w3) + b3
    output = cp.exp(z3) / cp.sum(cp.exp(z3))

    # Compute the predicted label and check if it's correct
    predicted_label = cp.argmax(output)
    true_label = test_labels[i]
    if predicted_label == true_label:
        num_correct += 1

    # Print the predicted label and true label
    print("Image {} - predicted label: {}, true label: {}".format(i+1, predicted_label, true_label))

# Print the overall accuracy
accuracy = num_correct / len(test_images)
print("Test accuracy: {:.4f}".format(accuracy))


# Plot the training loss
plt.figure(figsize=(6, 4))
plt.plot(train_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# Plot the training accuracy
plt.figure(figsize=(6, 4))
plt.plot(train_accuracy)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.show()
