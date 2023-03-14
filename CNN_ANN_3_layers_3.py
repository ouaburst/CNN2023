# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 22:05:24 2023

@author: Oualid Burstrom
"""


import numpy as np
import matplotlib.pyplot as plt
import mnist

# Applies 2D convolution on an image using a kernel with specified stride, padding, and pooling
# The input image can be 2D or 3D, and the kernel must be a 2D array
# If padding is 'same', the function pads the input image to ensure that the output has the same spatial dimensions as the input
# If pooling is 'max', the function applies max pooling with a specified pool size to the output of the convolution
# Returns the output image as a 2D NumPy array

def conv2D(image, kernel, stride=(1, 1), padding='valid', pooling=None, pool_size=(2, 2)):

    # Check input dimensions
    assert len(image.shape) in [2, 3], "Input image must be 2D or 3D array"
    assert len(kernel.shape) == 2, "Kernel must be a 2D array"
    assert isinstance(stride, tuple) and len(stride) == 2, "Stride must be a tuple of two integers"
    if pooling:
        assert pooling == 'max', "Pooling must be 'max'"

    # Add padding if required
    if padding == 'same':
        pad_h = int(((image.shape[0]-1)*stride[0]+kernel.shape[0]-image.shape[0])/2)
        pad_w = int(((image.shape[1]-1)*stride[1]+kernel.shape[1]-image.shape[1])/2)
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    elif padding != 'valid':
        raise ValueError("Padding must be either 'valid' or 'same'")

    # Compute output shape
    out_h = int((image.shape[0]-kernel.shape[0])/stride[0] + 1)
    out_w = int((image.shape[1]-kernel.shape[1])/stride[1] + 1)
    output = np.zeros((out_h, out_w))

    # Perform convolution
    for i in range(0, image.shape[0]-kernel.shape[0]+1, stride[0]):
        for j in range(0, image.shape[1]-kernel.shape[1]+1, stride[1]):
            output[int(i/stride[0]), int(j/stride[1])] = np.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)

    # Apply pooling if required
    if pooling:
        pool_out_h = int(np.ceil(out_h / pool_size[0]))
        pool_out_w = int(np.ceil(out_w / pool_size[1]))
        pool_output = np.zeros((pool_out_h, pool_out_w))
        for i in range(0, out_h, pool_size[0]):
            for j in range(0, out_w, pool_size[1]):
                pool_window = output[i:i+pool_size[0], j:j+pool_size[1]]
                pool_output[int(i/pool_size[0]), int(j/pool_size[1])] = np.max(pool_window[:min(pool_size[0], pool_window.shape[0]), :min(pool_size[1], pool_window.shape[1])])
    output = pool_output

    return output


# Define a convolutional neural network with two convolutional layers
# Each convolutional layer has six filters and uses max pooling
# The input image is convolved with six different 5x5 kernels in the first layer
# The output of the second convolutional layer is flattened and returned

def conv_net(image):
    # Define the kernel and filter size
    kernel_size = (5, 5)
    
    # Create six different kernels
    kernels = [np.clip(np.random.randn(*kernel_size),-0.5, 0.5) for _ in range(6)]  

    # First convolutional layer with six filters
    conv1 = [conv2D(image, kernel, stride=(1, 1), padding='valid', pooling='max', pool_size=(2, 2)) for kernel in kernels]
    
    # Second convolutional layer with six filters
    conv2 = [conv2D(conv, kernel, stride=(1, 1), padding='valid', pooling='max', pool_size=(2, 2)) for kernel, conv in zip(kernels, conv1)]

    # Flatten the output of the second convolutional layer
    flattened = np.concatenate(conv2).flatten()

    return flattened

num_train_samples = 1000  # Set the desired number of training samples
num_test_samples = 500    # Set the desired number of testing samples

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
train_images = np.array([conv_net(img) for img in train_images])
test_images = np.array([conv_net(img) for img in test_images])


# Define the activation function (ReLU) and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Define the activation function (sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
w1 = np.random.randn(input_size, hidden_layer_1) * np.sqrt(2 / input_size)
b1 = np.zeros((1, hidden_layer_1))

w2 = np.random.randn(hidden_layer_1, hidden_layer_2) * np.sqrt(2 / hidden_layer_1)
b2 = np.zeros((1, hidden_layer_2))

w3 = np.random.randn(hidden_layer_2, output_size) * np.sqrt(2 / hidden_layer_2)
b3 = np.zeros((1, output_size))
'''
# Initialize the weights and biases randomly when using sigmoid
w1 = np.random.randn(input_size, hidden_size1)
b1 = np.random.randn(hidden_size1)
w2 = np.random.randn(hidden_size1, hidden_size2)
b2 = np.random.randn(hidden_size2)
w3 = np.random.randn(hidden_size2, output_size)
b3 = np.random.randn(output_size)
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
    z1 = np.dot(train_images, w1) + b1
    a1 = relu(z1)
    #a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = relu(z2)
    #a2 = sigmoid(z2)
    z3 = np.dot(a2, w3) + b3
    output = np.exp(z3) / np.sum(np.exp(z3), axis=1, keepdims=True)

    # Compute the loss and accuracy
    loss = -np.sum(np.log(output[range(len(train_labels)), train_labels])) / len(train_labels)
    predicted_labels = np.argmax(output, axis=1)
    accuracy = np.mean(predicted_labels == train_labels)

    # Backward pass
    dz3 = output
    dz3[range(len(train_labels)), train_labels] -= 1
    dw3 = np.dot(a2.T, dz3) / len(train_labels)
    db3 = np.sum(dz3, axis=0) / len(train_labels)
    da2 = np.dot(dz3, w3.T)
    dz2 = da2 * relu_derivative(z2)
    #dz2 = da2 * sigmoid_derivative(z2)
    dw2 = np.dot(a1.T, dz2) / len(train_labels)
    db2 = np.sum(dz2, axis=0) / len(train_labels)
    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * relu_derivative(z1)
    #dz1 = da1 * sigmoid_derivative(z1)
    dw1 = np.dot(train_images.T, dz1) / len(train_labels)
    db1 = np.sum(dz1, axis=0) / len(train_labels)

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
        print("Epoch {}/{} - loss: {:.4f} - accuracy: {:.4f}".format(epoch+1, num_epochs, loss, accuracy))

# Evaluate the network on the test set
z1 = np.dot(test_images, w1) + b1
a1 = relu(z1)
#a1 = sigmoid(z1)
z2 = np.dot(a1, w2) + b2
a2 = relu(z2)
#a2 = sigmoid(z2)
z3 = np.dot(a2, w3) + b3
output = np.exp(z3) / np.sum(np.exp(z3), axis=1, keepdims=True)
predicted_labels = np.argmax(output, axis=1)
accuracy = np.mean(predicted_labels == test_labels)
print("Test accuracy: {:.4f}".format(accuracy))


# Evaluate the network on the test set
num_correct = 0
for i in range(len(test_images)):
    # Forward pass
    z1 = np.dot(test_images[i], w1) + b1
    a1 = relu(z1)
    #a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = relu(z2)
    #a2 = sigmoid(z2)
    z3 = np.dot(a2, w3) + b3
    output = np.exp(z3) / np.sum(np.exp(z3))

    # Compute the predicted label and check if it's correct
    predicted_label = np.argmax(output)
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
