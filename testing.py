# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 17:27:53 2023

@author: Oualid Burström
"""

# Import necessary libraries
import numpy as np
import mnist
import matplotlib.pyplot as plt

# Load mnist dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Set the number of training and test samples to be used
num_train_samples = 60000  # Set the desired number of training samples
num_test_samples = 10000    # Set the desired number of testing samples

# Reduce the number of training and test images
train_images = train_images[:num_train_samples]
train_labels = train_labels[:num_train_samples]
test_images = test_images[:num_test_samples]
test_labels = test_labels[:num_test_samples]

# Normalize the pixel values of the images to be between 0 and 1
train_images = train_images / 255
test_images = test_images / 255

# Define the convolution operation
def convolution(image, kernel, stride, padding=0):
    kernel_size = kernel.shape[0]
    output_size = (image.shape[0] - kernel_size + 2 * padding) // stride + 1
    output = np.zeros((output_size, output_size))
    padded_image = np.pad(image, ((padding, padding), (padding, padding)))

    for y in range(0, output_size):
        for x in range(0, output_size):
            img_region = padded_image[y * stride : y * stride + kernel_size, x * stride : x * stride + kernel_size]
            output[y, x] = np.sum(img_region * kernel)

    return output


# Define the max pooling operation
def max_pooling(image, pool_size, stride):
    # Calculate the size of the output feature map
    output_size = (image.shape[0] - pool_size) // stride + 1
    # Create an empty output feature map
    output = np.zeros((output_size, output_size))
    # Perform max pooling operation
    for y in range(0, output_size):
        for x in range(0, output_size):
            img_region = image[y * stride : y * stride + pool_size, x * stride : x * stride + pool_size]
            output[y, x] = np.max(img_region)
    return output

# Define the CNN
def cnn(image, conv_filters_1, conv_filters_2, pool_size, pool_stride):
    # Apply the first convolution and max pooling operations on the image using the specified filters
    feature_maps_1 = [convolution(image, conv_filter, 1) for conv_filter in conv_filters_1]
    pooled_maps_1 = [max_pooling(feature_map, pool_size, pool_stride) for feature_map in feature_maps_1]

    # Apply the second convolution and max pooling operations on the pooled_maps_1 using the specified filters
    feature_maps_2 = [convolution(pooled_map, conv_filter, 1) for pooled_map, conv_filter in zip(pooled_maps_1, conv_filters_2)]
    pooled_maps_2 = [max_pooling(feature_map, pool_size, pool_stride) for feature_map in feature_maps_2]

    # Flatten the resulting feature maps into a 1D vector
    output = np.concatenate([pooled_map.flatten() for pooled_map in pooled_maps_2])
    
    return output, feature_maps_1, pooled_maps_1, feature_maps_2, pooled_maps_2


# Define the ReLU activation function
def relu(x):
    return np.maximum(x, 0)

# Define the softmax activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Define the ANN
def ann(input_data, weights, biases):
    # Calculate the output of the input layer
    input_layer = np.dot(input_data, weights[0]) + biases[0]
   # Calculate the output of the hidden layer with ReLU activation function
    hidden_layer = relu(np.dot(input_layer, weights[1]) + biases[1])
    # Calculate the output of the output layer with softmax activation function
    output_layer = softmax(np.dot(hidden_layer, weights[2]) + biases[2])
    return input_layer, hidden_layer, output_layer

conv_filters_1 = [np.random.randn(5, 5) * 0.01 for _ in range(6)]
conv_filters_2 = [np.random.randn(5, 5) * 0.01 for _ in range(6)]
pool_size = 2
pool_stride = 2

input_size = 4 * 4 * 6  # (4x4) pooled feature maps * 6 filters


hidden_size = 64
output_size = 10

weights = [
    np.random.randn(input_size, hidden_size) * 0.01,
    np.random.randn(hidden_size, hidden_size) * 0.01,
    np.random.randn(hidden_size, output_size) * 0.01,
]

biases = [
    np.zeros((1, hidden_size)),
    np.zeros((1, hidden_size)),
    np.zeros((1, output_size)),
]

def convolution_backward(d_output, image, kernel, stride):
    kernel_size = kernel.shape[0]
    d_kernel = np.zeros_like(kernel)
    d_image = np.zeros_like(image)

    for y in range(d_output.shape[0]):
        for x in range(d_output.shape[1]):
            d_kernel += d_output[y, x] * image[y * stride : y * stride + kernel_size, x * stride : x * stride + kernel_size]
            d_image[y * stride : y * stride + kernel_size, x * stride : x * stride + kernel_size] += d_output[y, x] * kernel

    return d_kernel, d_image

def max_pooling_backward(d_output, image, pool_size, stride):
    d_image = np.zeros_like(image)

    for y in range(d_output.shape[0]):
        for x in range(d_output.shape[1]):
            img_region = image[y * stride : y * stride + pool_size, x * stride : x * stride + pool_size]
            mask = img_region == np.max(img_region)
            d_image[y * stride : y * stride + pool_size, x * stride : x * stride + pool_size] += d_output[y, x] * mask

    return d_image

# Train the network
learning_rate = 0.01    # Set the learning rate
epochs = 5              # Set the number of epochs

training_losses = []    # Create an empty list to store the training losses
training_accuracies = []    # Create an empty list to store the training accuracies

for epoch in range(epochs):

    epoch_losses = []
    epoch_corrects = 0
    
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        # Apply the CNN and ANN on the input image
        cnn_output, feature_maps_1, pooled_maps_1, feature_maps_2, pooled_maps_2 = cnn(image, conv_filters_1, conv_filters_2, pool_size, pool_stride)
        input_layer, hidden_layer, ann_output = ann(cnn_output.reshape(1, -1), weights, biases)

        # One-hot encode the label
        one_hot_label = np.zeros((1, output_size))
        one_hot_label[0, label] = 1

        # Calculate loss and gradients
        loss = -np.sum(one_hot_label * np.log(ann_output))
        epoch_losses.append(loss)
        pred_label = np.argmax(ann_output, axis=1)
        epoch_corrects += int(pred_label == label)        
        d_output = ann_output - one_hot_label
        d_hidden = np.dot(d_output, weights[2].T) * (hidden_layer > 0)
        d_input = np.dot(d_hidden, weights[1].T)

        # Update the weights and biases using backpropagation and the Adam optimizer
        weights[2] -= learning_rate * np.dot(hidden_layer.T, d_output)
        biases[2] -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
        weights[1] -= learning_rate * np.dot(input_layer.T, d_hidden)
        biases[1] -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)
        weights[0] -= learning_rate * np.dot(cnn_output.reshape(-1, 1), d_input)
        
        # Backpropagate through the CNN
        d_cnn_output = d_input.reshape(-1, 4, 4, 6)
        d_feature_maps_2 = [max_pooling_backward(d_cnn_output[:, :, :, i], feature_maps_2[i], pool_size, pool_stride) for i in range(6)]
        d_conv_filters_2 = [np.zeros_like(conv_filters_2[i]) for i in range(6)]
        d_pooled_maps_1 = [np.zeros_like(pooled_maps_1[i]) for i in range(6)]
        
        for i in range(6):
            d_conv_filters_2[i], d_pooled_maps_1[i] = convolution_backward(d_feature_maps_2[i], pooled_maps_1[i], conv_filters_2[i], 1)
        
        d_feature_maps_1 = [max_pooling_backward(d_pooled_maps_1[i], feature_maps_1[i], pool_size, pool_stride) for i in range(6)]
        d_conv_filters_1 = [np.zeros_like(conv_filters_1[i]) for i in range(6)]
         
        for i in range(6):
            d_conv_filters_1[i], _ = convolution_backward(d_feature_maps_1[i], image, conv_filters_1[i], 1)
         
        # Update the convolutional filters using the computed gradients
        for i in range(6):
            conv_filters_1[i] -= learning_rate * d_conv_filters_1[i]
            conv_filters_2[i] -= learning_rate * d_conv_filters_2[i]                             
        
        # Print the loss for every 1000th sample
        if i % 1000 == 0:
            print(f"Epoch: {epoch + 1}, Sample: {i}, Loss: {loss}")

    # Calculate the average loss and accuracy for this epoch
    avg_epoch_loss = np.mean(epoch_losses)
    epoch_accuracy = epoch_corrects / num_train_samples    
    # Store the values
    training_losses.append(avg_epoch_loss)
    training_accuracies.append(epoch_accuracy)
    # Print the average loss and accuracy for this epoch
    print(f"Epoch: {epoch + 1}, Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%")            

# Evaluate the networks
test_cnn_outputs = np.array([cnn(image, conv_filters_1, pool_size, pool_stride) for image in test_images])
test_ann_outputs = ann(test_cnn_outputs, weights, biases)
_, _, test_ann_outputs = ann(test_cnn_outputs, weights, biases)
accuracy = np.mean(np.argmax(test_ann_outputs, axis=1) == test_labels)
print(f"Test set accuracy: {accuracy * 100:.2f}%")

# Plot the training loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), training_losses, 'o-', markersize=5)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), training_accuracies, 'o-', markersize=5)
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy")
plt.grid()

plt.show()