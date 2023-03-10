import numpy as np
import matplotlib.pyplot as plt
import mnist

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


# Define the convolutional neural network architecture
def conv_net(image):
    # Define the kernel and filter size
    kernel_size = (5, 5)
    
    # Create six different kernels
    #kernels = [np.random.randn(*kernel_size) for _ in range(6)]
    kernels = [np.clip(np.random.randn(*kernel_size),-0.5, 0.5) for _ in range(6)]
    

    # First convolutional layer with six filters
    conv1 = [conv2D(image, kernel, stride=(1, 1), padding='valid', pooling='max', pool_size=(2, 2)) for kernel in kernels]
    
    # Second convolutional layer with six filters
    conv2 = [conv2D(conv, kernel, stride=(1, 1), padding='valid', pooling='max', pool_size=(2, 2)) for kernel, conv in zip(kernels, conv1)]

    # Flatten the output of the second convolutional layer
    flattened = np.concatenate(conv2).flatten()

    return flattened


# Load the MNIST dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Flatten the input images
train_images = train_images.reshape(-1, 784)
test_images = test_images.reshape(-1, 784)

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the network architecture
input_size = 784  # 28x28
hidden_size = 128
output_size = 10

# Define the activation function (ReLU) and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Initialize the weights and biases randomly
w1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
w2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)

# Define the hyperparameters
learning_rate = 0.1
num_epochs = 50

# Initialize lists to store the loss and accuracy for each epoch
train_loss = []
train_accuracy = []

# Train the network
for epoch in range(num_epochs):
    # Forward pass
    z1 = np.dot(train_images, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    output = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)

    # Compute the loss and accuracy
    loss = -np.sum(np.log(output[range(len(train_labels)), train_labels])) / len(train_labels)
    predicted_labels = np.argmax(output, axis=1)
    accuracy = np.mean(predicted_labels == train_labels)

    # Backward pass
    dz2 = output
    dz2[range(len(train_labels)), train_labels] -= 1
    dw2 = np.dot(a1.T, dz2) / len(train_labels)
    db2 = np.sum(dz2, axis=0) / len(train_labels)
    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * relu_derivative(z1)
    dw1 = np.dot(train_images.T, dz1) / len(train_labels)
    db1 = np.sum(dz1, axis=0) / len(train_labels)

    # Update the weights and biases
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
z2 = np.dot(a1, w2) + b2
output = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)
predicted_labels = np.argmax(output, axis=1)
accuracy = np.mean(predicted_labels == test_labels)
print("Test accuracy: {:.4f}".format(accuracy))

# Evaluate the network on the test set
num_correct = 0
for i in range(len(test_images)):
    # Forward pass
    z1 = np.dot(test_images[i], w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    output = np.exp(z2) / np.sum(np.exp(z2))

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

