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

import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network architecture
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize the weights and biases for the hidden layer and output layer
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias2 = np.zeros((1, self.output_size))

    # Forward pass through the neural network
    def forward(self, x):
        # Compute the output of the hidden layer
        self.hidden_output = sigmoid(np.dot(x, self.weights1) + self.bias1)

        # Compute the output of the output layer
        self.output = sigmoid(np.dot(self.hidden_output, self.weights2) + self.bias2)

        return self.output

    # Backward pass through the neural network
    def backward(self, x, y, output):
        # Compute the error in the output layer
        error = y - output

        # Compute the derivative of the error with respect to the output layer weights and biases
        d_weights2 = np.dot(self.hidden_output.T, error * sigmoid_derivative(output))
        d_bias2 = np.sum(error * sigmoid_derivative(output), axis=0, keepdims=True)

        # Compute the derivative of the error with respect to the hidden layer weights and biases
        d_weights1 = np.dot(x.T, np.dot(error * sigmoid_derivative(output), self.weights2.T) * sigmoid_derivative(self.hidden_output))
        d_bias1 = np.sum(np.dot(error * sigmoid_derivative(output), self.weights2.T) * sigmoid_derivative(self.hidden_output), axis=0, keepdims=True)

        # Update the weights and biases using the gradients
        self.weights2 += d_weights2
        self.bias2 += d_bias2
        self.weights1 += d_weights1
        self.bias1 += d_bias1

    # Train the neural network using backpropagation
    def train(self, x, y, epochs):
        for epoch in range(epochs):
            # Forward pass through the neural network
            output = self.forward(x)

            # Backward pass through the neural network
            self.backward(x, y, output)

    # Predict the class label for a given input
    def predict(self, x):
        # Forward pass through the neural network
        output = self.forward(x)

        # Convert the output to a class label
        class_label = np.argmax(output)

        return class_label

# Load the MNIST dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Preprocess the data
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_images = train_images.reshape(-1, 28, 28)
test_images = test_images.reshape(-1, 28, 28)


# Flatten the output of the CNN for use as input to the ANN
train_features = np.array([conv_net(image).flatten() for image in train_images])
test_features = np.array([conv_net(image).flatten() for image in test_images])

# Define the input, hidden, and output sizes for the ANN
input_size = train_features.shape[1]
hidden_size = 32
output_size = 10

# Instantiate the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network
nn.train(train_features, train_labels, epochs=10)

# Evaluate the neural network on the test set
correct = 0
for i in range(len(test_images)):
    # Compute the predicted label for the current test image
    predicted_label = nn.predict(test_features[i])

    # Compute the true label for the current test image
    true_label = test_labels[i]

    # Print the predicted label and the true label
    print("Predicted label:", predicted_label)
    print("True label:", true_label)

    # Update the number of correctly classified test images
    if predicted_label == true_label:
        correct += 1

# Compute the accuracy on the test set
accuracy = correct / len(test_images)
print("Test set accuracy:", accuracy)
