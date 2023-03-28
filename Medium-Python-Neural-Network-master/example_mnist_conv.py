import numpy as np

from network import Network
from fc_layer import FCLayer
from conv_layer import ConvLayer
from flatten_layer import FlattenLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime
import mnist


# Load mnist dataset
x_train = mnist.train_images()
y_train = mnist.train_labels()
x_test = mnist.test_images()
y_test = mnist.test_labels()

# Set the number of training and test samples to be used
num_train_samples = 10000  # Set the desired number of training samples
num_test_samples = 2000    # Set the desired number of testing samples

# Reduce the number of training and test images
x_train = x_train[:num_train_samples]
y_train = y_train[:num_train_samples]
x_test = x_test[:num_test_samples]
y_test = y_test[:num_test_samples]

# Reshape the input data
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical format
y_train = np.eye(10)[y_train.astype('int32')]
y_test = np.eye(10)[y_test.astype('int32')]


# Network
net = Network()
net.add(ConvLayer((28, 28, 1), (3, 3), 1))  # input_shape=(28, 28, 1)   ;   output_shape=(26, 26, 1) 
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FlattenLayer())                     # input_shape=(26, 26, 1)   ;   output_shape=(1, 26*26*1)
net.add(FCLayer(26*26*1, 100))              # input_shape=(1, 26*26*1)  ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 10))                   # input_shape=(1, 100)      ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=100, learning_rate=0.1)

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])