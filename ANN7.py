# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 20:11:53 2023

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

nr_letters = 10
nr_training= 100  
nr_tests = 10     
correct = np.zeros((nr_tests))
y_train = np.zeros((nr_training, nr_letters))
train_labels = [0,1,2,3,4,5,6,7,8,9]*10 
test_labels = [0,1,2,3,4,5,6,7,8,9]       
y_train = np.zeros((nr_training, nr_letters))
y_test = np.zeros((nr_tests, nr_letters))
X_train = np.zeros((nr_training, 784)) 

for i in range(nr_training):
    filename = "mnist_jpg/" + str(i) + ".jpg"
    img = Image.open(filename).convert('L')
    img.load()
    X = np.asarray(img)
    X = X.flatten()
    X = np.array(X)
    X = normalise(X)    
    X_train[i] = X 
    y = train_labels[i] 
    y_train[i, y] = 1   

X_test = np.zeros((nr_tests,784))    
for i in range(nr_tests):
    filename = "mnist_jpg/" + str(i) + ".jpg"
    img = Image.open(filename).convert('L')
    img.load()
    y = test_labels[i]
    y_test[i, y] = 1    
    X = np.asarray(img)
    X = X.flatten()
    X = np.array(X)
    X = normalise(X)    
    X_test[i] = X
    pyplot.imshow(img)
    plt.show()

input=X_train
target=y_train

print('input shape: ', input.shape)

b1=0.35
b2=0.6
Etot=.1
size_data=784
size_hidden=100
input_w=(np.random.randn(size_data,size_hidden))-0.5


def forward_pass(weight1,input,b1):
    Z = np.matmul(input, weight1) + b1
    return Z, sigmoid(Z)

def sigmoid(z):  
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1.0 - sigmoid(z)) 

def backward_init(target,out,hout,hidden_w):
    dEtot_dout_o=-(target-out)
    dout_o_dnet=sigmoid_derivative(out)
    dnet_dw=hout
                                                           
    delta0=dEtot_dout_o*dout_o_dnet
    diff=np.matmul(dnet_dw.T,delta0)
    hidden_w=hidden_w-0.5*diff #uppdaterar vikten för noden
    return hidden_w

for i in range(0,100):
    Etot_old=Etot
    for j in range(0,10):
        
        [net1b,out]=forward_pass(input_w,np.array([input[j]]),b2)
        Etot=np.sum(0.5*np.power((target[j]-out),2))
        input_wo=input_w
        input_w=backward_init(target[j],out,np.array([input[j]]),input_wo)

    print(Etot)
    plt.plot([i-1,i],[Etot_old, Etot], 'b')

[net1b,out1]=forward_pass(input_w,X_test[0],b2)
[net1b,out2]=forward_pass(input_w,X_test[1],b2)
[net1b,out3]=forward_pass(input_w,X_test[2],b2)
[net1b,out4]=forward_pass(input_w,X_test[3],b2)
[net1b,out5]=forward_pass(input_w,X_test[4],b2)
[net1b,out6]=forward_pass(input_w,X_test[5],b2)
[net1b,out7]=forward_pass(input_w,X_test[6],b2)
[net1b,out8]=forward_pass(input_w,X_test[7],b2)
[net1b,out9]=forward_pass(input_w,X_test[8],b2)
[net1b,out10]=forward_pass(input_w,X_test[9],b2)
confusionmatrix=np.array([out1,out2,out3,out4,out5,out6,out7,out8,out9,out10])

print(np.round(confusionmatrix,2))


