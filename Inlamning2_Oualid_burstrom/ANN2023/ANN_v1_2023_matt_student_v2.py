# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:18:34 2020

@author: Magnus Bengtsson
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import pyplot

def normalise(X):
    mean = np.mean(X)
    s = np.std(X)
    if s != 0:
        return np.divide(np.subtract(X, mean), s)
    else:
        return X/255 - 0.5
#import cupy as np enbart på GPU baserade datorer Nvidia
#ANN proof of concept
#Definitions
#function ANN_v2()
input=np.array([0.05, 0.1, 0.3, 0.1, 0.3])
nr_letters = 5
nr_training= 10     # 10 träningsbokstäver som är 5x5 pixlar stora
nr_tests = 5        # 6 testbokstäver som är 5x5 pixlar stora, varav en är felaktig
correct = np.zeros((nr_tests))
y_train = np.zeros((nr_training, nr_letters))
train_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]    # 0 för A, 1 för B, 2 för C, 3 för D, 4 för E
test_labels = [0, 1, 2, 3, 4]           # 0 för A, 1 för B, 2 för C, 3 för D, 4 för E och den sista bilden är felaktig
y_train = np.zeros((nr_training, nr_letters))
y_test = np.zeros((nr_tests, nr_letters))
X_train = np.zeros((nr_training, 25)) #problemet uppstår här med att np.zeros skapar en matris, men det går inte att lägga in en matris i en 1D array se rad 122 + 129 för hantering av input
# 1t.jpg = A, 2t.jpg = A, 3t.jpg = B, 4t.jpg = B, 5t.jpg = C, 6t.jpg = C 
for i in range(nr_training):
    filename = "" + str(i+1) + "t.jpg"
    img = Image.open(filename).convert('L')
    
    img.load()
    X = np.asarray(img)
    
    X = X.flatten()
    #X=np.random.rand(25)
    X = np.array(X)
    X = normalise(X)    
    X_train[i] = X 
    y = train_labels[i]  # 2 testfall av varje bokstav
    y_train[i, y] = 1   
#    pyplot.imshow(img)
#    plt.show()

# testar bilder som saknar några pixlar
X_test = np.zeros((nr_tests,25))    
for i in range(nr_tests):
    filename = ""+str(i+1)+".jpg"
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
#size_data=25
#size_hidden=30
#size_hidden2=20

#input=np.random.rand(size_data)-0.5
input=X_train
target=y_train

#input=np.random.rand(size_data)-0.5
input=X_train                                   ### (i x j) 10 * 25 (images * pixels)
print('input shape: ', input.shape)

#ANN proof of concept
#Definitions
#function ANN_v2()
#input=np.array([[[0.05, 8,0.05, 8]],[[4.0,2.1,0.05, 8]],[[5.0,2.1,0.05, 8]]])
#target=np.array([[0.01, 0.99,0.01],[0.99, 0.01,0.01],[0.01, 0.01,0.99]])
b1=0.35
b2=0.6
Etot=.1
#input_w=np.asarray([[.15, .25, 0.3],[.2, .3, 0.1],[.15, .25, 0.3],[.2, .3, 0.1]]) 
size_data=25
size_hidden=5
input_w=(np.random.randn(size_data,size_hidden))-0.5


#hidden_w=(np.random.randn(size_hidden,6)) #-0.5

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
                                                           
    #diff=np.matmul((dEtot_dout_o*dout_o_dnet).T,dnet_dw).T
    
    delta0=dEtot_dout_o*dout_o_dnet
    diff=np.matmul(dnet_dw.T,delta0)
    #diff_b1=np.mean(dEtot_dout_o*dout_o_dnet,axis=0)
    hidden_w=hidden_w-0.5*diff #uppdaterar vikten för noden
    return hidden_w



##################main loop#########################################
for i in range(0,100):
    Etot_old=Etot
    for j in range(0,10):
        
        [net1b,out]=forward_pass(input_w,np.array([input[j]]),b2)
########%=====
    
        Etot=np.sum(0.5*np.power((target[j]-out),2))
###Etot=sum(E);
##%Backward pass
        input_wo=input_w
        input_w=backward_init(target[j],out,np.array([input[j]]),input_wo)
##%Hidden layer


    print(Etot)
    plt.plot([i-1,i],[Etot_old, Etot], 'b')
    ############################################################
    
[net1b,out1]=forward_pass(input_w,X_test[0],b2)
[net1b,out2]=forward_pass(input_w,input[1],b2)
[net1b,out3]=forward_pass(input_w,input[2],b2)
[net1b,out4]=forward_pass(input_w,input[3],b2)
[net1b,out5]=forward_pass(input_w,input[4],b2)
[net1b,out6]=forward_pass(input_w,input[5],b2)
[net1b,out7]=forward_pass(input_w,input[6],b2)
[net1b,out8]=forward_pass(input_w,input[7],b2)
[net1b,out9]=forward_pass(input_w,input[8],b2)
[net1b,out10]=forward_pass(input_w,input[9],b2)
confusionmatrix=np.array([out1,out2,out3,out4,out5,out6,out7,out8,out9,out10])

print(np.round(confusionmatrix,2))










