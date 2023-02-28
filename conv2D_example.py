# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:53:00 2023

@author: MNBE
"""
import mnist
import numpy as np
from matplotlib import pyplot
from numpy import loadtxt


gradient_data = loadtxt('data.csv', delimiter=',')
train_images = mnist.train_images()[:1]


testdata=train_images[0]


def conv2D(X,w,matrix_size,w_size):
    temp=np.zeros((matrix_size-w_size+1,matrix_size-w_size+1))
    for i in range(matrix_size-w_size+1):
        for j in range(matrix_size-w_size+1):
            temp[i,j]=sum(sum(X[i:i+w_size,j:j+w_size]*w))
    return temp        
def L(target,out):
    padding=np.zeros((out.shape[0]+2,out.shape[0]+2))
    DLerror=-(target-out)
    padding[1:out.shape[0]+1,1:out.shape[0]+1]=DLerror
    return padding     
def E(target,out):
    padding=np.zeros((out.shape[0]+2,out.shape[0]+2))
    DLerror=-(target-out)
    
    return DLerror 

    
pyplot.imshow(testdata)

#testy=(testy/ 255) - 0.5
Wdelta_cnn=conv2D(testdata,gradient_data,testdata.shape[0],gradient_data.shape[0])

# check with other application that Wdelta_cnn is correct
#[[[-1.02803673 -1.036953   -1.03700237]
 # [-1.06904916 -1.07444242 -1.06178236]
  #[-1.05650158 -1.05535278 -1.03704934]]]
  
