# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:07:04 2018

@author: Randeep
"""

import numpy as np
import os
import re
import imageio
from skimage.transform import resize

def sigmoid(x):
    """
    Sigmoid activation function
    
    Used to activate linear equation of input matrix, weights and bias
    
    Parameters:
    -----------
    x : numpy array
        Input to be activated
    
    Returns:
    --------
        numpy array
        Activated input
    """
    
    return (1/(1+np.exp(-x)))

def init_parameters(dimension):
    """
    Weights and bias initializer
    
    Used to initialize weights and bias to 0
    
    Parameters:
    -----------
    dimesions : int
        Dimension of input features
    
    Returns:
    --------
    w : numpy array
        Weight vector of size (dimension, 1)
        
    b : float
        Bias scalar        
    """
    w = np.zeros([dimension, 1], dtype = np.float64)
    b = 0
    return w, b

def forward_propagation(X, Y, w, b):
    m = X.shape[1]
    Z = np.matmul(w.T, X) + b
    A = sigmoid(Z)
    Cost = -(np.matmul(Y, np.log(A).T) + np.matmul((1 - Y), np.log(1 - A).T))/m 
    
    return Cost, A

def back_propagation(X, Y, A):
    m = X.shape[1]
    dw = (np.matmul(X, (A - Y).T))/m
    db = np.sum((A - Y), axis = 1)/m
    return dw, db


def predict(X, w, b):
    m = X.shape[1]
    Y_predictions = np.zeros((1, m), dtype = np.int8)
    Z = np.matmul(w.T, X) + b
    A = sigmoid(Z)
    Y_predictions = (A > .5).astype(int)
    return Y_predictions

def gradient_descent(X, Y, w, b, num_iterations, learning_rate, print_iteration):
    Costs = []
    for i  in range(num_iterations):
        Cost, A =  forward_propagation(X, Y, w, b)
        dw, db = back_propagation(X, Y, A)
        w -= learning_rate * dw
        b -= learning_rate * db
        if i != 0 and i % print_iteration == 0:
            Costs.append(Cost)        
            print ("Cost after iteration %i: %f" %(i, Cost))
    return w, b, Costs

def single_node_neural_net(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_iteration = 0):    
    n = X_train.shape[0]
    w, b =  init_parameters(n)
    w, b, Costs = gradient_descent(X_train, Y_train, w, b, num_iterations, learning_rate, print_iteration)
    Y_train_predictions = predict(X_train, w, b)
    Y_test_predictions = predict(X_test, w, b)    
    Y_train_accuracy = 100 - np.mean(np.abs(Y_train_predictions - Y_train))*100
    Y_test_accuracy = 100 - np.mean(np.abs(Y_test_predictions - Y_test))*100
    return Y_train_accuracy, Y_test_accuracy, w, b, Costs, Y_train_predictions, Y_test_predictions

def load_datasets(filepath, dimension, batch):
    
    dirs = np.random.choice(os.listdir(filepath), batch)
    split = batch*3//4
    train_files = dirs[:split]
    test_files = dirs[split:batch]        
    X_train = np.zeros((split, dimension, dimension, 3))
    Y_train = np.zeros((1, split))
    X_test = np.zeros((batch - split, dimension, dimension, 3))
    Y_test = np.zeros((1, batch - split))
    for i, file in enumerate(train_files):        
        filename = filepath + '/' + file
        x = imageio.imread(filename)        
        x = resize(x, [dimension,dimension,3])
        X_train[i] = x
        
        if re.match('cat.*', file):
            Y_train[:, i] = 1
        else:
            Y_train[:, i] = 0
    for i, file in enumerate(test_files):        
        filename = filepath + '/' + file
        x = imageio.imread(filename)        
        x = resize(x, [dimension,dimension,3])
        X_test[i] = x
        
        if re.match('cat.*', file):
            Y_test[:, i] = 1
        else:
            Y_test[:, i] = 0
    
    return X_train, Y_train, X_test, Y_test
   
def flatten_dataset(X):
    return X.reshape((X.shape[0], -1)).T

def normalize_dataset(X):
    return X/255

X_train, Y_train, X_test, Y_test = load_datasets('F:/train', 125, 20000)


X_train = flatten_dataset(X_train)
X_test = flatten_dataset(X_test)

train_accuracy, test_accuracy, w, b, Costs, Y_train_pre, Y_test_pre = single_node_neural_net(X_train, Y_train, X_test, Y_test, 1000, 0.001, 100)

print(train_accuracy)
print(test_accuracy)
