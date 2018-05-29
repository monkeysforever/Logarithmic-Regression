# -*- coding: utf-8 -*-
"""
This program performs logistic regression on a dataset of images of cats and dogs and outputs a classifier which can classify an image as a dog or cat.
The code can also be reused for other datasets with some minor changes

@author: Randeep
"""

import numpy as np
import os
import re
import imageio
from skimage.transform import resize

def sigmoid(x):
    #The function returns values ranging from 0 to 1, which helps to quantify the probability of the image being of a cat
    #The argument x is expected to be a  numpy array
    return (1/(1+np.exp(-x)))

def init_parameters(dimension):
    #The function initializes the weights and bias as 0. For logistic regression initialization to 0 is good enough.
    #The argument dimension refers to the size of input features and in our case will be the values of the pixels of a image
    w = np.zeros([dimension, 1], dtype = np.float64)
    b = 0
    return w, b

def forward_propagation(X, Y, w, b):
    #The function computes the hypothesis Z: x1w1 + x2w2 ....xnwn + b and applies the sigmoid function and finally calculates the cost
    #m refers to the number of training examples
    #Due to vectorization our code is clean and without any for loops. Vectorization not only helps readability as well as is more computationally efficient.
    #Our computational graph looks like: X->Z->A->Cost
    m = X.shape[1]
    # Z.shape is (1, m)
    Z = np.matmul(w.T, X) + b
    # A.shape is Z.shape
    A = sigmoid(Z)
    #A here refers to our predicted probability and Cost is the total logarithmic loss over all training examples
    #Cost is a scalar
    Cost = -(np.matmul(Y, np.log(A).T) + np.matmul((1 - Y), np.log(1 - A).T))/m 
    
    return Cost, A

def back_propagation(X, Y, A):
    #The function moves in the backward direction of our computation graphs and calculates the gradient of Cost, i.e, dw and db
    #With dw and db we change the weights and bias to reduce our cost 
    m = X.shape[1]
    #dw.shape is (n, 1) where n is input feature size
    #db is a scalar
    dw = (np.matmul(X, (A - Y).T))/m
    db = np.sum((A - Y), axis = 1)/m
    return dw, db


def predict(X, w, b):
    #The functions takes in an input with our adjusted weights and bias to produce a prediction ranging from 0 to 1
    #If the prediction exceeds 0.5 the value is rounded to 1 and depicts a cat picture
    m = X.shape[1]
    Y_predictions = np.zeros((1, m), dtype = np.int8)
    Z = np.matmul(w.T, X) + b
    A = sigmoid(Z)
    #Y_prediction.shape is (1, m) where m is the number of examples in X
    Y_predictions = (A > .5).astype(int)
    return Y_predictions

def gradient_descent(X, Y, w, b, num_iterations, learning_rate, print_iteration):
    #The function adjusts the values of weights and bias so as to reduce the Cost
    # w, b returned are the final weights and bias after training and can be used to make predictions
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

def logistic_regression(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_iteration = 0):
    #The functions calls the above helper functions to create a classifier from the given dataset    
    n = X_train.shape[0]
    #Initialize weights and bias
    w, b =  init_parameters(n)
    #Compute best weights and bias
    w, b, Costs = gradient_descent(X_train, Y_train, w, b, num_iterations, learning_rate, print_iteration)
    #Make training example predictions
    Y_train_predictions = predict(X_train, w, b)
    #Make test example predictions
    Y_test_predictions = predict(X_test, w, b)
    #Calculate training set accuracy    
    Y_train_accuracy = 100 - np.mean(np.abs(Y_train_predictions - Y_train))*100
    #Calculate test set accuracy
    Y_test_accuracy = 100 - np.mean(np.abs(Y_test_predictions - Y_test))*100
    return Y_train_accuracy, Y_test_accuracy, w, b, Costs, Y_train_predictions, Y_test_predictions

def load_datasets(filepath, dimension, file_count):
    #The functions loads the images of our datasets as numpy arrays
    #For other data sets this function will have to be tweaked
    #Get the file names of the images in folder with filepath and randomly choose the number of files specified by file_count
    #file_count limits the number of files to be loaded
    dirs = np.random.choice(os.listdir(filepath), file_count)
    #Split the data into training set and test set, we use the ration 3:1
    split = file_count*3//4
    train_files = dirs[:split]
    test_files = dirs[split:file_count]
    #dimensions of X_train are (training examples, height, width, number of channels), for rgb images number of channels is 3
    X_train = np.zeros((split, dimension, dimension, 3))
    Y_train = np.zeros((1, split))
    X_test = np.zeros((file_count - split, dimension, dimension, 3))
    Y_test = np.zeros((1, file_count - split))
    for i, file in enumerate(train_files):        
        filename = filepath + '/' + file
        x = imageio.imread(filename)
        #resize all images to the same dimensions        
        x = resize(x, [dimension,dimension,3])
        X_train[i] = x
        #The images with cat in their names are cat pictures so we create an output vector where 1 refers to cat pictures and 0 refers to dog pictures
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
    #Our program requires the shape of input matrix to be (height*width*numberofchannels, m) where m is number of training examples
    #The input matrices we get from load_datasets are of shape (m, height, width, number of channels)
    #The function reshapes the input matrices
    return X.reshape((X.shape[0], -1)).T

def normalize_dataset(X):
    #Diving the input matrix by largest possible value of a pixel makes converging towards lowest cost faster
    #For other datasets look at numpy.linalg.norm
    return X/255


#Steps to create classifier
    
#1.Load the dataset, remember to put in right filepath and not load too many images    
#X_train, Y_train, X_test, Y_test = load_datasets('F:/train', 125, 5000)

#2.Flatten training set and test set inputs
#X_train = flatten_dataset(X_train)
#X_test = flatten_dataset(X_test)

#3.Train your model
#train_accuracy, test_accuracy, w, b, Costs, Y_train_pre, Y_test_pre = logistic_regression(X_train, Y_train, X_test, Y_test, 1000, 0.001, 100)

#Use predict functions with weigths and bias after training to make predictions
#Train and test accuracy can be improved by changing the values of hyperparameters learning_rate, num_iterations, file_count