# -*- coding: utf-8 -*-
"""
This program performs logistic regression on three types of datasets which can be generated using sklearn.
@author: Randeep
"""

import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame

SPLIT_RATIO = 90

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
    Cost = np.squeeze(Cost)
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

def generate_dataset(examples_count, dataset_type = 'moons'):
    #This function generates different types of data sets and divides the dataset into training and test sets
    #The datasets supported are blobs, moons and circles
    #Shape of X for test and training is (number of features, examples)
    #Shape of Y for test and training is (1, examples)    
    if dataset_type == 'blobs':
        X, y = make_blobs(n_samples=examples_count, centers=2, n_features=2)        
    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=examples_count, noise=0.1)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=examples_count, noise=0.05)
        
    X = X.T
    y = np.reshape(y, (1, y.shape[0]))   
    split_index = X.shape[1]*SPLIT_RATIO//100    
    indices = np.random.permutation(X.shape[1])
    training_idx, test_idx = indices[:split_index], indices[split_index:]    
    X_training, X_test, Y_training, Y_test = X[:, training_idx], X[:, test_idx], y[:, training_idx], y[:, test_idx]    
    return X_training, X_test, Y_training, Y_test
    

def plot_dataset(X, Y):
    #This function plots the dataset        
    df = DataFrame(dict(x=X[0,:], y=X[1,:], label=Y[0, :]))
    colors = {0:'red', 1:'blue'}
    fig, ax = pyplot.subplots()    
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()
    
def plot_cost(Costs):
    pyplot.plot(Costs)
    pyplot.show()

#Steps to create classifier
    
#1.Create the dataset of your choosing
#X_training, X_test, Y_training, Y_test = generate_dataset(5000, 'moons')

#2.Train your model
#train_accuracy, test_accuracy, w, b, Costs, Y_train_pre, Y_test_pre = logistic_regression(X_training, Y_training, X_test, Y_test, 5000, .005, 100)


#4.plot your training data and predicted data to see ur predictions
#plot_dataset(X_training, Y_training)
#plot_dataset(X_training, Y_train_pre)

#Use predict functions with weigths and bias after training to make predictions
#Train and test accuracy can be improved by changing the values of hyperparameters learning_rate, num_iterations, examples_count
#Try to plot the cost values with number of iterations to get an intuition about the reducing cost using plot_cost function