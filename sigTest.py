import numpy as np
import pickle
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

# Paste your sigmoid function here

    
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    X= (1/(1 + np.exp(-z)))
    return  X

# Paste your nnObjFunction here
def nnObjFunction(params, *args):
    
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    

    w1 = params[0:(n_hidden) * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[((n_hidden) * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
     

    #bias to x
    bias = np.repeat(np.array([[1]]), training_data.shape[0], 0)
    
    x = np.append(training_data, bias, 1)
    
    #Flipping w1 for matrix mult
    w1 = np.transpose(w1)
    z = sigmoid(np.dot(x, w1))
    print("XwBias")
    print(x)
    print("W1")
    print(w1)
    print("Z")
    print(z)
    
    #bias to z
    bias = np.repeat(np.array([[1]]), z.shape[0], 0)
    z = np.append(z, bias, 1)
    
    #flipping w2 for matrix 
    w2 = np.transpose(w2)  
    oh = sigmoid(np.dot(z, w2))
    print("ZwBias")
    print(z)
    print("W2")
    print(w2)
    print("oh")
    print(oh)

    #placeholder avoiding error out
    obj_grad = 0
    obj_val =0
              
    return (obj_val,obj_grad)
    


n_input = 5
n_hidden = 3
n_class = 2
training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
training_label = np.array([0,1])
lambdaval = 0
params = np.linspace(-5,5, num=26)
args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)
objval,objgrad = nnObjFunction(params, *args)
print("Objective value:")
print(objval)
print("Gradient values: ")
print(objgrad)
