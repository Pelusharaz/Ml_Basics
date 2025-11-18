from numpy import size, zeros, log, sum
from sigmoid import sigmoid
import numpy as np

# COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
#   COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
#   theta as the parameter for regularized logistic regression and the
#   gradient of the cost w.r.t. to the parameters.

def costFunctionReg(theta, X, y, _lambda):

    # Initialize some useful values
    m = size(y, 0)  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = zeros(size(theta, 0))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    y_pred = sigmoid(X.dot(theta))
    J = np.sum(y * -np.log(y_pred) - (1 - y) * -np.log(1 - y_pred)) / m
    reg = np.sum(theta[1:]**2)
    n=len (theta)
    J +=(_lambda/(2*n) ) * reg


    grad = X.T.dot(y_pred - y) / m

    return J, grad
    # =============================================================
