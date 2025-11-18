import numpy as np


def computeCost(X, y, theta):
    # # Initialize some useful values
    # m = len(y)  # number of training examples
    # J = 0  # Initialize cost variable
    #
    # # Loop through each training example
    # for i in range(m):
    #     # Compute the prediction for the current example
    #     y_pred = theta[0] * X[i, 0] + theta[1] * X[i, 1]  # Assuming theta has two parameters (intercept and slope)
    #
    #     # Add the squared error for this training example to J
    #     J += (y_pred - y[i]) ** 2
    #
    # # After the loop, divide by 2*m to compute the final cost
    # J /= (2 * m)

    m = y.size
    J = np.sum ((X.dot(theta)-y)**2) / (2*m)

    return J