import numpy as np
from .compute_cost import compute_cost


# GRADIENTDESCENT Performs gradient descent to learn theta
#   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
#   taking num_iters gradient steps with learning rate alpha


def gradient_descent(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size  # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #
        # Hint: X.shape = (97, 2), y.shape = (97, ), theta.shape = (2, )

        theta = theta.copy() #create a copy for theta
        y_pred = X.dot(theta) # calculate the predicted value

        # Gradient for theta[0]
        pred_1 = np.sum((y_pred - y) * X[:, 0]) / m  # performing the 1st Gradient step
        theta[0] = theta[0] - alpha * pred_1

        # Gradient for theta[1]
        pred_2 = np.sum((y_pred - y) * X[:, 1]) / m # performing the 2nd Gradient step
        theta[1] = theta[1] - alpha * pred_2

        # ===========================================================
        # Save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
