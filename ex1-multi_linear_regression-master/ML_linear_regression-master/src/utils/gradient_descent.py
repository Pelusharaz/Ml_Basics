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

        theta = theta.copy()
        y_pred = X.dot(theta)

        # Gradient for theta[0]
        par_der_0 = np.sum((y_pred - y) * X[:, 0]) / m
        theta[0] = theta[0] - alpha * par_der_0

        # Gradient for theta[1]
        par_der_1 = np.sum((y_pred - y) * X[:, 1]) / m
        theta[1] = theta[1] - alpha * par_der_1

        # ===========================================================
        # Save the cost every iteration
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
