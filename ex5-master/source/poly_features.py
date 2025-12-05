import numpy as np

def poly_features(X, power=8):
    """
    POLYFEATURES Maps X (1D vector) into the p-th power
      [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
      maps each example into its polynomial features where
      X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];

    """
    # X_poly = np.zeros((X.shape[0],power))

    """% ====================== YOUR CODE HERE ======================
    Instructions: Given a vector X, return a matrix X_poly where the p-th 
            column of X contains the values of X to the p-th power.
    """


    # for i in range (power):
    #     X_poly[:, i] = X ** (i + 1)
    #
    #
    # return np.vstack(X_poly).T

    X = X.flatten()  # ensure 1D vector
    m = X.shape[0]

    X_poly = np.zeros((m, power))

    for i in range(power):  # 0..power-1  (Correct)
        X_poly[:, i] = X ** (i + 1)  # X^1, X^2, ..., X^power

    return X_poly



