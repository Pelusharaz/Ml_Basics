import matplotlib.pyplot as plt
from numpy import where

# PLOTDATA Plots the data points X and y into a new figure
#   PLOTDATA(x,y) plots the data points with * for the positive examples
#   and o for the negative examples. X is assumed to be a Mx2 matrix.


def plotData(X, y):

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option 'r*' for the positive
    #               examples and 'bo' for the negative examples.
    #

    # plt.plot (X[y==1,0], X[y==1,1],'r*')
    pos = where(y == 1)
    neg = where(y == 0)

    # Plot positive examples
    plt.plot(X[pos, 0], X[pos, 1], 'r*')

    # Plot negative examples
    plt.plot(X[neg, 0], X[neg, 1], 'bo')

    pass # delete this line
    # =========================================================================
