"""  initialConditions: this function simulates the stroke
on a drum membrane (surface) through a Gaussian 2D distribution

"""

import numpy as np


def gaussianDistribution(gridPoints, standardDeviation, amplitude, x_strike, y_strike, leftBoundary, rightBoundary):
    x = np.linspace(leftBoundary, rightBoundary, gridPoints)
    y = np.linspace(leftBoundary, rightBoundary, gridPoints)
    X, Y = np.meshgrid(x, y)

    #  set position of the strike in the y, y axis
    #  0 < x,y_strike < grid boundaries
    #x_strike = x[(math.floor(x.size / 3))]
    #y_strike = y[(math.floor(y.size / 3))]

    #standardDeviation = -5
    #  the standard deviation will influence how wide
    #  the gauss distribution will be
    #  the boundary conditions of the border never being displaced
    #  vertically (z-axis) must always be met
    z = amplitude*np.exp(standardDeviation*(np.power(X-x_strike,2)+np.power(Y-y_strike,2)))
    #  the function np.power performs an element-wise to the power of something
    #  the function np.exp computes the exponential value also element-wise
    #  for all the elements of a matrix or an array

    return z
