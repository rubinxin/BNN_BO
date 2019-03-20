"""
Created on 24 Sept 2018

@author: robin
"""

import numpy as np
import math

'''Benchmark Test Function'''

def egg(x):
    '''2D eggholder
    f_min = -9.596407
    x_min = [1.0, 0.7895]'''
    x0 = x[:, 0] * 512
    x1 = x[:, 1] * 512
    term1 = -(x1 + 47) * np.sin(np.sqrt(np.abs(x1 + x0 / 2 + 47)))
    term2 = -x0 * np.sin(np.sqrt(np.abs(x0 - (x1 + 47))))
    y = (term1 + term2) / 100
    return y[:, None]

def hartmann(x):
    '''6D hartmann
    f_min = -18.22368011
    x_min = [0.20169,0.150011,0.476874,0.275332,0.311652,0.6573]
    '''
    a = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                  [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                  [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                  [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
    c = np.array([1.0, 1.2, 3.0, 3.2])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                  [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                  [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                  [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    y = 0.0
    for i in range(4):
        sum = 0.0
        for j in range(6):
            sum = sum - a[i][j]*(x[:,j]-p[i][j])**2
        y = y - c[i]*np.exp(sum)
    y_biased = 10*(y+1.5)
    return y_biased[:, None]

def branin(x):
    x = np.atleast_2d(x)
    a = x[:, 0] * 15 - 5
    b = x[:, 1] * 15
    y_unscaled = (b - (5.1 / (4 * np.pi ** 2)) * a ** 2 + 5 * a / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(
        a) + 10
    y = (y_unscaled / 10) - 15
    return y[:, None]