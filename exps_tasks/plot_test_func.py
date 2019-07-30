#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from exps_tasks.math_functions import get_function
'''
Test
'''
# obj_func = 'egg-2d' # all completed except DNGO
# obj_func = 'rosenbrock-2d' # all completed except DNGO
obj_func = 'ackley-1d' # all completed except DNGO

f1, bounds, _, true_fmin = get_function('ackley-1d')
f2, bounds, _, true_fmin = get_function('rastrigin-1d')

d = bounds.shape[0]
def f(x):
    y = f1(x) + f2(x)/40
    return y

if d == 1:
    fig = plt.figure(figsize=(6,4))
    x_grid = np.linspace(-1, 1, 100)[:, None]
    fvals = f(x_grid)
    plt.plot(x_grid, fvals, "k--")
    plt.show

elif d ==2:
    x1, x2 = np.mgrid[-1:1:50j, -1:1:50j]
    grid = np.vstack((x1.flatten(), x2.flatten())).T
    fvals = f(grid)
    Z =  fvals.reshape(50,50)
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, Z, cmap=cm.coolwarm)
    ax.view_init(32, 20)
    plt.show()
fig.savefig(f'{obj_func}.pdf', bbox_inches='tight')

