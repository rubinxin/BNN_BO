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
obj_func = 'ackley-2d' # all completed except DNGO

f, bounds, _, true_fmin = get_function(obj_func)
d = bounds.shape[0]

x1, x2 = np.mgrid[-1:1:50j, -1:1:50j]
grid = np.vstack((x1.flatten(), x2.flatten())).T
fvals = f(grid)
Z =  fvals.reshape(50,50)

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, Z, cmap=cm.coolwarm)
ax.view_init(32, 20)
plt.show()
# f.savefig(f'figures/{obj_func}2d.pdf', bbox_inches='tight')

