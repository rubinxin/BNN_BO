#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from exps_tasks.sync_funcs import egg
from models.gp import GPModelML
'''
Test
'''

def test_gp():
    #  Specify the objective function and parameters (noise variance, input dimension, initial observation
    obj_func = egg
    var_noise = 1.0e-10
    d = 2
    initialsamplesize = 30

    np.random.seed(1)
    x_ob = np.random.uniform(0., 1., (initialsamplesize, d))
    y_ob = obj_func(x_ob) + np.sqrt(var_noise) * np.random.randn(initialsamplesize, 1)

    # ------ Test grid -------- #
    x1, x2 = np.mgrid[0:1:50j, 0:1:50j]
    X = np.vstack((x1.flatten(), x2.flatten())).T
    y = obj_func(X)


    # -- GP model ----#
    # kern = GPy.kern.Matern52(2, variance=1., ARD=True)
    GP = GPModelML(exact_feval=True,ARD=True)
    GP._update_model(x_ob, y_ob)
    m, s = GP.predict(X)

    # # ------ Plot figures -------- #
    figure, axes = plt.subplots(2,1, figsize=(10, 10))
    sub1 = axes[0].contourf(x1, x2, y.reshape(50, 50))
    axes[0].plot(x_ob[:,0], x_ob[:,1],'r^')
    axes[0].set_title('objective func ')

    sub2 = axes[1].contourf(x1, x2, m.reshape(50, 50))
    axes[0].plot(x_ob[:,0], x_ob[:,1],'r^')
    axes[0].set_title('objective func ')
    plt.show()

if __name__ == '__main__':
    test_gp()
