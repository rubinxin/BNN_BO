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
from exps_tasks.math_functions import get_function
from models.gp import GPModel
from utilities.utilities import get_init_data
import os
'''
Test
'''
func_name = 'GM-1d'
f, x_bounds, _, true_fmin = get_function(func_name)
d = x_bounds.shape[0]
n_init = 20
var_noise = 1.0e-10

def test_gp():
    #  Specify the objective function and parameters (noise variance, input dimension, initial observation
    np.random.seed(3)
    x_ob, y_ob = get_init_data(obj_func=f, noise_var=var_noise, n_init =n_init, bounds=x_bounds)


    # ------ Test grid -------- #
    if d == 2:
        x1, x2 = np.mgrid[-1:1:50j, -1:1:50j]
        X = np.vstack((x1.flatten(), x2.flatten())).T
        y = f(X)
    else:
        X = np.linspace(-1,1,100)[:,None]
        y = f(X)

    # -- GP model ----#
    # kern = GPy.kern.Matern52(2, variance=1., ARD=True)
    GP = GPModel(exact_feval=True,ARD=True)
    GP._update_model(x_ob, y_ob)
    m, s = GP.predict(X)

    # # ------ Plot figures -------- #
    if d == 2:
        figure, axes = plt.subplots(2, 1, figsize=(10, 10))
        sub1 = axes[0].contourf(x1, x2, y.reshape(50, 50))
        axes[0].plot(x_ob[:,0], x_ob[:,1],'rx')
        axes[0].set_title('objective func ')

        sub2 = axes[1].contourf(x1, x2, m.reshape(50, 50))
        axes[1].plot(x_ob[:,0], x_ob[:,1],'rx')
        pred_title=f'prediction by GP'
        axes[1].set_title(pred_title)
        plt.show()
    else:
        figure, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes.plot(x_ob,y_ob, 'rx')
        axes.plot(X, y, 'r--')
        axes.plot(X, m, 'b')
        axes.fill_between(X.flatten(), (m - s).flatten(), (m + s).flatten(), color='blue', alpha=0.30)
        axes.set_title('1D Regression')
        plt.show()

    # ------ Save figures -------- #
    saving_path = 'data/syntheticFns/'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    fig_name = func_name + f'_gp.png'
    print(fig_name)
    # figure.savefig(saving_path + fig_name)

if __name__ == '__main__':
    test_gp()
