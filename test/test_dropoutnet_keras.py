#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""
import sys
sys.path.insert(0, '../')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from exps_tasks.math_functions import get_function
from old.dropoutnn import DropoutNet
from utilities.utilities import get_init_data
import os
'''
Test
'''
# func_name = 'camelback-2d'
func_name = 'GM-1d'
f, x_bounds, _, true_fmin = get_function(func_name)
d = x_bounds.shape[0]
n_init = 5
var_noise = 1.0e-10


def test_dropoutnet():
    #  Specify the objective function and parameters (noise variance, input dimension, initial observation
    np.random.seed(3)
    x_ob, y_ob = get_init_data(obj_func=f, noise_var=var_noise, n_init =n_init, bounds=x_bounds)

    # ------ Test grid -------- #
    if d == 2:
        x1, x2 = np.mgrid[-1:1:50j, -1:1:50j]
        X = np.vstack((x1.flatten(), x2.flatten())).T
        y = f(X)
    else:
        X = np.linspace(-1, 1, 100)[:, None]
        y = f(X)


    # -- DropoutNet model ----#
    dropout = 0.05
    T = 100
    tau = 20
    bs = int(n_init/5)
    tbs = 50
    n_hidden = [50,50,50]
    # n_hidden = [100]


    DropNet = DropoutNet(n_epochs = 40, n_hidden=n_hidden, dropout=dropout, T = T, tau=tau, batch_size=bs)
    DropNet._update_model(x_ob, y_ob)
    m, s = DropNet.predict(X, test_batch_size=tbs)

    # ------ Plot figures -------- #
    if d == 2:
        figure, axes = plt.subplots(2,1, figsize=(10, 10))
        sub1 = axes[0].contourf(x1, x2, y.reshape(50, 50))
        axes[0].plot(x_ob[:,0], x_ob[:,1],'rx')
        axes[0].set_title('objective func ')

        sub2 = axes[1].contourf(x1, x2, m.reshape(50, 50))
        axes[1].plot(x_ob[:,0], x_ob[:,1],'rx')
        pred_title=f'prediction by NN:dropout={dropout},T={T},tau={tau},BS={bs},TBS={tbs}'
        axes[1].set_title(pred_title)
        plt.show()
    else:
        figure, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes.plot(X, y, 'r--')
        axes.plot(x_ob, y_ob, 'kx')
        axes.plot(X, m, 'b')
        axes.fill_between(X.flatten(), (m - s).flatten(), (m + s).flatten(), color='blue', alpha=0.10)
        axes.set_title('1D Regression')
        plt.show()

    # ------ Save figures -------- #
    saving_path = 'data/syntheticFns/'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    fig_name= func_name + f'_dropout={dropout}_T={T}_tau={tau}_BS={bs}_TBS={tbs}.png'
    print(fig_name)
    # figure.savefig(saving_path + fig_name)

if __name__ == '__main__':
    test_dropoutnet()
