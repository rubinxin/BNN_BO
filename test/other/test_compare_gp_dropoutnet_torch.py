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
from models.mcdrop import MCDROPWarp
from models.dngo import DNGOWrap
from models.bohamiann import BOHAMIANNWarp
from utilities.utilities import get_init_data
import os
'''
Test
'''
func_name = 'sinc-1d'
# func_name = 'camelback-2d'
f, x_bounds, _, true_fmin = get_function(func_name)
d = x_bounds.shape[0]
n_init = 20
var_noise = 1.0e-10

def test_compare_gp_with_dropnet():
    #  Specify the objective function and parameters (noise variance, input dimension, initial observation
    np.random.seed(3)
    x_ob, y_ob = get_init_data(obj_func=f, noise_var=var_noise, n_init =n_init, bounds=x_bounds)
    rng = np.random.RandomState(42)
    x = rng.rand(n_init)

    # ------ Test grid -------- #
    if d == 2:
        x1, x2 = np.mgrid[-1:1:50j, -1:1:50j]
        X = np.vstack((x1.flatten(), x2.flatten())).T
        y = f(X)
        # Dropout NN Configuration
        dropout = 0.05
        T = 100
        tau = 1.0
        bs = 10
        tbs = 50
        n_hidden = [50, 50, 50]

    else:
        X = np.linspace(-1,1,100)[:,None]
        y = f(X)

        # Dropout NN Configuration
        dropout = 0.05
        T = 100
        tau = 20
        bs = 10
        tbs = 50
        n_hidden = [50, 50, 50]
        # n_hidden = [100]

    # -- GP model ----#
    GP = GPModel(exact_feval=True,ARD=True)
    GP._update_model(x_ob, y_ob)
    m_gp, s_gp = GP.predict(X)

    # -- MCDropoutNet or DNGO model ----#
    Net = DNGOWrap()
    # Net = MCDROPWarp()
    # Net = BOHAMIANNWarp(num_samples=600)
    Net._update_model(x_ob, y_ob)
    m_net, s_net = Net.predict(X)

    # # ------ Plot figures -------- #
    if d == 2:
        figure, axes = plt.subplots(3, 1, figsize=(6, 18))
        sub1 = axes[0].contourf(x1, x2, y.reshape(50, 50))
        axes[0].plot(x_ob[:,0], x_ob[:,1],'rx')
        axes[0].set_title('objective func ')

        sub2 = axes[1].contourf(x1, x2, m_gp.reshape(50, 50))
        axes[1].plot(x_ob[:,0], x_ob[:,1],'rx')
        gp_title=f'prediction by GP'
        axes[1].set_title(gp_title)

        sub2 = axes[2].contourf(x1, x2, m_net.reshape(50, 50))
        axes[2].plot(x_ob[:,0], x_ob[:,1],'rx')
        dropnet_title=f'prediction by NN:dropout={dropout},T={T},tau={tau},BS={bs},TBS={tbs}'
        axes[2].set_title(dropnet_title)
        plt.show()
    else:
        figure, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(x_ob,y_ob, 'ro')
        axes[0].plot(X, y, 'k--')
        axes[0].plot(X, m_gp, 'b')
        axes[0].fill_between(X.flatten(), (m_gp - s_gp).flatten(), (m_gp + s_gp).flatten(), color='orange', alpha=0.80)
        axes[0].fill_between(X.flatten(), (m_gp - 2*s_gp).flatten(), (m_gp + 2*s_gp).flatten(), color='orange', alpha=0.60)
        axes[0].fill_between(X.flatten(), (m_gp - 3*s_gp).flatten(), (m_gp + 3*s_gp).flatten(), color='orange', alpha=0.40)
        axes[0].set_title('1D GP Regression')

        axes[1].plot(x_ob,y_ob, 'ro')
        axes[1].plot(X, y, 'k--')
        axes[1].plot(X, m_net, 'b')
        axes[1].fill_between(X.flatten(), (m_net - s_net).flatten(), (m_net + s_net).flatten(), color='orange', alpha=0.80)
        axes[1].fill_between(X.flatten(), (m_net - 2*s_net).flatten(), (m_net + 2*s_net).flatten(), color='orange', alpha=0.60)
        axes[1].fill_between(X.flatten(), (m_net - 3*s_net).flatten(), (m_net + 3*s_net).flatten(), color='orange', alpha=0.40)
        dropnet_title=f'prediction by NN:dropout={dropout},T={T},tau={tau},BS={bs},TBS={tbs}'
        axes[1].set_title(dropnet_title)
        plt.show()

    # ------ Save figures -------- #
    saving_path = 'data/syntheticFns/'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    # gp_fig_name = func_name + f'_gp.png'
    # dropnet_fig_name = func_name + f'_gp.png'
    fig_name = 'compare_gp_dropnet.png'
    # print(dropnet_fig_name)
    # figure.savefig(saving_path + fig_name)

if __name__ == '__main__':
    test_compare_gp_with_dropnet()
