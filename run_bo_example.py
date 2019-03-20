#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""
import numpy as np
from exps_tasks.sync_funcs import egg
from BayesOpt import Bayes_opt
from exps_tasks.math_functions import get_function
from utilities.utilities import get_init_data

'''
Test
'''
seed_size = 1
#  Specify the objective function and parameters (noise variance, input dimension, initial observation
f, x_bounds, _, true_fmin = get_function('branin-2d')
# obj_func = egg
var_noise = 1.0e-10
d = x_bounds.shape[0]
n_init = 3
lb = np.zeros(d)
hb= np.ones(d)

for j in range(seed_size):
    # specify the random seed and generate observation data
    seed = j*10
    np.random.seed(seed)
    x_init, y_init = get_init_data(obj_func=f, noise_var=var_noise, n_init =n_init, bounds=x_bounds)

    # run Bayesian optimisation:
    bayes_opt = Bayes_opt(f, bounds = x_bounds, noise_var=var_noise)
    bayes_opt.initialise(x_init, y_init)

    # output of Bayesian optimisation:
    # X_query, Y_query - query points selected by BO;
    # X_opt, Yopt      - guesses of the global optimum/optimiser (= optimum point of GP posterior mean)
    X_query,Y_query,X_optimum,Y_optimum = bayes_opt.iteration_step(iterations=50, seed=seed,bo_method='MES')
