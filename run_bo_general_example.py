#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""

import numpy as np
from BayesOpt_General import Bayes_opt
from exps_tasks.math_functions import get_function
from utilities.utilities import get_init_data
import os
import argparse
import pickle
'''
Test
'''
seed_size = 3
model_type = 'MCDROP'
batch_size = 1
bo_method = 'LCB'
obj_func = 'rosenbrock-2d'
#  Specify the objective function and parameters (noise variance, input dimension, initial observation
f, x_bounds, _, true_fmin = get_function(obj_func)


# obj_func = egg
var_noise = 1.0e-10
d = x_bounds.shape[0]
n_init = 3

X_opt_all_seeds = []
Y_opt_all_seeds = []
X_query_all_seeds = []
Y_query_all_seeds = []
for j in range(seed_size):
    # specify the random seed and generate observation data
    seed = j
    np.random.seed(seed)
    x_init, y_init = get_init_data(obj_func=f, noise_var=var_noise, n_init =n_init, bounds=x_bounds)

    # run Bayesian optimisation:
    bayes_opt = Bayes_opt(func=f, bounds = x_bounds, noise_var=var_noise)
    # model_type: GP or MCDROP or MCCONC or DNGO or BOHAM
    bayes_opt.initialise(X_init=x_init, Y_init=y_init, model_type=model_type, batch_size=1, bo_method=bo_method)

    # output of Bayesian optimisation:
    X_query,Y_query,X_opt,Y_opt, time_record = bayes_opt.iteration_step(iterations=20, seed=seed)
    # X_query, Y_query - query points selected by BO;
    # X_opt, Yopt      - guesses of the global optimum/optimiser (= optimum point of GP posterior mean)

    # store data
    X_opt_all_seeds.append(X_opt)
    Y_opt_all_seeds.append(Y_opt)
    X_query_all_seeds.append(X_query)
    Y_query_all_seeds.append(Y_query)

    saving_path = 'data/' + obj_func
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    #
    # X_opt_file_name = saving_path + 'X_opt/' + model_type + bo_method + str(batch_size)
    # Y_opt_file_name = saving_path + '/Y_opt' + model_type + bo_method + str(batch_size)
    results_file_name = saving_path + '/' + model_type + bo_method + str(batch_size)


