#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""
import numpy as np
from exps_tasks.sync_funcs import egg
from BayesOpt import Bayes_opt

'''
Test
'''
seed_size = 1
#  Specify the objective function and parameters (noise variance, input dimension, initial observation
obj_func = egg
var_noise = 1.0e-10
d = 2
initialsamplesize = 3

for j in range(seed_size):
    # specify the random seed and generate observation data
    seed = j*10
    np.random.seed(seed)
    x_ob = np.random.uniform(0., 1., (initialsamplesize, d))
    y_ob = obj_func(x_ob) + np.sqrt(var_noise) * np.random.randn(initialsamplesize, 1)

    # run Bayesian optimisation:
    # obj_func  - objective function to be minimised
    # lb,hb     - lower and upper bound for inputs/search space
    # var_noise - 0 if the objective function is noiseless , otherwise specify the variance for a gaussian noise
    bayes_opt = Bayes_opt(obj_func, lb=np.zeros(d), hb=np.ones(d), noise_var=var_noise)
    bayes_opt.initialise(x_ob, y_ob)

    # output of Bayesian optimisation:
    # X_query, Y_query - query points selected by BO;
    # X_opt, Yopt      - guesses of the global optimum/optimiser (= optimum point of GP posterior mean)
    X_query,Y_query,X_optimum,Y_optimum = bayes_opt.iteration_step(iterations=50, seed=seed,bo_method='LCB')
