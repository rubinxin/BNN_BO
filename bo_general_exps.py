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
Run Bayesian Optimisation Experiments 
'''
def BNN_BO_Exps(obj_func, model_type, bo_method, batch_option, batch_size,
                num_iter=40, seed_size=20, util_type='se_y', activation='tanh'):

    #  Specify the objective function and parameters (noise variance, input dimension, initial observation
    f, x_bounds, _, true_fmin = get_function(obj_func)
    var_noise = 1.0e-10
    d = x_bounds.shape[0]
    n_init = d*10

    saving_path = 'data/' + obj_func
    # saving_path = '/data/engs-bayesian-machine-learning/sedm4615/BNN_BO_data/'  + obj_func

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    if model_type == 'LCBNN' or model_type == 'LCCD':
        results_file_name = saving_path + '/' + model_type + activation + util_type + bo_method + str(batch_size)
    else:
        results_file_name = saving_path + '/' + model_type + activation + bo_method + str(batch_size)

    if os.path.exists(results_file_name):
        with open(results_file_name, 'rb') as exist_data_filefile2:
            existing_results = pickle.load(exist_data_filefile2)

        X_opt_all_seeds = existing_results['X_opt']
        Y_opt_all_seeds = existing_results['Y_opt']
        X_query_all_seeds = existing_results['X_query']
        Y_query_all_seeds = existing_results['Y_query']
        time_all_seeds = existing_results['runtime']

        if isinstance(X_query_all_seeds, list):
            X_query_all_seeds = X_query_all_seeds
            Y_query_all_seeds = Y_query_all_seeds
            time_all_seeds    = time_all_seeds
        else:
            X_query_all_seeds = list(X_query_all_seeds)
            Y_query_all_seeds = list(Y_query_all_seeds)
            time_all_seeds = list(time_all_seeds)

        s_start = len(Y_opt_all_seeds)
        print(f"Using existing data from seed{s_start} onwards")


    else:
        s_start = 0
        X_opt_all_seeds = []
        Y_opt_all_seeds = []
        X_query_all_seeds = []
        Y_query_all_seeds = []
        time_all_seeds = []

    for j in range(s_start, seed_size):
        # specify the random seed and generate observation data
        seed = j
        np.random.seed(seed)
        x_init, y_init = get_init_data(obj_func=f, noise_var=var_noise, n_init =n_init, bounds=x_bounds)

        # run Bayesian optimisation:
        bayes_opt = Bayes_opt(func=f, bounds = x_bounds, noise_var=var_noise)
        # model_type: GP or MCDROP or DNGO or BOHAM
        bayes_opt.initialise(X_init=x_init, Y_init=y_init, model_type=model_type, bo_method=bo_method,
                             batch_option=batch_option, batch_size=batch_size, seed=seed,  util_type=util_type,
                             actv_func = activation)

        # output of Bayesian optimisation:
        X_query,Y_query,X_opt,Y_opt, time_record = bayes_opt.iteration_step(iterations=num_iter)
        # X_query, Y_query - query points selected by BO;
        # X_opt, Yopt      - guesses of the global optimum/optimiser (= optimum point of GP posterior mean)

        # store data
        X_opt_all_seeds.append(X_opt)
        Y_opt_all_seeds.append(Y_opt)
        X_query_all_seeds.append(X_query)
        Y_query_all_seeds.append(Y_query)
        time_all_seeds.append(time_record)

        results = {'X_opt': X_opt_all_seeds,
                   'Y_opt': Y_opt_all_seeds,
                   'X_query': X_query_all_seeds,
                   'Y_query': Y_query_all_seeds,
                   'runtime': time_all_seeds}

        with open(results_file_name, 'wb') as file:
            pickle.dump(results, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
    parser.add_argument('-f', '--func', help='Objective function',
                        default='egg-2d', type=str)
    parser.add_argument('-m', '--model', help='Surrogate model: GP or MCDROP or MCCONC or DNGO or BOHAM or LCBNN',
                        default='MCDROP', type=str)
    parser.add_argument('-acq', '--acq_func', help='Acquisition function: LCB, EI, MES',
                        default='LCB', type=str)
    parser.add_argument('-bm', '--batch_opt', help='Batch option: CL, KB',
                        default='CL', type=str)
    parser.add_argument('-b', '--batch_size', help='BO Batch size. Default = 1',
                        default=1, type=int)
    parser.add_argument('-nitr', '--max_itr', help='Max BO iterations. Default = 40',
                        default=10, type=int)
    parser.add_argument('-s', '--nseeds', help='Number of random initialisation. Default = 20',
                        default=10, type=int)
    parser.add_argument('-uo', '--util_opt', help='Utility function type for Loss Calibration: se_yclip, se_y ,Default=se_y',
                        default='se_prod_y', type=str)
    parser.add_argument('-a', '--actv', help='Activation function type: tanh, relu, Default=tanh',
                        default='tanh', type=str)

    args = parser.parse_args()
    print(f"Got arguments: \n{args}")
    obj_func = args.func
    model = args.model
    acq_func = args.acq_func
    batch_opt = args.batch_opt
    batch_n = args.batch_size
    n_itrs = args.max_itr
    n_seeds = args.nseeds
    util_opt = args.util_opt
    actv = args.actv

    BNN_BO_Exps(obj_func=obj_func, model_type=model, bo_method=acq_func, batch_option=batch_opt, batch_size=batch_n,
                num_iter=n_itrs, seed_size=n_seeds, util_type = util_opt, activation=actv)
