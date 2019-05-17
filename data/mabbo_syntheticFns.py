#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:31:49 2018

@author: shivap
"""


# =============================================================================
#  EXP3 Algorithm as Multi-arm bandit problem with BO for Synthetic functions (2D)
#        Arm 0: Rosenbrock, Arm 1: Sixhump Camel Arm 2: Beale 
#  EXP3 should select Arm 1: sixhump camel as the best function 
#  The best value should converge to 1.03 
# =============================================================================
import sys
sys.path.append('../bayesopt')
sys.path.append('../ml_utils')
sys.path.append('../slice_sampling')
import numpy as np
import matplotlib.pyplot as plt
from MAB.MAB_BO import MAB_BO
from MAB.Categorical_BO import Categorical_BO
import os
import pickle

import testFunctions.syntheticFunctions


f = testFunctions.syntheticFunctions.func2C
categories = [3,5]

bounds_mabbo = [{'name': 'h1', 'type': 'discrete', 'domain': (0,1,2)},
                {'name': 'h2', 'type': 'discrete', 'domain': (0,1,2,3,4)},
                {'name': 'x1', 'type': 'continuous', 'domain': (-2,2)},
                {'name': 'x2', 'type': 'continuous', 'domain': (-2,2)}]

bounds_catbo = [
    {'name': 'h1', 'type': 'categorical', 'domain': (0,1,2)},
    {'name': 'h2', 'type': 'categorical', 'domain': (0,1,2,3,4)},
    {'name': 'x1', 'type': 'continuous', 'domain': (-2,2)},
    {'name': 'x2', 'type': 'continuous', 'domain': (-2,2)}
]


#%%


trials = 1    # no of times to repeat the experiment
budget = 200     # budget for bayesian optimisation
seed   = 42     # seed for random number generator
acq_type = 'LCB'
# initN - no of initial points to be generated for BO
# bounds - bounds of input for BO
# acq_type - List of valid acquisition functions for GPyopt

saving_path = 'data/syntheticFns/'
if not os.path.exists(saving_path):
    os.makedirs(saving_path)

# #%% Run MAB-BO Algorithm
mabbo = MAB_BO(objfn=f, initN=3, bounds=bounds_mabbo, acq_type=acq_type, C=categories, rand_seed=seed)
mabbo.runoptimBatchList(trials, budget)

# np.save(saving_path+'MABBOmean_best_vals'+acq_type, mabbo.mean_best_vals)
# np.save(saving_path+'MABBOerr_best_vals'+acq_type, mabbo.err_best_vals)
# np.save(saving_path+'MABBObest_input_vals', mabbo.best_val_list[0][4])

#%% Run GPyOpt Default BO For Categorical Inputs with the same initial data as BO_EXP3
# catBO = Categorical_BO(objfn=f, initN=3, bounds=bounds_catbo, acq_type=acq_type, C=categories, rand_seed=seed)
# catBO.runCatBO_Trials(trials, budget)
#
# np.save(saving_path+'CatBOmean_best_vals'+acq_type, catBO.mean_best_vals)
# np.save(saving_path+'CatBOerr_best_vals'+acq_type, catBO.err_best_vals)
# # np.save(saving_path+'MABBObest_input_vals', catBO.best_val_list[0][4])
#
# print(f"MABBO_best = {mabbo.mean_best_vals[-1]} vs CATBO_best = {catBO.mean_best_vals[-1]}")

# #%% Run GPyOpt Default BO For Categorical Inputs with different initial data
# catBO2 = Categorical_BO(objfn=f, initN=3, bounds=bounds_catbo, acq_type='LCB', C=categories, rand_seed=seed)
# catBO2.runCatBOdiffinit_Trials(trials, budget)
#
# np.save(saving_path+'CatBODmean_best_vals', catBO.mean_best_vals)
# np.save(saving_path+'CatBODerr_best_vals', catBO.err_best_vals)
# np.save(saving_path+'CatBODbest_vals', catBO.best_vals)

