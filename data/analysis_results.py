#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pickle
'''
Test
'''

# obj_func = 'rosenbrock-2d'
# obj_func = 'egg-2d'
# obj_func = 'branin-2d'
# obj_func = 'hartmann-6d'
# obj_func = 'ackley-10d'
obj_func = 'michalewicz-10d'
numEpoch = 1000
batch_size = 10
bo_method = 'LCB'
# models_all = ['DNGO','MCDROP','BOHAM','LCBNNse_y','LCBNNse_yclip']
# models_all = ['MCDROP','MCDROP1K','LCBNNse_y','LCBNNse_y1K','DNGO']
models_all = ['MCDROP','MCDROP1K','LCBNNse_y','LCBNNse_y1K']

# models_all = ['LCCD','MCDROP1K','LCBNNse_y1K','DNGO']

# models_all = ['DNGO','MCDROP','BOHAM','LCBNNse_y']
#
# models_all = ['GP','MCDROP','LCBNNse_y','LCBNNse_yclip']
indx = range(30)

f = plt.figure(figsize=(4,4))
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

for model_type in models_all:

    results_file_name = obj_func + '/' + model_type + bo_method + str(batch_size)

    with open(results_file_name, 'rb') as file:
        results = pickle.load(file)

    Y_opt_all_seeds = results['Y_opt']
    print(model_type + f'seeds={len(Y_opt_all_seeds)}')
    mean_bestVals = np.mean(Y_opt_all_seeds, 0)
    err_bestVals = np.std(Y_opt_all_seeds, 0)/ len(Y_opt_all_seeds)
    plt.errorbar(indx, mean_bestVals[indx], err_bestVals[indx], label=model_type)

plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Best Value so far", fontsize=12)
plt.title(obj_func, fontsize=12)
plt.legend(prop={'size': 12},loc='lower left')
plt.show()
# plt.show()
results_file_name
f.savefig(obj_func + '/' + obj_func + bo_method + str(batch_size) + "epoch=" + str(numEpoch) + ".pdf", bbox_inches='tight')

