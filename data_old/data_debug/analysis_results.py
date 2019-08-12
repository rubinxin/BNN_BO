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
obj_func = 'egg-2dtanh'        # all completed except DNGO, MCCONCrelu
# obj_func = 'branin-2d'       # all completed except DNGO, LCCDtanh
# obj_func = 'hartmann-6d'      # all completed DNGO
# obj_func = 'ackley-10d'         # all completed except LCBNNtanhse_yseeds, DNGO
# obj_func = 'michalewicz-10d'      # all imcomplete exc
numEpoch = 1000
batch_size = 1
bo_method = 'LCB'
models_all = ['LCBNNtanhse_yclip','LCBNNtanhse_prod_y','LCBNNtanhse_y']
#
indx = range(60)

f = plt.figure(figsize=(4,4))
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

for model_type in models_all:

    results_file_name = obj_func + '/' + model_type + bo_method + str(batch_size)

    with open(results_file_name, 'rb') as file:
        results = pickle.load(file)

    Y_opt_all_seeds = results['Y_opt']
    Time_all_seeds = results['runtime']
    print(model_type + f'seeds={len(Y_opt_all_seeds)}')

    if len(Y_opt_all_seeds) == 1:
        mean_bestVals = Y_opt_all_seeds[0]
        err_bestVals = np.zeros(len(Y_opt_all_seeds[0]))
    else:
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

