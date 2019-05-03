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
# obj_func = 'rosenbrock-2d' # all completed
# obj_func = 'egg-2d'        # all completed except LCCDtanh, LCCDrelu, MCCONCrelu
# obj_func = 'branin-2d'       # all completed except LCCDtanh
# obj_func = 'hartmann-6d'      # all completed except LCCDtanh, LCCDrelu,MCCONCrelu
obj_func = 'ackley-10d'         # all completed except LCCDtanh, LCCDrelu,MCCONCrelu
# obj_func = 'michalewicz-10d'      # all imcomplete
numEpoch = 1000
batch_size = 1
bo_method = 'LCB'
# models_all = ['DNGO','MCDROP','BOHAM','LCBNNse_y','LCBNNse_yclip']
# models_all = ['LCBNNtanhse_y','MCDROPtanh','LCCDtanh','MCCONCtanh']
models_all = ['LCBNNtanhse_y','MCDROPtanh','LCCDtanh','MCCONCtanh','LCBNNreluse_y','MCDROPrelu']
# models_all = ['LCBNNtanhse_y','MCDROPtanh','LCCDtanh','MCCONCtanh','LCBNNreluse_y','MCDROPrelu','LCCDrelu','MCCONCrelu']

# models_all = ['DNGO','MCDROPtanh','BOHAM','GP','LCBNNtanhse_y']
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

