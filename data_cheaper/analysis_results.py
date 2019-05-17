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
Plot test functionss
'''
obj_func = 'rosenbrock-2d' # all completed except DNGO
lb,ub = 0.005, 0.02

# obj_func = 'egg-2d'        # all completed except DNGO, LCCD versions
# lb,ub = -8.5, -5.5
# obj_func = 'hartmann-6d'      # all completed DNGO
# lb,ub = -3.2, -1.6
# obj_func = 'ackley-10d'         # all completed except LCBNNtanhse_yseeds,LCBNNtanhse_prod_yseeds, DNGO
# lb,ub = 0.8, 0.99

# obj_func = 'michalewicz-10d'      # all imcomplete exc
numEpoch = 1000
batch_size = 1
bo_method = 'LCB'
activation = 'relu'
#
if activation == 'relu':
    models_all = ['MCDROPrelu','MCCONCrelu','LCBNNreluse_y','LCBNNreluse_yclip','LCBNNreluse_prod_y','LCCDreluse_prod_y','LCCDreluse_yclip','LCCDreluse_y']
else:
    models_all = ['MCDROPtanh','MCCONCtanh','LCBNNtanhse_y','LCBNNtanhse_yclip','LCBNNtanhse_prod_y','LCCDtanhse_prod_y','LCCDtanhse_yclip','LCCDtanhse_y']

# models_all = ['LCCDtanhse_y','LCCDtanhse_yclip', 'LCCDtanhse_prod_y']
# models_all = ['LCBNNreluse_y','LCBNNreluse_yclip','LCBNNreluse_prod_y','MCDROPrelu']
# models_all = [ 'LCCDreluse_y','LCCDreluse_yclip']
# models_all = ['LCBNNreluse_y','MCDROPrelu','LCCDe_y','MCCONCrelu','LCBNNreluse_prod_y']
# models_all = ['LCBNNtanhse_y','MCDROPtanh','LCCDtanh','MCCONCtanh','LCBNNreluse_y','MCDROPrelu','LCCDrelu','MCCONCrelu']

# models_all = ['DNGO','MCDROPtanh','BOHAM','GP','LCBNNtanhse_y']

indx = range(0,61,6)
fmt_list = ['P--','o--','x-','x-', 'x-', 'p-', 'p-','p-']
colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

f = plt.figure(figsize=(6,8))
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

for i, model_type in enumerate(models_all):

    try:
        results_file_name = obj_func + '/' + model_type + bo_method + str(batch_size)

        with open(results_file_name, 'rb') as file:
            results = pickle.load(file)

        Y_opt_all_seeds = results['Y_opt']
        Time_all_seeds = results['runtime']
        print(model_type + f'seeds={len(Y_opt_all_seeds)}')
        # Y_opt_all_seeds = Y_opt_all_seeds[0:4]
        mean_bestVals = np.mean(Y_opt_all_seeds, 0)
        err_bestVals = 0.3 * np.std(Y_opt_all_seeds, 0)/ np.sqrt(len(Y_opt_all_seeds))
        plt.errorbar(indx, mean_bestVals[indx], err_bestVals[indx], color= colour_cycle[i], label=model_type, fmt=fmt_list[i], markersize=10)
    except:
        print('Dont have results for this method')
        pass
plt.ylim(lb,ub)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Best Value so far", fontsize=12)
plt.title(obj_func, fontsize=12)
# plt.legend(prop={'size': 12},loc='lower left')
plt.legend(prop={'size': 12},loc='upper right')

plt.show()
# plt.show()
# f.savefig(obj_func + '/' + obj_func + bo_method + str(batch_size) + "epoch=" + str(numEpoch) + ".pdf", bbox_inches='tight')
f.savefig(f'figures/{obj_func}{bo_method}{activation}{batch_size}_epoch={numEpoch}.pdf', bbox_inches='tight')

