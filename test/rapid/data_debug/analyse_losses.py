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
Plot losses
'''
activation = 'tanh'
obj_func = 'gramcy1D'        # all completed except LCCDtanh, LCCDrelu, MCCONCrelu
max_epochs =  60
n_units = 10
regularisation = False
bo_method = 'LCB'
seed = 42
fmt_list = ['s', '^', 'o']
folder_name = f'{obj_func}_yval_L3_regu{regularisation}/{activation}_concdrop'

itr_range = 4
epoch_list_idx = -1
util = 'se_ysample_clip'
# methods = ['concdrop','lccd']
method = f'lccd_{util}'
figure, axes = plt.subplots(itr_range, 2, figsize=(8, 12), sharex='col')
colour_names = ['red','orange','gold','forestgreen','royalblue']
for k in range(0, itr_range):

    # ---- Load results for Conc Dropout or LCCD ------
    print(f'Load results for {method}')
    results_file_name = f"{folder_name}/{method}_results_s{seed}_itr{k}_n{n_units}_e{max_epochs}"
    with open(results_file_name, 'rb') as file:
        results = pickle.load(file)

    epoch_list = results['epoch_list']
    train_results = results['train_results']
    val_results = results['val_results']

    # concdrop_results:
    #   epoch list;
    #   concdrop_train_results: train mse loss,train time;
    #   concdrop_val_results: m, v, ev, av, ppp, val loss, predict time

    # lccd_results:
    #   epoch list;
    #   lccd_train_results train mse loss, train logutil, log gain for each data point, train time;
    #   lccd_val_results: m, v, ev, av, ppp, val loss, predict time

    # ---- Plot validation results ----------
    n_val_points = len(epoch_list)
    val_idx = range(n_val_points)
    val_ppp = np.zeros(n_val_points)
    val_loss = np.zeros(n_val_points)
    for iv in val_idx:
        val_ppp[iv] = val_results[iv][4]
        val_loss[iv] = val_results[iv][5]

    axes[k, 1].plot(val_idx, val_loss[val_idx], '-', label=f'valid_l', color=colour_names[-1])
    ax_ppp = axes[k, 1].twinx()  # instantiate a second axes that shares the same x-axis
    ax_ppp.plot(val_idx, - val_ppp[val_idx], '-', label=f'valid_ll', color=colour_names[1])
    ax_ppp.legend(loc='upper right')
    ax_ppp.set_ylabel('ppp')
    axes[k, 1].set_ylabel('rmse')
    axes[k, 1].set_title(f'{method}_t={k}')
    axes[k, 1].legend(loc='upper left')

    # ---- Plot training results ----------
    train_idx = range(epoch_list[epoch_list_idx])[:-2]
    train_loss = train_results[epoch_list_idx][0]

    if method.startswith('lccd'):
        log_util = train_results[epoch_list_idx][1]
        calibrated_loss = train_loss - log_util
        axes[k,0].plot(train_idx, - log_util[train_idx],'-.', label=f'nega_logU',color=colour_names[1])
        axes[k,0].plot(train_idx, calibrated_loss[train_idx],'-', label=f'calibrated_l',color=colour_names[0])

    axes[k, 0].plot(train_idx, train_loss[train_idx], '--', label=f'mse_l',color=colour_names[-1])
    axes[k,0].set_title(f'lccd_{util}_t={k}')
    axes[k,0].legend(loc='upper right')

figure.suptitle(f"{obj_func}: Train/Valid Loss and Util {util} at seed{seed} ", fontsize=16)
plt.show()
# figure.savefig(f' {obj_func}{activation}_debug_val_train_loss_plots_itr{k}_seed{seed}_e{n_epochs}.pdf', bbox_inches='tight')
print('hold')
