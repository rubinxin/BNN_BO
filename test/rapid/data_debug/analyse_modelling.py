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
import os
'''
Plot modelling behaviour
'''
activation = 'tanh'
func_name = 'gramcy1D_yval'        # all completed except LCCDtanh, LCCDrelu, MCCONCrelu
regularisation = False
save_results  = False

if func_name == 'gramcy1D_yval':
    def f(x_0):
        x = 2 * x_0 + 0.5
        f = (np.sin(x * 4 * np.pi) / (2 * x) + (x - 1) ** 4) - 4
        y = 2 * f / 5 + 3 / 5
        return y

elif func_name == 'modified_sin1D':
    def f(x_0):
        x = (7.5 - 2.7) * x_0 + 2.7
        f = (np.sin(x) + np.sin(10 / 3 * x))
        y = 3 / 4 * f + 1 / 4
        return y

n_units = 10
max_epochs =  60
epoch_list_idx = -1
folder_name = f'{func_name}_L3_regu{regularisation}/{activation}_concdrop'
util = 'se_ysample_clip'
colour_names = ['red','orange','gold','forestgreen','royalblue']
methods = ['concdrop',f'lccd_{util}']

total_itr = 4
n_init = 26
n_per_update = 5
seed = 42
bar_width = 0.003
opaticity = 0.6

# Generate data
np.random.seed(seed)
x_grid = np.linspace(0, 1, 100)[:, None]
x_grid_plot = x_grid.flatten()
fvals = f(x_grid)

if func_name == 'modified_sin1D':
    x_train_unsort = np.random.uniform(0, 1, n_init + n_per_update * total_itr * 3)[:, None]
    y_train_unsort = f(x_train_unsort)
    y_indices = y_train_unsort.argsort(0).flatten()
    y_train = y_train_unsort[y_indices[::-1]]
    x_train = x_train_unsort[y_indices[::-1]]
    x_new_set = [np.random.uniform(0.84, 0.95, n_per_update), np.random.uniform(0.08, 0.2, n_per_update),
                 np.random.uniform(0.43, 0.6, n_per_update), np.random.uniform(0.4, 0.65, n_per_update)]

elif func_name == 'gramcy1D_yval':
    x_train_unsort = np.random.uniform(0, 1, n_init + n_per_update * total_itr)[:, None]
    y_train_unsort = f(x_train_unsort)
    y_indices = y_train_unsort.argsort(0).flatten()
    y_train = y_train_unsort[y_indices[::-1]]
    x_train = x_train_unsort[y_indices[::-1]]
    x_new_set = [np.random.uniform(0.62, 0.73, n_per_update), np.random.uniform(0.33, 0.48, n_per_update),
                 np.random.uniform(0.13, 0.24, n_per_update), np.random.uniform(0.35, 0.65, n_per_update)]


for epoch_list_idx in [0,1]:

    x_old = x_train[:n_init]
    y_old = y_train[:n_init]
    y_new = None
    x_new = None
    x = np.copy(x_old)
    y = np.copy(y_old)

    figure, axes = plt.subplots(3, total_itr, figsize=(25, 10), gridspec_kw={'height_ratios': [2, 2, 1]}, sharex=True)

    for k in range(total_itr):

        subplot_titles = [f'Conc Dropout t={k}', f'LCCD {util} t={k}']
        pred_means, pred_var = [], []
        pred_e_var, pred_a_var  = [], []
        train_time, predict_time  = [], []

        for i in range(len(methods)):

            # ------------ Load data -----------------
            results_file_name = f"{folder_name}/{methods[i]}_results_s{seed}_itr{k}_n{n_units}_e{max_epochs}"
            with open(results_file_name, 'rb') as file:
                results = pickle.load(file)
            epoch_list = results['epoch_list']
            n_epoch = epoch_list[epoch_list_idx]
            train_results = results['train_results']
            val_results = results['val_results']

            print(f'Load results for {methods[i]} at epoch={n_epoch}')

            # concdrop/lccd_val_results: m, v, ev, av, ppp, val loss, predict time
            m = val_results[epoch_list_idx][0].flatten()
            v = val_results[epoch_list_idx][1].flatten()
            ev = val_results[epoch_list_idx][2].flatten()
            av = val_results[epoch_list_idx][3].flatten()

            train_time.append(train_results[epoch_list_idx][-1])
            predict_time.append(val_results[epoch_list_idx][6])

            # ------------ Plot regression -----------------
            axes[i, k].plot(x_grid_plot, fvals, "k--")
            axes[i, k].plot(x_grid_plot, np.mean(y) * np.ones_like(fvals), "m--")

            if k > 0:
                axes[i, k].plot(x_old, y_old, "ko")
                axes[i, k].plot(x_new, y_new, "r^")
            else:
                axes[i, k].plot(x_old, y_old, "r^")


            axes[i, k].plot(x_grid_plot, m, "blue")
            axes[i, k].fill_between(x_grid_plot, m + np.sqrt(v), m - np.sqrt(v), color="blue",
                                    alpha=0.2, label='aleatoric std')
            axes[i, k].fill_between(x_grid_plot, m + np.sqrt(ev), m - np.sqrt(ev), color="blue", alpha=0.4,
                                    label='epistemic std')

            axes[i, k].set_title(subplot_titles[i])
            axes[i, k].set_ylabel('y')
            axes[i, k].legend()

        # ------------ Plot utility at each training point -----------------
        #   lccd_train_results train mse loss, train logutil, log gain for each data point, train time;
        log_gain_average = train_results[epoch_list_idx][-2]
        axes_loggain = axes[-1, k]
        axes_loggain.set_title(f'log gain {util}')
        # log_gain_average_off = log_gain_average + 0.05
        log_gain_average_off = log_gain_average

        if k > 0:
            axes_loggain.bar(x_old.flatten(), log_gain_average_off.flatten()[:-n_per_update],
                             bar_width, alpha=opaticity, color='k', edgecolor='k')
            axes_loggain.bar(x_new.flatten(), log_gain_average_off.flatten()[-n_per_update],
                             bar_width, alpha=opaticity, color='r', edgecolor='r')
        else:
            log_gain_average_off = log_gain_average
            axes_loggain.bar(x.flatten(), log_gain_average_off.flatten(), bar_width, alpha=opaticity, color='r',
                             edgecolor='r')
        # log_gain_average_pos = log_gain_average_off[np.where(log_gain_average_off>0)]
        # axes_loggain.set_ylim([np.min(log_gain_average_pos)-0.05, np.max(log_gain_average_pos)+0.05])

        # bar plot for log conditional gain for each data point averaged over all epoches
        x_old = np.copy(x)
        y_old = np.copy(y)

        # Generate new data
        x_new = x_new_set[k][:, None]
        y_new = f(x_new)
        x = np.vstack((x_old, x_new))
        y = np.vstack((y_old, y_new))

    plt.grid()
    plt.tight_layout()
    figure.suptitle(f'Gramcy N_epoch={n_epoch}', fontsize=14)
    plt.show()

    # ------------ Save plots -----------------
    fig_save_path = os.path.join(f'{func_name}_L3_regu{regularisation}',
                                 f'util{util_str}_s{seed}_lccd_warm_{warm_start}_{func_name}_nunits={n_units}' \
                                 f'_nepochs={num_epochs}_n_init={n_init}_act={act_func}_l=1e-1_total_itr_{total_itr}')
    if save_results:
        figure.savefig(fig_save_path + ".pdf", bbox_inches='tight')

print('hold')
