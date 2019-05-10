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
obj_func = 'egg-2d'        # all completed except LCCDtanh, LCCDrelu, MCCONCrelu

numEpoch = 1000
batch_size = 1
bo_method = 'LCB'
models_all = ['DNGO','LCBNNtanhse_y','MCDROPtanh','LCCDtanh','MCCONCtanh','LCBNNreluse_y','MCDROPrelu']
seed = 0
fmt_list = ['s', '^', 'o']

itr_range = 60

idx = range(0,1000-2,50)
util_set = ['se_y','se_prod_y','se_yclip']

for k in range(0,itr_range,10):
    figure, axes = plt.subplots(2, 2, figsize=(16 , 8), sharex='col', sharey='row')
    axes_list = axes_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for i, util in enumerate(util_set):

        mse_loss_file_name = obj_func + '/' + f"s{seed}_itr{k}lcbnn_train_mes_loss_{util}.npy"
        util_file_name = obj_func + '/' + f"s{seed}_itr{k}lcbnn_train_util_{util}.npy"

        mse_loss = np.load(mse_loss_file_name)
        log_util = np.load(util_file_name)
        mean_mse_loss = np.mean(mse_loss, 1)
        mean_log_util = np.mean(log_util, 1)
        mean_calibrated_loss = np.mean(mse_loss - log_util,1)

        axes_list[i].plot(idx,mean_mse_loss[idx], '--', label=f'mse_l')
        axes_list[i].plot(idx, - mean_log_util[idx],'-.', label=f'nega_logU')
        axes_list[i].plot(idx,mean_calibrated_loss[idx],'-', label=f'calibrated_l')

        axes_list[i].set_title(f'{util}')
        axes_list[i].legend(loc='upper right')

        axes_list[-1].plot(idx,mean_mse_loss[idx], '--', label=f'{util}')

    axes_list[-1].set_title('mse_loss')
    axes_list[-1].legend(loc='upper right')

    figure.suptitle(f"Egg-2D: MSE Loss/Log Utility/ Calibrated Loss at itr{k} seed{seed} ", fontsize=16)
    # plt.legend(prop={'size': 12})
    # plt.show()
    figure.savefig(f'{obj_func}/figures/debug_loss_plots_itr{k}_seed{seed}.pdf', bbox_inches='tight')
    plt.show()
    print('hold')
