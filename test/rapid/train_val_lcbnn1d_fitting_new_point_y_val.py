import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pybnn.mcdrop_test import MCDROP
from pybnn.lcbnn_test import LCBNN
import time
from matplotlib import cm
import argparse
import os
import pickle

def fitting_new_points_1D(func_name, n_units, num_epochs, seed, T, length_scale, n_init, mc_tau, regul, warm_start,
                          util_str, activation, n_per_update, n_val_points, drop_p):
    print(f'{func_name}: seed={seed}')
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

    saving_path = f'data_debug/{func_name}_L{3}_regu{regul}_lcbnn_{drop_p}_n{n_units}/'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    saving_model_path = os.path.join(saving_path,
                                 'saved_model_mcdrop/')
    if not os.path.exists(saving_model_path):
        os.makedirs(saving_model_path)

    act_func = activation

    save_results_path = os.path.join(saving_path,
                                 f'{act_func}_mcdrop/')
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    save_results = True
    display_time = True
    epoch_interval = int((num_epochs-50)/n_val_points)-1
    # epoch_interval = epoch_interval
    # epoch_interval = 5

    dropout = drop_p
    weight_decay = 1e-6
    total_itr = 4
    normalise = False
    np.random.seed(seed)
    x_grid = np.linspace(0, 1, 100)[:, None]
    fvals = f(x_grid)

    # x_new datas
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

    else:
        print('not implemented')

    x_old = x_train[:n_init]
    y_old = y_train[:n_init]
    y_new = None
    x_new = None
    x     = np.copy(x_old)
    y     = np.copy(y_old)

    figure, axes = plt.subplots(3, total_itr, figsize=(25, 10),  gridspec_kw={'height_ratios': [2, 2, 1]},sharex=True)

    for k in range(total_itr):

        if warm_start:
            itr_k = k
        else:
            itr_k = 0

        mcdrop_train_results_allepoch = []
        mcdrop_val_results_allepoch = []

        lcbnn_train_results_allepoch = []
        lcbnn_val_results_allepoch = []

        epoch_list = []

        for epochs in range(50, num_epochs, epoch_interval):
            epoch_list.append(epochs)
            # -----------------------------------------------------------
            # -- Train and Prediction with Conc Dropout Dropout Model ---
            # -----------------------------------------------------------

            start_train_time_mcdrop = time.time()
            model_mcdrop = MCDROP(num_epochs=epochs,n_units_1=n_units, n_units_2=n_units, n_units_3=n_units, dropout_p=dropout,weight_decay=weight_decay, length_scale=length_scale,
                              T=T, rng=seed, actv=act_func,
                                              normalize_input=normalise, normalize_output=normalise)

            # Train
            mcdrop_train_mse_loss = model_mcdrop.train(x, y.flatten(), itr=itr_k, saving_path=saving_model_path)
            train_time_mcdrop = time.time() - start_train_time_mcdrop
            start_predict_time_mcdrop = time.time()

            # Predict
            m_mcdrop, v_mcdrop, ev_mcdrop, av_mcdrop, mcdrop_val_loglike_per_point, mcdrop_val_loss = model_mcdrop.validate(x_grid, fvals)
            predict_time_mcdrop = time.time() - start_predict_time_mcdrop

            # Store results
            mcdrop_train_results = [mcdrop_train_mse_loss, train_time_mcdrop]
            mcdrop_train_results_allepoch.append(mcdrop_train_results)

            mcdrop_val_results = [m_mcdrop, v_mcdrop, ev_mcdrop, av_mcdrop, mcdrop_val_loglike_per_point, mcdrop_val_loss, predict_time_mcdrop]
            mcdrop_val_results_allepoch.append(mcdrop_val_results)

            # ----------------------------------------------------------------
            # -- Train and Prediction with lcbnn with Conc Dropout and Util ---
            # ----------------------------------------------------------------


            start_train_time_lcbnn = time.time()
            model_lcbnn_u = LCBNN(num_epochs=epochs, n_units_1=n_units, n_units_2=n_units,n_units_3=n_units,
                                weight_decay=weight_decay, length_scale=length_scale, T=T, util_type=util_str, 
                                 rng=seed, actv=act_func,dropout_p=dropout, normalize_input=normalise, normalize_output=normalise)
            # Train
            lcbnn_train_mse_loss, lcbnn_train_logutil, log_gain_average = \
                model_lcbnn_u.train(x, y.flatten(), itr=itr_k, n_per_itr=n_per_update, saving_path=saving_model_path)

            train_time_lcbnn = time.time() - start_train_time_lcbnn

            start_predict_time_lcbnn= time.time()
            # Predict
            m_lcbnn_u, v_lcbnn_u, ev_lcbnn_u, av_lcbnn_u, lcbnn_val_loglike_per_point, lcbnn_val_loss = model_lcbnn_u.validate(x_grid, fvals)
            predict_time_lcbnn = time.time() - start_predict_time_lcbnn

            # Store results
            lcbnn_train_results = [lcbnn_train_mse_loss, lcbnn_train_logutil, log_gain_average, train_time_lcbnn]
            lcbnn_train_results_allepoch.append(lcbnn_train_results)

            lcbnn_val_results = [m_lcbnn_u, v_lcbnn_u, ev_lcbnn_u, av_lcbnn_u, lcbnn_val_loglike_per_point, lcbnn_val_loss, predict_time_lcbnn]
            lcbnn_val_results_allepoch.append(lcbnn_val_results)


            if save_results:
                # Save train loss and validation loss/ll of concrete dropout
                mcdrop_results_path = os.path.join(save_results_path, f"mcdrop_results_s{seed}_itr{k}_n{n_units}_e{num_epochs}")

                # mcdrop_train_results: train mse loss,train time
                # mcdrop_val_results: m, v, ev, av, ppp, val loss, predict time
                mcdrop_results = {'epoch_list': epoch_list,
                                'train_results': mcdrop_train_results_allepoch,
                                'val_results': mcdrop_val_results_allepoch}
                with open(mcdrop_results_path, 'wb') as mcdrop_file:
                    pickle.dump(mcdrop_results, mcdrop_file)

                # Save train loss and utils of lcbnn and validation loss/ll of lcbnn
                lcbnn_results_path = os.path.join(save_results_path, f"lcbnn_{util_str}_results_s{seed}_itr{k}_n{n_units}_e{num_epochs}")

                # lcbnn_train_results: train mse loss, train logutil, log gain for each data point, train time
                # lcbnn_val_results: m, v, ev, av, ppp, val loss, predict time
                lcbnn_results = {'epoch_list': epoch_list,
                                'train_results': lcbnn_train_results_allepoch,
                                'val_results': lcbnn_val_results_allepoch,
                                'y_old': y_old,
                                'y_new': y_new}

                with open(lcbnn_results_path, 'wb') as lcbnn_file:
                    pickle.dump(lcbnn_results, lcbnn_file)

        # -- Plot Final Epoch Results ---
        subplot_titles = [f'MCDrop t={k}',f'LC MCDrop {util_str} t={k}']
        pred_means = [m_mcdrop, m_lcbnn_u]
        pred_var = [v_mcdrop, v_lcbnn_u]
        pred_e_var = [ev_mcdrop, ev_lcbnn_u]
        pred_a_var = [av_mcdrop, av_lcbnn_u]
        train_time = [train_time_mcdrop, train_time_lcbnn]
        predict_time = [predict_time_mcdrop, predict_time_lcbnn]
        for i in range(len(pred_means)):
            x_grid_plot = x_grid.flatten()
            m = pred_means[i].flatten()
            v = pred_var[i].flatten()
            ev = pred_e_var[i].flatten()
            av = pred_a_var[i].flatten()

            bar_width = 0.003
            opaticity = 0.6
            axes[i, k].plot(x_grid_plot, fvals, "k--")
            axes[i, k].plot(x_grid_plot, np.mean(y) * np.ones_like(fvals), "m--")

            if k > 0:
                axes[i, k].plot(x_old, y_old, "ko")
                axes[i, k].plot(x_new, y_new, "r^")
                axes_loggain = axes[-1, k]
                axes_loggain.set_title(f'log gain {util_str}')
                # log_gain_average_off = log_gain_average + 0.05
                log_gain_average_off = log_gain_average
                axes_loggain.bar(x_old.flatten(), log_gain_average_off.flatten()[:-n_per_update],
                                 bar_width, alpha=opaticity, color='k', edgecolor='k')
                axes_loggain.bar(x_new.flatten(), log_gain_average_off.flatten()[-n_per_update],
                                 bar_width, alpha=opaticity, color='r',edgecolor='r')
                # log_gain_average_pos = log_gain_average_off[np.where(log_gain_average_off>0)]
                # axes_loggain.set_ylim([np.min(log_gain_average_pos)-0.05, np.max(log_gain_average_pos)+0.05])

                # axes_loggain.set_ylim([0, 1])
            else:
                axes[i, k].plot(x_old, y_old, "r^")
                axes_loggain = axes[-1, k]
                axes_loggain.set_title(f'log gain {util_str}')
                # log_gain_average_off = log_gain_average + 0.05
                log_gain_average_off = log_gain_average

                axes_loggain.bar(x.flatten(), log_gain_average_off.flatten(), bar_width, alpha=opaticity, color='r',
                                 edgecolor='r')
                # log_gain_average_pos = log_gain_average_off[np.where(log_gain_average_off>0)]
                # axes_loggain.set_ylim([np.min(log_gain_average_pos)-0.05, np.max(log_gain_average_pos)+0.05])

            axes[i, k].plot(x_grid_plot, pred_means[i], "blue")
            # axes[i, k].fill_between(x_grid_plot, m + np.sqrt(v), m - np.sqrt(v), color="blue", alpha=0.2)
            axes[i, k].fill_between(x_grid_plot, m + np.sqrt(ev) + np.sqrt(av), m - np.sqrt(ev) - np.sqrt(av), color="blue", alpha=0.2, label='aleatoric std')
            axes[i, k].fill_between(x_grid_plot, m + np.sqrt(ev), m - np.sqrt(ev), color="blue", alpha=0.4, label='epistemic std')

            axes[i, k].set_title(subplot_titles[i])
            axes[i, k].set_ylabel('y')
            axes[i, k].legend()
            plt.grid()

        # bar plot for log conditional gain for each data point averaged over all epoches
        # axes_loggain = axes[-1, k].twinx()
        x_old = np.copy(x)
        y_old = np.copy(y)
        # Generate new data
        # x_new = x_train[20+n_per_update*k:20+n_per_update*(k+1)]
        # y_new = y_train[20+n_per_update*k:20+n_per_update*(k+1)]
        x_new = x_new_set[k][:, None]
        y_new = f(x_new)
        x     = np.vstack((x_old,x_new))
        y     = np.vstack((y_old,y_new))

    plt.tight_layout()
    plt.show()

    fig_save_path = os.path.join(saving_path, f'util{util_str}_s{seed}_lcbnn_warm_{warm_start}_{func_name}_nunits={n_units}' \
           f'_nepochs={num_epochs}_n_init={n_init}_act={act_func}_l=1e-1_total_itr_{total_itr}')
    if save_results:
        figure.savefig(fig_save_path + ".pdf", bbox_inches='tight')

    if display_time:
        for m, train_t, predict_t in zip(subplot_titles, train_time, predict_time):
            print(f'method:{m}, train_time={train_t}, predict_time={predict_t}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
    parser.add_argument('-n', '--n_units', help='Number of neurons per layer',
                        default=10, type=int)
    parser.add_argument('-e', '--n_epochs', help='Number of training epoches',
                        default=5000, type=int)
    parser.add_argument('-s', '--seed', help='Random seeds [0,6,11,12,13,23,29]',
                        default=42, type=int)
    parser.add_argument('-t', '--samples', help='MC samples for prediction',
                        default=80, type=int)
    parser.add_argument('-ne', '--new', help='Number of new points per iteration',
                        default=5, type=int)
    parser.add_argument('-l', '--ls', help='length scale value',
                        default=0.1, type=float)
    parser.add_argument('-p', '--drop_p', help='dropout probability',
                        default=0.05, type=float)
    parser.add_argument('-i', '--n_init', help='Number of initial data',
                        default=26, type=int)
    parser.add_argument('-m', '--mc_tau', help='Learn tau empirically using MC samples during training',
                        default=False, type=bool)
    parser.add_argument('-r', '--regul', help='Add regularisation to training losses',
                        default=False, type=bool)
    parser.add_argument('-c', '--continue_training', help='Cold start (False) or Warm start (True)',
                        default=True, type=bool)
    parser.add_argument('-u', '--utility_type', help='Utlity function type',
                        default='linear_se_ytrue_clip', type=str)
    parser.add_argument('-a', '--actv_func', help='Activation function',
                        default='tanh', type=str)
    parser.add_argument('-f', '--func_name', help='Test function',
                        default='gramcy1D_yval', type=str)
    parser.add_argument('-nv', '--n_interval', help='Number of validation points',
                        default=1, type=int)

    args = parser.parse_args()
    print(f"Got arguments: \n{args}")
    n_units = args.n_units
    n_epochs = args.n_epochs
    s = args.seed
    T = args.samples
    ls = args.ls
    drop_p = args.drop_p

    n_init = args.n_init
    mc_tau = args.mc_tau
    regul = args.regul
    n_new = args.new
    warm = args.continue_training
    util = args.utility_type
    actv_func = args.actv_func
    func = args.func_name
    n_interval = args.n_interval

    fitting_new_points_1D(func_name= func, n_units=n_units, num_epochs=n_epochs, seed=s, T=T,
                          length_scale=ls, n_init=n_init, mc_tau = mc_tau, regul = regul, warm_start = warm,
                          util_str=util, activation = actv_func, n_per_update=n_new, n_val_points=n_interval, drop_p=drop_p)

