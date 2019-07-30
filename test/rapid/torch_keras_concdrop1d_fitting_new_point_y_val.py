import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pybnn.mcconcretedrop_test import MCCONCRETEDROP
from pybnn.mcconcretedrop_keras import MCCONCRETEDROP_KERAS

import time
from matplotlib import cm
import argparse
import os
import pickle

def fitting_new_points_1D(func_name, n_units, num_epochs, seed, T, length_scale, n_init, mc_tau, regul, warm_start,
                          util_str, activation, n_per_update, epoch_interval, p_min, p_max):
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

    saving_path = f'data_debug/{func_name}_L{3}_regu{regul}_n{n_units}_ls{length_scale}_keras/'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    saving_model_path = os.path.join(saving_path,
                                 'saved_model_concdrop/')
    if not os.path.exists(saving_model_path):
        os.makedirs(saving_model_path)

    act_func = activation

    save_results_path = os.path.join(saving_path,
                                 f'{act_func}_concdrop/')
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    save_results = True
    display_time = True
    # epoch_interval = int(num_epochs/n_val_points)
    epoch_interval = epoch_interval
    # epoch_interval = 5
    epoch_ranges = range(50, num_epochs, epoch_interval)
    # epoch_ranges = [num_epochs]
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

    figure, axes = plt.subplots(2, total_itr, figsize=(25, 10),  gridspec_kw={'height_ratios': [5, 5]},sharex=True)

    for k in range(total_itr):

        if warm_start:
            itr_k = k
        else:
            itr_k = 0

        concdrop_train_results_allepoch = []
        concdrop_val_results_allepoch = []

        concdropkeras_train_results_allepoch = []
        concdropkeras_val_results_allepoch = []

        epoch_list = []

        for epochs in epoch_ranges:
            epoch_list.append(epochs)
            # -----------------------------------------------------------
            # -- Train and Prediction with Conc Dropout Dropout Model ---
            # -----------------------------------------------------------

            start_train_time_concdrop = time.time()
            model_concdrop = MCCONCRETEDROP(num_epochs=epochs, n_units_1=n_units, n_units_2=n_units,
                                              n_units_3=n_units, length_scale=length_scale, T=T,
                                              mc_tau=mc_tau, regu=regul, rng=seed,
                                              normalize_input=normalise, normalize_output=normalise, actv=act_func,
                                            init_p_min=p_min, init_p_max=p_max)

            # Train
            concdrop_train_mse_loss = model_concdrop.train(x, y.flatten(), itr=itr_k, saving_path=saving_model_path)
            train_time_concdrop = time.time() - start_train_time_concdrop
            start_predict_time_concdrop = time.time()

            # Predict
            m_concdrop, v_concdrop, ev_concdrop, av_concdrop, concdrop_val_loglike_per_point, concdrop_val_loss = model_concdrop.validate(x_grid, fvals)
            predict_time_concdrop = time.time() - start_predict_time_concdrop

            # Store results
            concdrop_train_results = [concdrop_train_mse_loss, train_time_concdrop]
            concdrop_train_results_allepoch.append(concdrop_train_results)

            concdrop_val_results = [m_concdrop, v_concdrop, ev_concdrop, av_concdrop, concdrop_val_loglike_per_point, concdrop_val_loss, predict_time_concdrop]
            concdrop_val_results_allepoch.append(concdrop_val_results)

            # ----------------------------------------------------------------
            # -- Train and Prediction with LCCD with Conc Dropout and Util ---
            # ----------------------------------------------------------------


            start_train_time_concdropkeras = time.time()
            model_concdropkeras_u = MCCONCRETEDROP_KERAS(num_epochs=epochs, n_units_1=n_units, n_units_2=n_units,
                                              n_units_3=n_units, length_scale=length_scale, T=T,
                                              mc_tau=mc_tau, regu=regul, rng=seed,
                                              normalize_input=normalise, normalize_output=normalise, actv=act_func,
                                            init_p_min=p_min, init_p_max=p_max)
            # Train
            concdropkeras_train_mse_loss = model_concdropkeras_u.train(x, y.flatten(), itr=itr_k, saving_path=saving_model_path)

            train_time_concdropkeras = time.time() - start_train_time_concdropkeras

            start_predict_time_concdropkeras= time.time()
            # Predict
            m_concdropkeras_u, v_concdropkeras_u, ev_concdropkeras_u, av_concdropkeras_u, concdropkeras_val_loglike_per_point, concdropkeras_val_loss = model_concdropkeras_u.validate(x_grid, fvals)
            predict_time_concdropkeras = time.time() - start_predict_time_concdropkeras

            # Store results
            concdropkeras_train_results = [concdropkeras_train_mse_loss, train_time_concdropkeras]
            concdropkeras_train_results_allepoch.append(concdropkeras_train_results)

            concdropkeras_val_results = [m_concdropkeras_u, v_concdropkeras_u, ev_concdropkeras_u, av_concdropkeras_u, concdropkeras_val_loglike_per_point, concdropkeras_val_loss, predict_time_concdropkeras]
            concdropkeras_val_results_allepoch.append(concdropkeras_val_results)


            if save_results:
                # Save train loss and validation loss/ll of concrete dropout
                conc_results_path = os.path.join(save_results_path, f"concdrop_results_s{seed}_itr{k}_n{n_units}_e{num_epochs}")

                # concdrop_train_results: train mse loss,train time
                # concdrop_val_results: m, v, ev, av, ppp, val loss, predict time
                conc_results = {'epoch_list': epoch_list,
                                'train_results': concdrop_train_results_allepoch,
                                'val_results': concdrop_val_results_allepoch}
                with open(conc_results_path, 'wb') as conc_file:
                    pickle.dump(conc_results, conc_file)

                # Save train loss and utils of LCCD and validation loss/ll of LCCD
                concdropkeras_results_path = os.path.join(save_results_path, f"concdropkeras_{util_str}_results_s{seed}_itr{k}_n{n_units}_e{num_epochs}")

                # concdropkeras_train_results: train mse loss, train logutil, log gain for each data point, train time
                # concdropkeras_val_results: m, v, ev, av, ppp, val loss, predict time
                concdropkeras_results = {'epoch_list': epoch_list,
                                'train_results': concdropkeras_train_results_allepoch,
                                'val_results': concdropkeras_val_results_allepoch,
                                'y_old': y_old,
                                'y_new': y_new}

                with open(concdropkeras_results_path, 'wb') as concdropkeras_file:
                    pickle.dump(concdropkeras_results, concdropkeras_file)

        # -- Plot Final Epoch Results ---
        subplot_titles = [f'Conc Dropout t={k}',f'Keras Version t={k}']
        pred_means = [m_concdrop, m_concdropkeras_u]
        pred_means = [m_concdrop, m_concdropkeras_u]
        pred_var = [v_concdrop, v_concdropkeras_u]
        pred_e_var = [ev_concdrop, ev_concdropkeras_u]
        pred_a_var = [av_concdrop, av_concdropkeras_u]
        train_time = [train_time_concdrop, train_time_concdropkeras]
        predict_time = [predict_time_concdrop, predict_time_concdropkeras]
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

            else:
                axes[i, k].plot(x_old, y_old, "r^")

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

    fig_save_path = os.path.join(saving_path, f'util{util_str}_s{seed}_concdropkeras_warm_{warm_start}_{func_name}_nunits={n_units}' \
           f'_nepochs={num_epochs}_n_init={n_init}_act={act_func}_l=1e-1_total_itr_{total_itr}')
    if save_results:
        figure.savefig(fig_save_path + ".pdf", bbox_inches='tight')

    if display_time:
        for m, train_t, predict_t in zip(subplot_titles, train_time, predict_time):
            print(f'method:{m}, train_time={train_t}, predict_time={predict_t}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
    parser.add_argument('-n', '--n_units', help='Number of neurons per layer',
                        default=50, type=int)
    parser.add_argument('-e', '--n_epochs', help='Number of training epoches',
                        default=5000, type=int)
    parser.add_argument('-s', '--seed', help='Random seeds [0,6,11,12,13,23,29]',
                        default=42, type=int)
    parser.add_argument('-t', '--samples', help='MC samples for prediction',
                        default=50, type=int)
    parser.add_argument('-ne', '--new', help='Number of new points per iteration',
                        default=5, type=int)
    parser.add_argument('-l', '--ls', help='length scale value',
                        default=0.1, type=float)
    parser.add_argument('-i', '--n_init', help='Number of initial data',
                        default=26, type=int)
    parser.add_argument('-m', '--mc_tau', help='Learn tau empirically using MC samples during training',
                        default=False, type=bool)
    parser.add_argument('-pl', '--pmin', help='initial min dropout p ',
                        default=0.1, type=float)
    parser.add_argument('-pu', '--pmax', help='initial max dropout p ',
                        default=0.1, type=float)
    parser.add_argument('-r', '--regul', help='Add regularisation to training losses',
                        default=True, type=bool)
    parser.add_argument('-c', '--continue_training', help='Cold start (False) or Warm start (True)',
                        default=True, type=bool)
    parser.add_argument('-u', '--utility_type', help='Utlity function type',
                        default='linear_se_ytrue_clip', type=str)
    parser.add_argument('-a', '--actv_func', help='Activation function',
                        default='tanh', type=str)
    parser.add_argument('-f', '--func_name', help='Test function',
                        default='gramcy1D_yval', type=str)
    parser.add_argument('-nv', '--n_interval', help='Epoch interval to run validation',
                        default=4940, type=int)

    args = parser.parse_args()
    print(f"Got arguments: \n{args}")
    n_units = args.n_units
    n_epochs = args.n_epochs
    s = args.seed
    T = args.samples
    ls = args.ls
    n_init = args.n_init
    mc_tau = args.mc_tau
    regul = args.regul
    n_new = args.new
    warm = args.continue_training
    util = args.utility_type
    actv_func = args.actv_func
    func = args.func_name
    n_interval = args.n_interval
    pmax = args.pmax
    pmin = args.pmin

    fitting_new_points_1D(func_name= func, n_units=n_units, num_epochs=n_epochs, seed=s, T=T,
                          length_scale=ls, n_init=n_init, mc_tau = mc_tau, regul = regul, warm_start = warm,
                          util_str=util, activation = actv_func, n_per_update=n_new, epoch_interval=n_interval,
                          p_min=pmin, p_max=pmax)

