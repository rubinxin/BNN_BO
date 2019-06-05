import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from exps_tasks.math_functions import get_function
from pybnn.mcdrop_test_n_hidden import MCDROP
from pybnn.lcbnn_test_n_hidden import LCBNN
import time
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from matplotlib import cm
import argparse
import os

func_name='gramcy1D_yval'
def f(x_0):
    x = 2*x_0 + 0.5
    return (np.sin(x * 4 * np.pi) / (2*x) + (x-1)**4)-4

def fitting_new_points_1D(n_units, num_epochs, seed, T, length_scale, n_init, mc_tau, regul, warm_start):
    print(f'{func_name}: seed={seed}')
    n_units_list  = [int(item) for item in n_units.split(',')]

    saving_path = f'../data_debug/{func_name}/l{len(n_units_list)}_n{n_units_list[-1]}/'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    saving_model_path = os.path.join(saving_path,
                                 'saved_model/')
    if not os.path.exists(saving_model_path):
        os.makedirs(saving_model_path)

    # act_func = 'relu'
    act_func = 'tanh'

    saving_loss_path = os.path.join(saving_path,
                                 f'{act_func}/')
    if not os.path.exists(saving_loss_path):
        os.makedirs(saving_loss_path)

    save_loss = True
    display_time = True
    dropout = 0.05
    weight_decay = 1e-6
    n_per_update = 5
    total_itr = 5
    normalize_input_and_output = True

    np.random.seed(seed)
    x_grid = np.linspace(0, 1, 100)[:, None]
    fvals = f(x_grid)

    x_train = np.random.uniform(0, 1, 20+n_per_update*total_itr)[:, None]
    x_train = np.sort(x_train, 0)
    y_train = f(x_train)

    if not normalize_input_and_output:
        x_grid, x_mean, x_std = zero_mean_unit_var_normalization(x_grid)
        fvals, f_mean, f_std = zero_mean_unit_var_normalization(fvals)
        x_train = (x_train - x_mean)/x_std
        y_train = (y_train - f_mean)/f_std

    x_old = x_train[:20]
    y_old = y_train[:20]

    x     = np.copy(x_old)
    y     = np.copy(y_old)

    figure, axes = plt.subplots(2, total_itr, figsize=(20, 10))

    for k in range(total_itr):

        if warm_start:
            itr_k = k
        else:
            itr_k = 0

        # -- Train and Prediction with MC Dropout Model ---
        start_train_time_mc = time.time()
        model_mcdrop = MCDROP(num_epochs=num_epochs,n_units=n_units_list,
                              dropout_p=dropout,weight_decay=weight_decay, length_scale=length_scale,
                              T=T, rng=seed, actv=act_func, normalize_input=normalize_input_and_output, normalize_output=normalize_input_and_output)
        # Train
        mcdrop_train_mse_loss = model_mcdrop.train(x, y.flatten(), itr=itr_k, saving_path=saving_model_path)
        train_time_mc = time.time() - start_train_time_mc
        start_predict_time_mc = time.time()
        m_mcdrop, v_mcdrop = model_mcdrop.predict(x_grid)
        # Predict
        predict_time_mc = time.time() - start_predict_time_mc
        # m_mcdrop = zero_mean_unit_var_denormalization(m_mcdrop, y_train_mean, y_train_std)
        # v_mcdrop *= y_train_std ** 2

        if save_loss:
            mcdrop_loss_saving_path = os.path.join(saving_loss_path,
                                                   f"mcdrop_train_mes_loss_s{seed}_itr{k}_n{n_units}_e{num_epochs}")
            np.save(mcdrop_loss_saving_path, mcdrop_train_mse_loss)

        # -- Train and Prediction with LCBNN with MC Dropout mode and Se_y Util ---
        util_set = ['iteration']
        m_lcbnn_set = []
        v_lcbnn_set = []
        lcbnn_train_time = []
        lcbnn_pred_time = []

        for u in util_set:
            start_train_time_lcbnn = time.time()
            model_lcbnn_u = LCBNN(num_epochs=num_epochs,n_units=n_units_list,
                                   weight_decay=weight_decay, length_scale=length_scale, T=T, util_type=u, rng=seed,
                                  actv=act_func, normalize_input=normalize_input_and_output, normalize_output=normalize_input_and_output)
            # Train
            # augment input with k
            lcbnn_train_mse_loss, lcbnn_train_logutil = \
                model_lcbnn_u.train(x, y.flatten(), itr=itr_k, n_per_itr=n_per_update, saving_path=saving_model_path)

            train_time_lcbnn = time.time() - start_train_time_lcbnn

            start_predict_time_lcbnn= time.time()
            # Predict
            m_lcbnn_u, v_lcbnn_u = model_lcbnn_u.predict(x_grid)
            predict_time_lcbnn = time.time() - start_predict_time_lcbnn

            # m_lcbnn_u = zero_mean_unit_var_denormalization(m_lcbnn_u, y_train_mean, y_train_std)
            # v_lcbnn_u *= y_train_std**2

            if save_loss:
                lcbnn_loss_saving_path = os.path.join(saving_loss_path,
                                                       f"lcbnn_train_mes_loss_{u}_s{seed}_itr{k}_n{n_units}_e{num_epochs}")
                lcbnn_logutil_saving_path = os.path.join(saving_loss_path,
                                                       f"lcbnn_train_logutil_{u}_s{seed}_itr{k}_n{n_units}_e{num_epochs}")
                np.save(lcbnn_loss_saving_path, lcbnn_train_mse_loss)
                np.save(lcbnn_logutil_saving_path, lcbnn_train_logutil)

            m_lcbnn_set.append(m_lcbnn_u)
            v_lcbnn_set.append(v_lcbnn_u)

            lcbnn_train_time.append(train_time_lcbnn)
            lcbnn_pred_time.append(predict_time_lcbnn)

        # Store all the timing
        train_time = [train_time_mc] + lcbnn_train_time
        predict_time = [predict_time_mc] + lcbnn_pred_time

        # -- Plot Results ---
        subplot_titles = [f'MC Dropout t={k}',f'LCBNN{util_set[0]} t={k}']
        pred_means = [m_mcdrop] + m_lcbnn_set
        pred_var = [v_mcdrop] + v_lcbnn_set

        for i in range(len(pred_means)):
            x_grid_plot = x_grid.flatten()
            m = pred_means[i].flatten()
            v = pred_var[i].flatten()
            axes[i, k].plot(x_grid_plot, fvals, "k--")

            if k > 0:
                axes[i, k].plot(x_old, y_old, "ko")
                axes[i, k].plot(x_new, y_new, "r^")
            else:
                axes[i, k].plot(x_old, y_old, "r^")

            axes[i, k].plot(x_grid_plot, pred_means[i], "blue")
            axes[i, k].fill_between(x_grid_plot, m + np.sqrt(v), m - np.sqrt(v), color="blue", alpha=0.2)
            axes[i, k].set_title(subplot_titles[i])
            plt.grid()

        x_old = np.copy(x)
        y_old = np.copy(y)
        # Generate new data
        x_new = x_train[20+n_per_update*k:20+n_per_update*(k+1)]
        y_new = y_train[20+n_per_update*k:20+n_per_update*(k+1)]

        x     = np.vstack((x_old,x_new))
        y     = np.vstack((y_old,y_new))

    plt.show()

    fig_save_path = os.path.join(saving_path, f's{seed}_lcbnn_warm_{warm_start}_{func_name}_nunits={n_units}' \
           f'_nepochs={num_epochs}_n_init={n_init}_act={act_func}_l=1e-1_total_itr_{total_itr}')

    figure.savefig(fig_save_path + ".pdf", bbox_inches='tight')

    if display_time:
        for m, train_t, predict_t in zip(subplot_titles, train_time, predict_time):
            print(f'method:{m}, train_time={train_t}, predict_time={predict_t}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
    # parser.add_argument('-n', '--n_units', help='Number of neurons per layer',
    #                     default=5, type=int)
    parser.add_argument('-n', '--n_units', help='Number of neurons per layer',
                        default='100', type=str)
    parser.add_argument('-e', '--n_epochs', help='Number of training epoches',
                        default=200, type=int)
    parser.add_argument('-s', '--seed', help='Random seeds [0,6,11,12,13,23,29]',
                        default=42, type=int)
    parser.add_argument('-t', '--samples', help='MC samples for prediction',
                        default=100, type=int)
    parser.add_argument('-l', '--ls', help='length scale value',
                        default=0.1, type=float)
    parser.add_argument('-i', '--n_init', help='Number of initial data',
                        default=20, type=int)
    parser.add_argument('-m', '--mc_tau', help='Learn tau empirically using MC samples during training',
                        default=False, type=bool)
    parser.add_argument('-r', '--regul', help='Add regularisation to training losses',
                        default=False, type=bool)
    parser.add_argument('-c', '--continue_training', help='Cold start (False) or Warm start (True)',
                        default=True, type=bool)

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
    warm = args.continue_training

    fitting_new_points_1D(n_units=n_units, num_epochs=n_epochs, seed=s, T=T,
                          length_scale=ls, n_init=n_init, mc_tau = mc_tau, regul = regul, warm_start = warm)

