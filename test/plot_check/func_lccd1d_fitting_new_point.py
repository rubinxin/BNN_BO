import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from exps_tasks.math_functions import get_function
from pybnn.mcdrop import MCDROP
from pybnn.mcconcretedrop import MCCONCRETEDROP
from pybnn.lcbnn import LCBNN
from pybnn.lccd import LCCD
import time
from matplotlib import cm
import argparse

func_name='gramcy1D'
def f(x_0):
    x = 2*x_0 + 0.5
    return (np.sin(x * 4 * np.pi) / (2*x) + (x-1)**4)-4

def fitting_new_points_1D(n_units, num_epochs, seed, T, length_scale, n_init, mc_tau, regul):
    print(f'{func_name}: seed={seed}')
    display_time = True
    dropout = 0.05
    weight_decay = 1e-6
    x_next = 0.2
    np.random.seed(seed)
    x = np.random.rand(n_init)[:, None]
    y = f(x)
    grid = np.linspace(0, 1, 100)[:, None]
    fvals = f(grid)

    # Add 1 new point
    y_next = f(x_next)
    x_new  = np.vstack((x, x_next))
    y_new  = np.vstack((y, y_next))

    # -- Train and Prediction with MC Dropout Model ---
    start_train_time_mcconcdrop = time.time()
    model_mcconcdrop = MCCONCRETEDROP(num_epochs=num_epochs, n_units_1=n_units, n_units_2=n_units,
                                      n_units_3=n_units,length_scale=length_scale, T=T,
                                      mc_tau=mc_tau, regu=regul, rng=seed)
    model_mcconcdrop.train(x, y.flatten())
    train_time_mcconcdrop = time.time() - start_train_time_mcconcdrop
    start_predict_time_mcconcdrop = time.time()
    m_mcconcdrop, v_mcconcdrop = model_mcconcdrop.predict(grid)
    predict_time_mcconcdrop = time.time() - start_predict_time_mcconcdrop

    # train with 1 additional new data
    model_mcconcdrop.train(x_new, y_new.flatten())
    m_mcconcdrop_new, v_mcconcdrop_new = model_mcconcdrop.predict(grid)


    # -- Train and Prediction with lccd with MC Dropout mode and Se_y Util ---
    start_train_time_lccd = time.time()

    util_set = ['se_y','se_yclip','se_prod_y']
    m_lccd_set = []
    v_lccd_set = []
    m_lccd_new_set = []
    v_lccd_new_set = []
    lccd_train_time = []
    lccd_pred_time = []

    for u in util_set:
        model_lccd_u = LCCD(num_epochs=num_epochs, n_units_1=n_units, n_units_2=n_units, n_units_3=n_units,
                             length_scale=length_scale, T=T, util_type=u,
                             mc_tau=mc_tau, regu=regul, rng=seed)
        
        model_lccd_u.train(x, y.flatten())
        train_time_lccd = time.time() - start_train_time_lccd

        start_predict_time_lccd= time.time()
        m_lccd_u, v_lccd_u = model_lccd_u.predict(grid)
        predict_time_lccd = time.time() - start_predict_time_lccd

        # train with 1 additional new data
        model_lccd_u.train(x_new, y_new.flatten())
        m_mcconcdrop_u_new, v_mcconcdrop_u_new = model_lccd_u.predict(grid)

        m_lccd_set.append(m_lccd_u)
        v_lccd_set.append(v_lccd_u)

        m_lccd_new_set.append(m_mcconcdrop_u_new)
        v_lccd_new_set.append(v_mcconcdrop_u_new)
        lccd_train_time.append(train_time_lccd)
        lccd_pred_time.append(predict_time_lccd)

    # Store all the timing
    train_time = [train_time_mcconcdrop] + lccd_train_time
    predict_time = [predict_time_mcconcdrop] + lccd_pred_time

    # -- Plot Results ---
    figure, axes = plt.subplots(2, 4, figsize=(20, 10))

    subplot_titles = ['Conc. Dropout','LCCD '+ util_set[0],
                      'LCCD ' + util_set[1],'LCCD ' + util_set[2]]


    pred_means_old = [m_mcconcdrop] + m_lccd_set
    pred_var_old = [v_mcconcdrop] + v_lccd_set
    pred_means_new = [m_mcconcdrop_new] + m_lccd_new_set
    pred_var_new = [v_mcconcdrop_new] + v_lccd_new_set

    pred_means_set = [pred_means_old, pred_means_new]
    pred_var_set = [pred_var_old, pred_var_new]

    # pred_means.append(m_lccd_set)
    for j in range(len(pred_means_set)):
        pred_means = pred_means_set[j]
        pred_var  = pred_var_set[j]
        for i in range(len(subplot_titles)):
            grid = grid.flatten()
            m = pred_means[i].flatten()
            v = pred_var[i].flatten()

            axes[j, i].plot(x, y, "ro")
            axes[j, i].plot(x_next, y_next, "r^")
            axes[j, i].plot(grid, fvals, "k--")
            axes[j, i].plot(grid, pred_means[i], "blue")
            axes[j, i].fill_between(grid, m + np.sqrt(v), m - np.sqrt(v), color="orange", alpha=0.8)
            axes[j, i].fill_between(grid, m + 2 * np.sqrt(v), m - 2 * np.sqrt(v), color="orange", alpha=0.6)
            axes[j, i].fill_between(grid, m + 3 * np.sqrt(v), m - 3 * np.sqrt(v), color="orange", alpha=0.4)
            axes[j, i].set_title(subplot_titles[i])
            plt.grid()

    # plt.show()
    path = 'figures_compare/' + func_name + 'n_init='+ str(n_init)+'_l=1e-1_seed' + str(seed) + '_nunits=' + \
           str(n_units) + '_nepochs=' + str(num_epochs) + '_wd=' + str(weight_decay) + \
            '_dropout=' + str(dropout) + '_regul=' + str(regul) + '_mc_tau=' + str(mc_tau)
    figure.savefig(path + ".pdf", bbox_inches='tight')

    if display_time:
        for m, train_t, predict_t in zip(subplot_titles, train_time, predict_time):
            print(f'method:{m}, train_time={train_t}, predict_time={predict_t}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
    parser.add_argument('-n', '--n_units', help='Number of neurons per layer',
                        default=50, type=int)
    parser.add_argument('-e', '--n_epochs', help='Number of training epoches',
                        default=500, type=int)
    parser.add_argument('-s', '--seed', help='Random seeds [0,6,11,12,13,23,29]',
                        default=6, type=int)
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

    fitting_new_points_1D(n_units=n_units, num_epochs=n_epochs, seed=s, T=T,
                          length_scale=ls, n_init=n_init, mc_tau = mc_tau, regul = regul)

