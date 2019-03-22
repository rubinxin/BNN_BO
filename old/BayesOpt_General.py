#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""

import numpy as np
from acq_funcs.acquisitions import EI, LCB, MES
from models.gp import GPModel
from models.dropoutnn import DropoutNet
from utilities.utilities import optimise_acqu_func, sample_fmin_Gumble

class Bayes_opt():
    def __init__(self, func, bounds, noise_var):
        self.func = func
        self.bounds = bounds
        self.noise_var = noise_var

    def initialise(self, X_init=None, Y_init=None, kernel=None, n_fmin_samples=1, model_type='GP',dropout=0.05, T = 100, n_hidden=[50, 50, 50], tau=1.0, batch_size=5):
        assert X_init.ndim == 2, "X_init has to be 2D array"
        assert Y_init.ndim == 2, "Y_init has to be 2D array"
        self.X_init = X_init
        self.Y_init = Y_init
        self.X = np.copy(X_init)
        self.Y = np.copy(Y_init)
        self.n_fmin_samples = n_fmin_samples

        # Input dimension
        self.X_dim = self.X.shape[1]
        # Find min observed functional value and its location
        self.arg_opt = np.atleast_2d(self.X[np.argmin(self.Y)])
        self.minY = np.min(self.Y)

        # Specify the model
        if model_type == 'GP':
            self.kernel = kernel
            self.ARD   = True
            if self.noise_var > 1e-8:
                self.model = GPModel(kernel=kernel, noise_var=self.noise_var, ARD=self.ARD )
            else:
                self.model = GPModel(kernel=kernel, exact_feval=True, ARD=self.ARD)
        elif model_type == 'DropNN':
            self.model = DropoutNet(n_hidden=n_hidden, dropout=dropout, T = T, tau=tau, batch_size=batch_size)

    def iteration_step(self, iterations, seed, bo_method):
        np.random.seed(seed)

        X_optimum = np.copy(np.atleast_2d(self.arg_opt))
        Y_optimum = np.copy(np.atleast_2d(self.minY))
        X_for_L2  = np.copy(X_optimum)
        Y_for_IR  = np.copy(Y_optimum)

        self.e = np.exp(1)

        #  Fit GP model to the data
        self.model._update_model(self.X, self.Y)

        # Specify acquisition function
        if bo_method == 'EI':
            acqu_func = EI(self.model)
        elif bo_method == 'LCB':
            acqu_func = LCB(self.model)
        elif bo_method == 'MES':
            fmin_samples = sample_fmin_Gumble(self.model, self.bounds, nMs=self.n_fmin_samples)
            self.model.fmin_samples = fmin_samples
            acqu_func = MES(self.model)

        for k in range(iterations):

            np.random.seed(seed)

            # optimise the acquisition function to get the next query point and evaluate at next query point

            x_next, max_acqu_value = optimise_acqu_func(acqu_func=acqu_func, bounds = self.bounds,X_ob=self.X)
            y_next = self.func(x_next) + np.random.normal(0, np.sqrt(self.noise_var), len(x_next))

            # augment the observation data
            self.X = np.vstack((self.X, x_next))
            self.Y = np.vstack((self.Y, y_next))

            #  update GP model with new data
            self.model._update_model(self.X, self.Y)
            self.minY = np.min(self.Y)

            #  resample the global minimum for MES
            if bo_method == 'MES':
                fmin_samples = sample_fmin_Gumble(self.model, self.bounds, nMs=self.n_fmin_samples)
                self.model.fmin_samples = fmin_samples

            # optimise the marginalised posterior mean to get the prediction for the global optimum/optimiser
            # x_opt, pos_opt = self._global_minimiser_cheap(self.pos_mean,func_gradient=self.d_pos_mean)
            # y_opt = self.func(x_opt)
            #  store data
            x_opt = np.copy(x_next)
            y_opt = np.copy(y_next)
            X_optimum = np.concatenate((X_optimum, np.atleast_2d(x_opt)))
            Y_optimum = np.concatenate((Y_optimum, np.atleast_2d(y_opt)))
            X_for_L2 = np.concatenate((X_for_L2, np.atleast_2d(X_optimum[np.argmin(Y_optimum),:])))
            Y_for_IR = np.concatenate((Y_for_IR, np.atleast_2d(min(Y_optimum))))

            if bo_method == 'MES':
                print( bo_method + ":seed:{seed},itr:{iteration},fmin_sampled:{min_sample}, x_next: {next_query_loc},y_next:{next_query_value}, acq value: {best_acquisition_value},"
                                        "x_opt:{x_opt_pred},y_opt:{y_opt_pred}"
                    .format(seed=seed, iteration=k,
                            min_sample = np.max(fmin_samples),
                            next_query_loc=x_next, next_query_value=y_next,
                            best_acquisition_value=max_acqu_value,
                            x_opt_pred=X_for_L2[-1, :],
                            y_opt_pred=Y_for_IR[-1, :]
                            ))
            else:
                print( bo_method +"seed:{seed},itr:{iteration},x_next: {next_query_loc},y_next:{next_query_value}, acq value: {best_acquisition_value},"
                    "x_opt:{x_opt_pred},y_opt:{y_opt_pred}"
                    .format(seed = seed,iteration=k,
                            next_query_loc=x_next,next_query_value=y_next,
                            best_acquisition_value=max_acqu_value,
                            x_opt_pred=X_for_L2[-1,:],
                            y_opt_pred=Y_for_IR[-1,:]
                            ))

        return self.X, self.Y, X_for_L2, Y_for_IR

