#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""

import numpy as np
from acq_funcs.acquisitions import EI, LCB, MES
from acq_funcs.acq_optimizer import Acq_Optimizer
from models.gp import GPModel
from models.mcdrop import MCDROPWarp
from models.dngo import DNGOWrap
from models.bohamiann import BOHAMIANNWarp
from models.lcbnn import LCBNNWarp
from models.lccd import LCCDWarp
from models.mcconcdrop import MCCONCDROPWarp
from utilities.utilities import sample_fmin_Gumble
import time

class Bayes_opt():
    def __init__(self, func, bounds, noise_var, saving_path):
        self.func = func
        self.bounds = bounds
        self.noise_var = noise_var
        self.saving_path = saving_path

    def initialise(self, X_init=None, Y_init=None, kernel=None, n_fmin_samples=1, model_type='GP',
                   n_hidden=[50, 50, 50], bo_method='LCB', batch_option='CL', batch_size=1, seed=42, util_type='se_ytrue_clip',
                   actv_func='tanh'):

        assert X_init.ndim == 2, "X_init has to be 2D array"
        assert Y_init.ndim == 2, "Y_init has to be 2D array"
        self.X_init = X_init
        self.Y_init = Y_init
        self.X = np.copy(X_init)
        self.Y = np.copy(Y_init)
        self.n_fmin_samples = n_fmin_samples
        self.bo_method = bo_method
        self.batch_option = batch_option
        self.batch_size = batch_size
        self.model_type = model_type
        self.seed = seed

        # Input dimension
        self.X_dim = self.X.shape[1]
        # Find min observed functional value and its location
        self.arg_opt = np.atleast_2d(self.X[np.argmin(self.Y)])
        self.minY = np.min(self.Y)

        # --- Param for NN models --- #
        mini_batch = 10
        T = 100
        l_s = 1e-1
        dropout_p = 0.05
        n_epochs = 1000
        activation = actv_func
        normalise_X = False
        normalise_Y = True

        # Specify the model
        if model_type == 'GP':
            self.kernel = kernel
            self.ARD   = True
            if self.noise_var > 1e-8:
                self.model = GPModel(kernel=kernel, noise_var=self.noise_var, ARD=self.ARD, seed=seed)
            else:
                self.model = GPModel(kernel=kernel, exact_feval=True, ARD=self.ARD, seed=seed)

        elif model_type == 'MCDROP':
            self.model = MCDROPWarp(mini_batch_size=mini_batch, num_epochs= n_epochs, n_units=n_hidden,
                                    dropout = dropout_p, length_scale = l_s, T = T, seed=seed, actv=activation,
                                    saving_path = self.saving_path, normalize_input=normalise_X, normalize_output=normalise_Y)
        elif model_type == 'MCCONC':
            self.model = MCCONCDROPWarp(mini_batch_size=mini_batch, num_epochs= n_epochs, n_units=n_hidden,
                                        length_scale=l_s, T = T, seed=seed, actv=activation,
                                    saving_path = self.saving_path, normalize_input=normalise_X, normalize_output=normalise_Y)
        elif model_type == 'LCBNN':
            self.model = LCBNNWarp(mini_batch_size=mini_batch,num_epochs= n_epochs, n_units=n_hidden,
                                   dropout=dropout_p,length_scale=l_s, T=T, util_type=util_type, seed=seed, actv=activation,
                                    saving_path = self.saving_path, normalize_input=normalise_X, normalize_output=normalise_Y)

        elif model_type == 'LCCD':
            self.model = LCCDWarp(mini_batch_size=mini_batch, num_epochs=n_epochs, n_units=n_hidden,
                                  length_scale=l_s, T=T, util_type=util_type, seed=seed, actv=activation,
                                    saving_path = self.saving_path, normalize_input=normalise_X, normalize_output=normalise_Y)

        elif model_type == 'DNGO':
            self.model = DNGOWrap(mini_batch_size=mini_batch,num_epochs= n_epochs, n_units=n_hidden, seed=seed,
                                 normalize_input = normalise_X, normalize_output = normalise_Y)

        elif model_type == 'BOHAM':
            self.model = BOHAMIANNWarp(num_samples=6000, keep_every=50, seed=seed,
                                       normalize_input=normalise_X, normalize_output=normalise_Y)

        # Specify acquisition function
        if self.bo_method == 'EI':
            acqu_func = EI(self.model)
        elif self.bo_method == 'LCB':
            acqu_func = LCB(self.model)
        elif self.bo_method == 'MES':
            fmin_samples = sample_fmin_Gumble(self.X, self.Y, self.model, self.bounds, nMs=self.n_fmin_samples)
            self.model.fmin_samples = fmin_samples
            acqu_func = MES(self.model)

        self.query_strategy = Acq_Optimizer(model=self.model, acqu_func=acqu_func, bounds=self.bounds,
                                            batch_size=batch_size, batch_method=batch_option)

    def iteration_step(self, iterations):

        np.random.seed(self.seed)

        X_query = np.copy(self.X)
        Y_query = np.copy(self.Y)
        X_opt  = np.copy(np.atleast_2d(self.arg_opt))
        Y_opt  = np.copy(np.atleast_2d(self.minY))
        time_record = np.zeros([iterations,2])
        self.e = np.exp(1)

        #  Fit GP model to the data
        self.model._update_model(self.X, self.Y)


        for k in range(iterations):

            # optimise the acquisition function to get the next query point and evaluate at next query point
            # x_next, max_acqu_value = optimise_acqu_func(acqu_func=acqu_func, bounds = self.bounds,X_ob=self.X)
            start_time = time.time()
            x_next_batch, acqu_value_batch = self.query_strategy.get_next(self.X, self.Y)
            max_acqu_value = np.max(acqu_value_batch)
            t_opt_acq = time.time()- start_time
            time_record[k,0] = t_opt_acq

            y_next_batch = self.func(x_next_batch) + np.random.normal(0, np.sqrt(self.noise_var), (x_next_batch.shape[0],1))

            # augment the observation data
            self.X = np.vstack((self.X, x_next_batch))
            self.Y = np.vstack((self.Y, y_next_batch))

            # file_name = self.saving_path + f"s{self.seed}_itr{k}"
            #  update GP model with new data
            start_time2 = time.time()
            self.model._update_model(self.X, self.Y)
            t_update_model = time.time()- start_time2
            time_record[k,1] =  t_update_model

            self.minY = np.min(self.Y)

            #  resample the global minimum for MES
            if self.bo_method == 'MES':
                fmin_samples = sample_fmin_Gumble(self.X, self.Y,self.model, self.bounds, nMs=self.n_fmin_samples)
                self.model.fmin_samples = fmin_samples

            # optimise the marginalised posterior mean to get the prediction for the global optimum/optimiser
            # x_opt, pos_opt = self._global_minimiser_cheap(self.pos_mean,func_gradient=self.d_pos_mean)
            # y_opt = self.func(x_opt)
            print(f'opt_aq_time={t_opt_acq};update_model_time={t_update_model}')

            #  store data
            X_query = np.vstack((X_query, np.atleast_2d(x_next_batch)))
            Y_query = np.vstack((Y_query, np.atleast_2d(y_next_batch)))
            X_opt = np.concatenate((X_opt, np.atleast_2d(X_query[np.argmin(Y_query),:])))
            Y_opt = np.concatenate((Y_opt, np.atleast_2d(min(Y_query))))

            if self.bo_method == 'MES':
                print( self.model_type + self.bo_method + self.batch_option + str(self.batch_size)+
                       ":seed:{seed},itr:{iteration},fmin_sampled:{min_sample}, x_next: {next_query_loc},"
                        "y_next:{next_query_value}, acq value: {best_acquisition_value},x_opt:{x_opt_pred},"
                       "y_opt:{y_opt_pred}"
                    .format(seed=self.seed, iteration=k,
                            min_sample = np.max(fmin_samples),
                            next_query_loc= x_next_batch[np.argmin(y_next_batch),:], next_query_value=np.min(y_next_batch),
                            best_acquisition_value=max_acqu_value,
                            x_opt_pred=X_opt[-1, :],
                            y_opt_pred=Y_opt[-1, :]
                            ))
            else:
                print( self.model_type + self.bo_method + self.batch_option + str(self.batch_size)+
                       "seed:{seed},itr:{iteration}, x_next: {next_query_loc},y_next:{next_query_value}, "
                       "acq value: {best_acquisition_value},x_opt:{x_opt_pred},y_opt:{y_opt_pred}"
                    .format(seed = self.seed,iteration=k,
                            next_query_loc=x_next_batch[np.argmin(y_next_batch),:],next_query_value=np.min(y_next_batch),
                            best_acquisition_value=max_acqu_value,
                            x_opt_pred=X_opt[-1,:],
                            y_opt_pred=Y_opt[-1,:]
                            ))

        return X_query, Y_query, X_opt, Y_opt, time_record

