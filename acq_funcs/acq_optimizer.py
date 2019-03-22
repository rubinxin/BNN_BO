#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017
@author: robin
"""
import numpy as np
from utilities.utilities import optimise_acqu_func

""""All acquisition functions are to be maximised"""

class Acq_Optimizer(object):
    """
    Optimizer Acquisition Functions
    """

    def __init__(self, model, acqu_func, bounds, batch_size=1, batch_method='CL'):
        self.model = model
        self.acqu_func = acqu_func
        self.batch_size = batch_size
        self.batch_method = batch_method
        self.bounds = bounds

    def get_next(self,  X, Y):
        """
        Computes the next batch points
        return:
        x_next_batch: B x d
        max_acqu_value: B x 1
        """

        # --- Kriging Believer ------ #
        if self.batch_method.upper() == 'KB':
            X_batch, batch_acq_value = optimise_acqu_func(acqu_func=self.acqu_func, bounds=self.bounds, X_ob=X)
            new_batch_point = X_batch

            temporal_X = np.copy(X)
            temporal_Y = np.copy(Y)
            # get the remaining points in the batch
            k = 1
            while self.batch_size > 1:

                # believe the predictor: the functional value at last query location is equal to its predicitve mean
                mu_new_batch_point, _ = self.model.predict(new_batch_point)

                # augment the observed data with previous query location and the preditive mean at that location
                temporal_X = np.vstack((temporal_X, new_batch_point))
                temporal_Y = np.vstack((temporal_Y, mu_new_batch_point))

                # update the surrogate model (no update on hyperparameter) and acq_func with the augmented observation data
                self.model._update_model(temporal_X, temporal_Y)

                new_batch_point, next_batch_acq_value = optimise_acqu_func(acqu_func=self.acqu_func,
                                                                           bounds=self.bounds, X_ob=X)

                X_batch = np.vstack((X_batch, new_batch_point))
                batch_acq_value = np.append(batch_acq_value, next_batch_acq_value)
                k += 1

        elif self.batch_method.upper() == 'CL':
            X_batch, batch_acq_value = optimise_acqu_func(acqu_func=self.acqu_func, bounds=self.bounds, X_ob=X)
            new_batch_point = X_batch

            temporal_X = np.copy(X)
            temporal_Y = np.copy(Y)

            L = np.min(temporal_Y)
            # get the remaining points in the batch
            k = 1

            while k < self.batch_size:
                # assume the functional value at last query location is equal to a past observation
                # augment the observed data with previous query location and the preditive mean at that location
                temporal_X = np.vstack((temporal_X, new_batch_point))
                temporal_Y = np.vstack((temporal_Y, L))

                # update the surrogate model (no update on hyperparameter) and acq_func with the augmented observation data
                self.model._update_model(temporal_X, temporal_Y)

                new_batch_point, next_batch_acq_value = optimise_acqu_func(acqu_func=self.acqu_func,
                                                                           bounds=self.bounds, X_ob=X)

                X_batch = np.vstack((X_batch, new_batch_point))
                batch_acq_value = np.vstack((batch_acq_value, next_batch_acq_value))
                k += 1

        # --- Thompson Sampling ------ #
        elif self.batch_method.upper() == 'TS ':
            print('Not implemented')

        return X_batch, batch_acq_value
