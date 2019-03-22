# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings
warnings.filterwarnings("ignore")

import math
from scipy.misc import logsumexp
import numpy as np

from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense
from keras import Model

class DropoutNet:

    def __init__(self, n_epochs = 40, dropout=0.3, T = 1000, n_hidden=[50, 50, 50], normalize=False,
                 tau=1.0, batch_size=128):
        """
        :param n_epochs: Numer of epochs for which to train the network.
        :param dropout: Dropout rate for all the dropout layers in the network.
        :param T: Number of MC samples(Forward Passes) to estimate uncertainty.
        :param n_hidden: A list with the number of neurons for each hidden layer.
        :param normalize: Whether to normalize the input features.
        :param tau: Tau value used for regularization
        """
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.T = T
        self.n_hidden = n_hidden
        self.normalize = normalize
        self.tau = tau
        self.batch_size = batch_size
        self.model = None

    def _create_model(self, X_train, Y_train):
        """
        :param X_train: N x d
        :param Y_train: N x 1
        :return:
        """
        if self.normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[ self.std_X_train == 0 ] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[ 1 ])
            self.mean_X_train = np.zeros(X_train.shape[ 1 ])

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
                  np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(Y_train)
        self.std_y_train = np.std(Y_train)

        Y_train_normalized = (Y_train - self.mean_y_train) / self.std_y_train
        # Y_train_normalized = np.array(Y_train_normalized, ndmin=2).T

        # We construct the network
        N = X_train.shape[0]
        lengthscale = 1e-2
        reg = lengthscale ** 2 * (1 - self.dropout) / (2. * N * self.tau)

        inputs = Input(shape=(X_train.shape[1],))
        inter = Dropout(self.dropout)(inputs, training=True)
        inter = Dense(self.n_hidden[0], activation='relu', W_regularizer=l2(reg))(inter)
        for i in range(len(self.n_hidden) - 1):
            inter = Dropout(self.dropout)(inter, training=True)
            inter = Dense(self.n_hidden[i + 1], activation='relu', W_regularizer=l2(reg))(inter)
        inter = Dropout(self.dropout)(inter, training=True)
        outputs = Dense(Y_train_normalized.shape[1], W_regularizer=l2(reg))(inter)
        model = Model(inputs, outputs)

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, Y_train_normalized, batch_size=self.batch_size, nb_epoch=self.n_epochs, verbose=0)
        self.model = model

    def _update_model(self,  X_all, Y_all):
        """
        Updates the model with new observations.
        """
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            # We normalize the test set
            X_all = (X_all - np.full(X_all.shape, self.mean_X_train)) / \
                     np.full(X_all.shape, self.std_X_train)
            Y_all_normalised = (Y_all - self.mean_y_train) / self.std_y_train
            model = self.model
            model.fit(X_all, Y_all_normalised, batch_size=self.batch_size, nb_epoch=self.n_epochs, verbose=0)
            self.model = model

    def predict(self, X_test, test_batch_size = 50):
        """
        Predictions with the model. Returns posterior means and standard deviations at X.
        """
        X_test = np.atleast_2d(X_test)

        # We normalize the test set
        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
                 np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data
        model = self.model
        # --------- Dropout VI Part (to get uncertainty measure) ------------------ #
        T = self.T
        Yt_hat = np.array([model.predict(X_test, batch_size=test_batch_size, verbose=0) for _ in range(T)])
        Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train # T x N TODO check with Adam

        MC_pred_mean = np.mean(Yt_hat, 0) #  N x 1
        Second_moment = np.mean(Yt_hat**2, 0)

        MC_pred_var = Second_moment + np.eye(Yt_hat.shape[-1])/self.tau - ( MC_pred_mean**2 )
        MC_pred_var = np.clip(MC_pred_var, 1e-10, np.inf)
        MC_pred_std = np.sqrt(MC_pred_var)
        return MC_pred_mean, MC_pred_std

    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        return 'Not implemented'