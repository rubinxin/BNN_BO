import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
from pybnn.base_model import BaseModel
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization


class Net(nn.Module):
    def __init__(self, n_inputs, dropout_p, decay, n_units=[50, 50, 50], actv='tanh'):
        super(Net, self).__init__()
        self.decay = decay
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.n_units = n_units
        self.fc1 = nn.Linear(n_inputs, n_units[0])
        self.fc_hidden = []
        for i in range(len(self.n_units) - 1):
            self.fc_hidden.append(nn.Linear(n_units[i], n_units[i+1]))
        self.out = nn.Linear(n_units[-1], 1)
        if actv == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()


    def forward(self, x):

        x = self.activation(self.fc1(x))

        for i in range(len(self.n_units)-1):
            x = self.dropout(x)
            x = self.activation(self.fc_hidden[i](x))

        x = self.dropout(x)
        return self.out(x)


class MCDROP(BaseModel):

    def __init__(self, batch_size=10, num_epochs=500,
                 learning_rate=0.01,
                 adapt_epoch=5000, n_units=[50, 50, 50],
                 dropout_p = 0.05, length_scale = 1e-1, weight_decay = 1e-6, T = 100,
                 normalize_input=True, normalize_output=True, rng=None, gpu=True, actv='tanh'):
        """
        This module performs MC Dropout for a fully connected
        feed forward neural network.

        Parameters
        ----------
        batch_size: int
            Batch size for training the neural network
        num_epochs: int
            Number of epochs for training
        learning_rate: float
            Initial learning rate for Adam
        adapt_epoch: int
            Defines after how many epochs the learning rate will be decayed by a factor 10
        n_units_1: int
            Number of units in layer 1
        n_units_2: int
            Number of units in layer 2
        n_units_3: int
            Number of units in layer 3
        dropoyt: float
            Dropout rate for all the dropout layers in the network.
        tau: float
            Tau value used for regularization
        T : int
            Number of samples for forward passes
        normalize_output : bool
            Zero mean unit variance normalization of the output values
        normalize_input : bool
            Zero mean unit variance normalization of the input values
        rng: np.random.RandomState
            Random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = np.random.RandomState(rng)

        self.seed = rng
        torch.manual_seed(self.seed)

        self.X = None
        self.y = None
        self.network = None
        # self.tau = tau
        self.decay = weight_decay
        self.length_scale = length_scale
        self.dropout_p = dropout_p
        self.T = T
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.gpu = gpu
        self.actv = actv

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_learning_rate = learning_rate

        self.n_units_list = n_units

        self.adapt_epoch = adapt_epoch # TODO check
        self.network = None
        self.models = []

        # Use GPU
        self.device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

    @BaseModel._check_shapes_train
    def train(self, X, y, itr=0, saving_path = None):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.

        """

        start_time = time.time()

        # Normalize inputs
        if self.normalize_input:
            self.X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
        else:
            self.X = X

        # Normalize ouputs
        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)
        else:
            self.y = y

        self.y = self.y[:, None]

        # Check if we have enough points to create a minibatch otherwise use all data points
        if self.X.shape[0] <= self.batch_size:
            batch_size = self.X.shape[0]
        else:
            batch_size = self.batch_size

        # Create the neural network
        features = X.shape[1]

        network = Net(n_inputs=features, dropout_p=self.dropout_p, decay=self.decay,
                      n_units=self.n_units_list, actv=self.actv)

        if self.gpu:
            network = network.to(self.device)
        #
        # optimizer = optim.Adam(network.parameters(),
        #                        lr=self.init_learning_rate, weight_decay=network.decay)
        optimizer = optim.Adam(network.parameters(),
                               lr=self.init_learning_rate)

        if itr > 0:
            model_loading_path = os.path.join(saving_path,
                                              f'mcdrop_k={itr-1}_{self.actv}_e{self.num_epochs}.pt')
            network.load_state_dict(torch.load(model_loading_path))
            # network.eval()
            # network.train()

        # Start training
        network.train()
        lc = np.zeros([self.num_epochs])
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()

            train_err = 0
            train_batches = 0

            for batch in self.iterate_minibatches(self.X, self.y,
                                                  batch_size, shuffle=True):

                inputs = Variable(torch.Tensor(batch[0]))
                targets = Variable(torch.Tensor(batch[1]))

                if self.gpu:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                optimizer.zero_grad()
                output = network(inputs)
                loss = torch.nn.functional.mse_loss(output, targets)
                loss.backward()
                optimizer.step()

                train_err += loss.cpu().data.numpy()
                train_batches += 1

            lc[epoch] = train_err / train_batches
            logging.debug("Epoch {} of {}".format(epoch + 1, self.num_epochs))
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time
            logging.debug("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
            logging.debug("Training loss:\t\t{:.5g}".format(train_err / train_batches))

        self.model = network
        # Saving models
        model_saving_path = os.path.join(saving_path,
                                         f'mcdrop_k={itr}_{self.actv}_e{self.num_epochs}.pt')
        torch.save(network.state_dict(), model_saving_path)
        print('mcdrop model saved')

        train_mse_loss_all_epoch = np.array(lc[:])

        return train_mse_loss_all_epoch

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0], \
            "The number of training points is not the same"
        if shuffle:
            indices = np.arange(inputs.shape[0])
            self.rng.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    @BaseModel._check_shapes_predict
    def predict(self, X_test):
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N input test points

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance

        """

        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Perform MC dropout
        model = self.model
        model.eval()
        T     = self.T


        # start_mc=time.time()
        # Yt_hat: T x N x 1
        if self.gpu:
            model.cpu()
            # Yt_hat = np.array([model(Variable(torch.Tensor(X_))).data.numpy() for _ in range(T)])
            Yt_hat = np.hstack([model(Variable(torch.Tensor(X_[:, np.newaxis]))).data.numpy() for _ in range(T)])
            # X_tensor = X_tensor.to(self.device)
            # Yt_hat = np.hstack([model(Variable(torch.Tensor(X_[:, np.newaxis]))).cpu().data.numpy() for i in range(T)])
        else:
            # Yt_hat = np.array([model(Variable(torch.Tensor(X_))).data.numpy() for _ in range(T)])
            Yt_hat = np.hstack([model(Variable(torch.Tensor(X_[:, np.newaxis]))).data.numpy() for _ in range(T)])

        # mc_time = time.time() - start_mc
        # print(f'mc_time={mc_time}')

        tau = self.length_scale**2 * (1.0 - self.model.dropout_p) / (2. * self.model.decay * self.X.shape[0])
        # MC_pred_mean = np.mean(Yt_hat, 0)  # N x 1
        # Second_moment = np.mean(Yt_hat ** 2, 0) # N x 1
        # MC_pred_var = Second_moment + 1./ tau - (MC_pred_mean ** 2)

        MC_pred_mean = Yt_hat.mean(axis=1)
        MC_pred_var = Yt_hat.var(axis=1) + 1./tau

        m = MC_pred_mean

        if MC_pred_var.shape[0] == 1:
            v = np.clip(MC_pred_var, np.finfo(MC_pred_var.dtype).eps, np.inf)
        else:
            v = np.clip(MC_pred_var, np.finfo(MC_pred_var.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        if self.normalize_output:
            m = zero_mean_unit_var_denormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2

        m = m.flatten()
        v = v.flatten()

        return m, v

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        Returns
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """

        inc, inc_value = super(MCDROP, self).get_incumbent()
        if self.normalize_input:
            inc = zero_mean_unit_var_denormalization(inc, self.X_mean, self.X_std)

        if self.normalize_output:
            inc_value = zero_mean_unit_var_denormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value
