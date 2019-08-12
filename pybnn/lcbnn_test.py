import time
import logging
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
from pybnn.base_model import BaseModel
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from pybnn.util.loss_calibration import cal_loss, utility
from pybnn.util.val_eval_metrics import val_test


def cal_loss(y_true, y_pred, util, H_x, y_pred_samples):
    a = 1.0
    mse_loss = nn.functional.mse_loss(y_pred, y_true)

    log_condi_gain = util(y_pred_samples, H_x, y_true)

    utility_value = a * log_condi_gain.mean()
    calibrated_loss = mse_loss - utility_value

    return calibrated_loss, mse_loss, log_condi_gain

def optimal_h(y_pred_samples, util):
    T, N, D = y_pred_samples.shape
    G_t = torch.zeros((N, D))
    for t in range(T):
        dec = torch.zeros((D, D, N))
        for d in range(D):
            dec[d, d] = torch.ones((N))
        G_t += util(dec, y_pred_samples[t])
    I = torch.eye(D)
    H_x = I[G_t.argmax(1)]
    return H_x, G_t

class Net(nn.Module):
    def __init__(self, n_inputs, dropout_p, decay, n_units=[50, 50, 50], actv='tanh'):
        super(Net, self).__init__()
        self.decay = decay
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.fc1 = nn.Linear(n_inputs, n_units[0])
        self.fc2 = nn.Linear(n_units[0], n_units[1])
        self.fc3 = nn.Linear(n_units[1], n_units[2])
        self.out = nn.Linear(n_units[2], 1)
        if actv == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

    def forward(self, x):

        x = self.activation(self.fc1(x))
        x = self.dropout(x)

        x = self.activation(self.fc2(x))
        x = self.dropout(x)

        x = self.activation(self.fc3(x))
        x = self.dropout(x)

        return self.out(x)


class LCBNN(BaseModel):

    def __init__(self, batch_size=10, num_epochs=500,
                 learning_rate=0.01, adapt_epoch=5000, n_units_1=50, n_units_2=50, n_units_3=50,
                 dropout_p=0.05, length_scale = 1e-1, weight_decay = 1e-6, T = 100,
                 normalize_input=True, normalize_output=True, seed=None, weights=None,
                 loss_cal=True, lc_burn=1, util_type='se_ytrue_clip',gpu=True, actv='tanh',
                 model_saving_path='./'):
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
        rng: random seed
            Random number generator
        """

        if seed is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = np.random.RandomState(seed)

        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)

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
        self.actv = actv

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_learning_rate = learning_rate

        self.n_units_1 = n_units_1
        self.n_units_2 = n_units_2
        self.n_units_3 = n_units_3
        self.adapt_epoch = adapt_epoch  # TODO check
        self.network = None
        self.models = []
        self.weights = weights
        self.loss_cal = loss_cal
        self.lc_burn = lc_burn
        self.util_type = util_type
        # Use GPU
        self.gpu = gpu
        self.device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
        self.model_saving_path = model_saving_path

    @BaseModel._check_shapes_train
    def train(self, X, y, itr=0):
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
        self.y_min = np.min(self.y)

        # print(f'mean_y_train={np.mean(self.y)}')
        # Check if we have enough points to create a minibatch otherwise use all data points
        if self.X.shape[0] <= self.batch_size:
            batch_size = self.X.shape[0]
        else:
            batch_size = self.batch_size

        # Create the neural network
        features = X.shape[1]

        network = Net(n_inputs=features, dropout_p=self.dropout_p, decay= self.decay,
                      n_units=[self.n_units_1, self.n_units_2, self.n_units_3], actv=self.actv)

        if itr > 0:
            model_load_path = os.path.join(self.model_saving_path,
                                              f'lcbnn_k={itr-1}_{self.actv}_{self.util_type}_'
                                              f'n{self.n_units_1}_e{self.num_epochs}.pt')
            network.load_state_dict(torch.load(model_load_path))  # TODO:check this

        if self.gpu:
            network = network.to(self.device)

        # optimizer = optim.Adam(network.parameters(),
        #                        lr=self.init_learning_rate,
        #                        weight_decay=self.decay)
        optimizer = optim.Adam(network.parameters(),
                               lr=self.init_learning_rate)


        # Start training
        lc = np.zeros([self.num_epochs])
        if self.loss_cal:
            util = utility(util_type=self.util_type, Y_train=self.y)

        train_mse_loss_all_epoch = []
        train_logutil_all_epoch = []
        y_utils = np.zeros_like(self.y)
        y_utils_counts = np.zeros_like(self.y)

        network.train()

        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()

            train_err = 0
            train_mse = 0
            train_batches = 0

            for batch in self.iterate_minibatches(self.X, self.y,
                                                  batch_size, shuffle=True):
                inputs = Variable(torch.Tensor(batch[0]))
                targets = Variable(torch.Tensor(batch[1]))

                if self.gpu:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                if epoch == 0 and self.loss_cal and self.lc_burn == 0:
                    h_x  = targets

                optimizer.zero_grad()
                output = network(inputs)

                if self.loss_cal and epoch >= self.lc_burn:
                    loss, mse_loss, log_condi_gain = cal_loss(targets, output, util, h_x, y_pred_samples)

                    for i in range(len(batch[0])):
                        batch_point_idx = np.where(self.y == batch[1][i])
                        y_utils[batch_point_idx] += log_condi_gain.cpu().data.numpy()[i]
                        y_utils_counts[batch_point_idx] += 1

                else:
                    loss =  torch.nn.functional.mse_loss(output, targets)
                    mse_loss = torch.zeros_like(loss)

                loss.backward(retain_graph=True)
                optimizer.step()

                train_err += loss.cpu().data.numpy()
                train_mse += mse_loss.cpu().data.numpy()
                train_batches += 1

                if self.loss_cal and epoch >= (self.lc_burn - 1):
                    y_pred_samples = [network(inputs) for _ in range(10)]
                    y_pred_samples = torch.stack(y_pred_samples)

                    if self.util_type == 'se_prod_y' or self.util_type == 'se_prod_yclip':
                        numerator = torch.sum(y_pred_samples * torch.exp(y_pred_samples),0)
                        denominator = torch.sum(torch.exp(y_pred_samples),0)
                        h_x = numerator / denominator
                    elif self.util_type == 'linear_se_ysample_clip' or self.util_type == 'se_ysample_clip':
                        h_x = targets
                    else:
                        y_pred_mean = torch.mean(y_pred_samples, 0)
                        h_x = y_pred_mean

            mean_train_mse = (train_mse/train_batches)
            mean_train_loss = (train_err/train_batches)
            train_mse_loss_all_epoch.append(mean_train_mse)
            training_logutil_np = mean_train_mse - mean_train_loss
            train_logutil_all_epoch.append(training_logutil_np)
            lc[epoch] = train_err / train_batches
            logging.debug("Epoch {} of {}".format(epoch + 1, self.num_epochs))
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time
            logging.debug("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
            logging.debug("Training loss:\t\t{:.5g}".format(train_err / train_batches))
            # if epoch % 20 == 0:
            #     print(f'epoch={epoch}:cal_loss={loss}')

        self.model = network

        # Saving models

        model_save_path = os.path.join(self.model_saving_path,
                                              f'lcbnn_k={itr}_{self.actv}_{self.util_type}_'
                                              f'n{self.n_units_1}_e{self.num_epochs}.pt')
        torch.save(network.state_dict(), model_save_path)  # TODO:check this
        print('lcbnn model saved')

        self.lc = lc
        y_utils_average          = y_utils / y_utils_counts
        train_mse_loss_all_epoch = np.array(train_mse_loss_all_epoch[2:])
        train_logutil_all_epoch = np.array(train_logutil_all_epoch[2:])

        return train_mse_loss_all_epoch, train_logutil_all_epoch, y_utils_average

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
    def predict(self, X_test, full_sample=False):
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
        # torch.manual_seed(self.seed)

        # Normalize inputs

        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Perform MC dropout
        model = self.model
        model.train()
        T = self.T

        # start_mc=time.time()
        # Yt_hat: T x N x 1
        gpu_test = False
        if gpu_test:
            X_tensor = Variable(torch.Tensor(X_[:, np.newaxis])).to(self.device)
            Yt_hat = torch.stack([model(X_tensor) for _ in range(T)]).view(T, X_.shape[0]).cpu().data.numpy()
        else:
            model.cpu()
            X_tensor = Variable(torch.Tensor(X_[:, np.newaxis]))
            Yt_hat = torch.stack([model(X_tensor) for _ in range(T)]).view(T, X_.shape[0]).data.numpy()

        # mc_time = time.time() - start_mc
        # print(f'mc_time={mc_time}')

        if full_sample:
            return Yt_hat
        else:
            tau = self.length_scale ** 2 * (1.0 - self.model.dropout_p) / (2. * self.model.decay * self.X.shape[0])
            aleatoric_uncertainty = 1. / tau
            epistemic_uncertainty = Yt_hat.var(axis=0)
            MC_pred_mean = Yt_hat.mean(axis=0)
            MC_pred_var = epistemic_uncertainty + aleatoric_uncertainty

            # MC_pred_mean = np.mean(Yt_hat, 0)  # N x 1
            # Second_moment = np.mean(Yt_hat ** 2, 0)  # N x 1
            # MC_pred_var = Second_moment - (MC_pred_mean ** 2)  + 1./tau

            m = MC_pred_mean.flatten()  # .flatten()

            if MC_pred_var.shape[0] == 1:
                v = np.clip(MC_pred_var, np.finfo(MC_pred_var.dtype).eps, np.inf)
                e_v = np.clip(epistemic_uncertainty, np.finfo(epistemic_uncertainty.dtype).eps, np.inf)
                a_v = np.clip(aleatoric_uncertainty, sys.float_info.epsilon, np.inf)
            else:
                e_v = np.clip(epistemic_uncertainty, np.finfo(epistemic_uncertainty.dtype).eps, np.inf)
                a_v = np.clip(aleatoric_uncertainty, sys.float_info.epsilon, np.inf)
                e_v[np.where((e_v < np.finfo(e_v.dtype).eps) & (e_v > -np.finfo(e_v.dtype).eps))] = 0

                v = np.clip(MC_pred_var, np.finfo(MC_pred_var.dtype).eps, np.inf)
                v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

            if self.normalize_output:
                m = zero_mean_unit_var_denormalization(m, self.y_mean, self.y_std)
                v *= self.y_std ** 2

            m = m.flatten()
            v = v.flatten()
            e_v = e_v.flatten()
            a_v = a_v.flatten()

            return m, v

    @BaseModel._check_shapes_predict
    def validate(self, X_test, Y_test):
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
        # torch.manual_seed(self.seed)

        # Normalize inputs

        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Perform MC dropout
        model = self.model
        model.train()
        T = self.T

        # start_mc=time.time()
        # Yt_hat: T x N x 1
        gpu_test = False
        if gpu_test:
            X_tensor = Variable(torch.Tensor(X_[:, np.newaxis])).to(self.device)
            Yt_hat = torch.stack([model(X_tensor) for _ in range(T)]).view(T, X_.shape[0]).cpu().data.numpy()
        else:
            model.cpu()
            X_tensor = Variable(torch.Tensor(X_[:, np.newaxis]))
            Yt_hat = torch.stack([model(X_tensor) for _ in range(T)]).view(T, X_.shape[0]).data.numpy()
        # mc_time = time.time() - start_mc
        # print(f'mc_time={mc_time}')


        tau = self.length_scale ** 2 * (1.0 - self.model.dropout_p) / (2. * self.model.decay * self.X.shape[0])
        aleatoric_uncertainty = 1. / tau
        epistemic_uncertainty = Yt_hat.var(axis=0)
        MC_pred_mean = Yt_hat.mean(axis=0)
        MC_pred_var = epistemic_uncertainty + aleatoric_uncertainty

        # MC_pred_mean = np.mean(Yt_hat, 0)  # N x 1
        # Second_moment = np.mean(Yt_hat ** 2, 0)  # N x 1
        # MC_pred_var = Second_moment - (MC_pred_mean ** 2)  + 1./tau

        m = MC_pred_mean.flatten()  # .flatten()

        if MC_pred_var.shape[0] == 1:
            v = np.clip(MC_pred_var, np.finfo(MC_pred_var.dtype).eps, np.inf)
            e_v = np.clip(epistemic_uncertainty, np.finfo(epistemic_uncertainty.dtype).eps, np.inf)
            a_v = np.clip(aleatoric_uncertainty, sys.float_info.epsilon, np.inf)
        else:
            e_v = np.clip(epistemic_uncertainty, np.finfo(epistemic_uncertainty.dtype).eps, np.inf)
            a_v = np.clip(aleatoric_uncertainty, sys.float_info.epsilon, np.inf)
            e_v[np.where((e_v < np.finfo(e_v.dtype).eps) & (e_v > -np.finfo(e_v.dtype).eps))] = 0

            v = np.clip(MC_pred_var, np.finfo(MC_pred_var.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        if self.normalize_output:
            m = zero_mean_unit_var_denormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2

        m = m.flatten()
        v = v.flatten()
        e_v = e_v.flatten()
        a_v = a_v.flatten()

        # validation performance evaluation:
        logvar = np.log(aleatoric_uncertainty)
        means = Yt_hat
        ppp, rmse = val_test(Y_test, T, means, logvar)

        rmse2 = np.mean((MC_pred_mean - Y_test.squeeze()) ** 2.) ** 0.5

        return m, v, e_v, a_v, ppp, rmse

