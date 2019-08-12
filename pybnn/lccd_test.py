import time
import logging
import os
import torch
from torch import nn

from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np
from pybnn.base_model import BaseModel
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from pybnn.util.val_eval_metrics import val_test
from pybnn.util.loss_calibration import cal_loss,utility, heteroscedastic_loss

class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)

        out = layer(self._concrete_dropout(x, p))

        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))

        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)

        input_dimensionality = x[0].numel()  # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer
        # regularization = dropout_regularizer

        return out, regularization

    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                     - torch.log(1 - p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x


class Net(nn.Module):
    def __init__(self, n_inputs, n_units=[50, 50, 50],
                 weight_regularizer=1e-6, dropout_regularizer=1e-5, actv='tanh',
                 p_min=0.1, p_max=0.1):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(n_inputs, n_units[0])
        self.linear2 = nn.Linear(n_units[0], n_units[1])
        self.linear3 = nn.Linear(n_units[1], n_units[2])
        self.out_mu = nn.Linear(n_units[2], 1)
        self.out_logvar = nn.Linear(n_units[2], 1)

        self.conc_drop1 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer,
                                          init_min=p_min, init_max=p_max)
        self.conc_drop2 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer,
                                          init_min=p_min, init_max=p_max)
        self.conc_drop3 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer,
                                          init_min=p_min, init_max=p_max)
        self.conc_drop_mu = ConcreteDropout(weight_regularizer=weight_regularizer,
                                            dropout_regularizer=dropout_regularizer,
                                            init_min=p_min, init_max=p_max)
        self.conc_drop_logvar = ConcreteDropout(weight_regularizer=weight_regularizer,
                                                dropout_regularizer=dropout_regularizer,
                                                init_min=p_min, init_max=p_max)
        if actv == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

    def forward(self, x):

        regularization = torch.empty(4, device=x.device)
        x1 = self.activation(self.linear1(x))
        x2, regularization[0] = self.conc_drop2(x1, nn.Sequential(self.linear2, self.activation))
        x3, regularization[1] = self.conc_drop3(x2, nn.Sequential(self.linear3, self.activation))

        mean, regularization[2] = self.conc_drop_mu(x3, self.out_mu)
        log_var, regularization[3] = self.conc_drop_logvar(x3, self.out_logvar)

        return mean, log_var, regularization.sum()
        # return mean, regularization.sum()

class LCCD(BaseModel):

    def __init__(self, batch_size=10, num_epochs=500,
                 learning_rate=0.01,
                 adapt_epoch=5000, n_units_1=50, n_units_2=50, n_units_3=50,
                 length_scale = 1e-1, T = 100, mc_tau=False, regu=False,
                 normalize_input=True, normalize_output=True, seed=42,
                 loss_cal=True, lc_burn=1, util_type='se_y', gpu=True, actv='tanh',
                 init_p_min=0.1, init_p_max=0.1, model_saving_path='./'):
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

        """

        if seed is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = np.random.RandomState(seed)

        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.loss_cal = loss_cal
        self.lc_burn = lc_burn
        self.util_type = util_type
        self.X = None
        self.y = None
        self.network = None
        self.length_scale = length_scale
        self.gpu = gpu
        self.device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

        self.mc_tau = mc_tau
        self.regu = regu
        self.actv = actv

        self.p_min = init_p_min
        self.p_max = init_p_max

        self.T = T
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_learning_rate = learning_rate

        self.n_units_1 = n_units_1
        self.n_units_2 = n_units_2
        self.n_units_3 = n_units_3
        self.adapt_epoch = adapt_epoch # TODO check
        self.network = None
        self.models = []
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

        N = self.X.shape[0]
        # Check if we have enough points to create a minibatch otherwise use all data points
        if N <= self.batch_size:
            batch_size = N
        else:
            batch_size = self.batch_size

        # Create the neural network
        features = X.shape[1]
        wr = self.length_scale ** 2. / N
        dr = 2. / N

        network = Net(n_inputs=features, n_units=[self.n_units_1, self.n_units_2, self.n_units_3],
                      weight_regularizer=wr, dropout_regularizer=dr, actv=self.actv,
                      p_min=self.p_min, p_max=self.p_max)

        if itr > 0:
            model_load_path = os.path.join(self.model_saving_path,
                                              f'lccd_k={itr-1}_{self.actv}_{self.util_type}_'
                                              f'n{self.n_units_1}_e{self.num_epochs}.pt')
            network.load_state_dict(torch.load(model_load_path))  # TODO:check this

        if self.gpu:
            network = network.to(self.device)

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
            # print(epoch)
            epoch_start_time = time.time()

            train_err = 0
            train_mse = 0
            train_batches = 0

            for batch in self.iterate_minibatches(self.X, self.y,
                                                  batch_size, shuffle=True):

                inputs =  Variable(torch.FloatTensor(batch[0]))
                targets = Variable(torch.FloatTensor(batch[1]))

                if self.gpu:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                if epoch == 0 and self.loss_cal and self.lc_burn == 0:
                    h_x  = targets

                optimizer.zero_grad()
                output, log_var, regularization = network(inputs)
                # output, regularization = network(inputs)

                if self.regu:

                    if self.loss_cal and epoch >= self.lc_burn:
                        loss, mse_loss, log_condi_gain = cal_loss(targets, output, util, h_x,
                                                                  y_pred_samples, log_var,
                                                                  regularization=regularization)

                        for i in range(len(batch[0])):
                            batch_point_idx = np.where(self.y == batch[1][i])
                            y_utils[batch_point_idx] += log_condi_gain.cpu().data.numpy()[i]
                            y_utils_counts[batch_point_idx] += 1

                    else:
                        loss = heteroscedastic_loss(targets, output, log_var) + regularization
                        mse_loss = torch.zeros_like(loss)

                else:

                    if self.loss_cal and epoch >= self.lc_burn:
                        loss, mse_loss, log_condi_gain = cal_loss(targets, output, util, h_x,
                                                                  y_pred_samples, log_var,
                                                                  regularization=None)
                        for i in range(len(batch[0])):
                            batch_point_idx = np.where(self.y == batch[1][i])
                            y_utils[batch_point_idx] += log_condi_gain.cpu().data.numpy()[i]
                            y_utils_counts[batch_point_idx] += 1

                    else:
                        loss = heteroscedastic_loss(targets, output, log_var)
                        mse_loss = torch.zeros_like(loss)

                loss.backward(retain_graph=True)
                optimizer.step()

                train_err += loss.cpu().data.numpy()
                train_mse += mse_loss.cpu().data.numpy()
                train_batches += 1

            if self.loss_cal and epoch >= (self.lc_burn - 1):
                mc_samples = [network(inputs) for _ in range(20)]
                y_pred_samples = torch.stack([tup[0] for tup in mc_samples])

                if self.util_type == 'se_prod_y' or self.util_type == 'se_prod_yclip':
                    numerator = torch.sum(y_pred_samples * torch.exp(y_pred_samples),0)
                    denominator = torch.sum(torch.exp(y_pred_samples),0)
                    h_x = numerator / denominator
                elif self.util_type == 'linear_se_ysample_clip' or self.util_type == 'se_ysample_clip':
                    h_x = targets
                else:
                    y_pred_mean = torch.mean(y_pred_samples, 0)
                    h_x = y_pred_mean

            mean_train_mse = (train_mse / train_batches)
            mean_train_loss = (train_err / train_batches)
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

        self.model = network
        self.lc = lc

        # Saving models
        model_save_path = os.path.join(self.model_saving_path,
                                         f'lccd_k={itr}_{self.actv}_{self.util_type}_'
                                         f'n{self.n_units_1}_e{self.num_epochs}.pt')
        torch.save(network.state_dict(), model_save_path)  # TODO:check this
        print('lccd model saved')

        # Estimate aleatoric uncertainty
        # X_train_tensor = Variable(torch.FloatTensor(self.X))
        # if self.gpu:
        #     X_train_tensor = X_train_tensor.to(self.device)
        # y_train_mc_samples = [network(X_train_tensor) for _ in range(self.T)]
        # y_train_predict_samples = torch.stack([tup[0] for tup in y_train_mc_samples]).view(self.T, N).cpu().data.numpy()
        # self.aleatoric_uncertainty = np.mean(np.mean((y_train_predict_samples - self.y.flatten())**2, 0))

        y_utils_average = y_utils / y_utils_counts
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

        # MC_samples : list T x N x 1
        # start_mc=time.time()
        gpu_test = False
        if gpu_test:
            X_tensor = Variable(torch.FloatTensor(X_)).to(self.device)
            MC_samples = [model(X_tensor) for _ in range(T)]
            means = torch.stack([tup[0] for tup in MC_samples]).view(T, X_.shape[0]).cpu().data.numpy()
            logvar = torch.stack([tup[1] for tup in MC_samples]).view(T, X_.shape[0]).cpu().data.numpy()
        else:
            model.cpu()
            MC_samples = [model(Variable(torch.FloatTensor(X_))) for _ in range(T)]
            means = torch.stack([tup[0] for tup in MC_samples]).view(T, X_.shape[0]).data.numpy()
            logvar = torch.stack([tup[1] for tup in MC_samples]).view(T, X_.shape[0]).data.numpy()


        # mc_time = time.time() - start_mc
        # print(f'mc_time={mc_time}')
        # compute uncertainties: aleatoric and epistemic (means_var)
        aleatoric_uncertainty = np.exp(logvar).mean(0)
        MC_pred_mean = np.mean(means, 0)  # N x 1
        means_var  = np.var(means, 0)
        MC_pred_var = means_var + aleatoric_uncertainty
        m = MC_pred_mean.flatten()


        if MC_pred_var.shape[0] == 1:
            e_v = np.clip(means_var, np.finfo(means_var.dtype).eps, np.inf)
            a_v = np.clip(aleatoric_uncertainty, np.finfo(aleatoric_uncertainty.dtype).eps, np.inf)
            v = np.clip(MC_pred_var, np.finfo(MC_pred_var.dtype).eps, np.inf)
        else:
            e_v = np.clip(means_var, np.finfo(means_var.dtype).eps, np.inf)
            a_v = np.clip(aleatoric_uncertainty, np.finfo(aleatoric_uncertainty.dtype).eps, np.inf)
            e_v[np.where((e_v < np.finfo(e_v.dtype).eps) & (e_v > -np.finfo(e_v.dtype).eps))] = 0
            a_v[np.where((a_v < np.finfo(a_v.dtype).eps) & (a_v > -np.finfo(a_v.dtype).eps))] = 0

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
        As well as the test log likelihood per point and rmse on test data

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N input test points
        Y_test: np.ndarray (N, D)

        Returns
        ----------
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

        # MC_samples : list T x N x 1
        # start_mc=time.time()
        gpu_test = False
        if gpu_test:
            X_tensor = Variable(torch.FloatTensor(X_)).to(self.device)
            MC_samples = [model(X_tensor) for _ in range(T)]
            means = torch.stack([tup[0] for tup in MC_samples]).view(T, X_.shape[0]).cpu().data.numpy()
            logvar = torch.stack([tup[1] for tup in MC_samples]).view(T, X_.shape[0]).cpu().data.numpy()
        else:
            model.cpu()
            MC_samples = [model(Variable(torch.FloatTensor(X_))) for _ in range(T)]
            means = torch.stack([tup[0] for tup in MC_samples]).view(T, X_.shape[0]).data.numpy()
            logvar = torch.stack([tup[1] for tup in MC_samples]).view(T, X_.shape[0]).data.numpy()


        # mc_time = time.time() - start_mc
        # print(f'mc_time={mc_time}')

        # compute uncertainties: aleatoric and epistemic (means_var)
        aleatoric_uncertainty = np.exp(logvar).mean(0)
        MC_pred_mean = np.mean(means, 0)
        means_var  = np.var(means, 0)
        MC_pred_var = means_var + aleatoric_uncertainty
        m = MC_pred_mean.flatten()


        if MC_pred_var.shape[0] == 1:
            e_v = np.clip(means_var, np.finfo(means_var.dtype).eps, np.inf)
            a_v = np.clip(aleatoric_uncertainty, np.finfo(aleatoric_uncertainty.dtype).eps, np.inf)
            v = np.clip(MC_pred_var, np.finfo(MC_pred_var.dtype).eps, np.inf)
        else:
            e_v = np.clip(means_var, np.finfo(means_var.dtype).eps, np.inf)
            a_v = np.clip(aleatoric_uncertainty, np.finfo(aleatoric_uncertainty.dtype).eps, np.inf)
            e_v[np.where((e_v < np.finfo(e_v.dtype).eps) & (e_v > -np.finfo(e_v.dtype).eps))] = 0
            a_v[np.where((a_v < np.finfo(a_v.dtype).eps) & (a_v > -np.finfo(a_v.dtype).eps))] = 0

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
        ppp, rmse = val_test(Y_test, T, means, logvar)

        return m, v, e_v, a_v, ppp, rmse
