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

# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)


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

def heteroscedastic_loss(true, mean, log_var):
    precision = torch.exp(-log_var)
    return torch.mean(torch.sum(precision * (true - mean)**2 + log_var, 1), 0)

class Net(nn.Module):
    def __init__(self, n_inputs, n_units=[50, 50, 50],
                 weight_regularizer=1e-6, dropout_regularizer=1e-5, actv='tanh'):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(n_inputs, n_units[0])
        self.linear2 = nn.Linear(n_units[0], n_units[1])
        self.linear3 = nn.Linear(n_units[1], n_units[2])
        self.out_mu = nn.Linear(n_units[2], 1)
        self.out_logvar = nn.Linear(n_units[2], 1)

        self.conc_drop1 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop3 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop_mu = ConcreteDropout(weight_regularizer=weight_regularizer,
                                            dropout_regularizer=dropout_regularizer)
        self.conc_drop_logvar = ConcreteDropout(weight_regularizer=weight_regularizer,
                                                dropout_regularizer=dropout_regularizer)

        if actv == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

    def forward(self, x):

        regularization = torch.empty(4, device=x.device)
        x1 = self.activation(self.linear1(x))
        # x1, regularization[0] = self.conc_drop1(x, nn.Sequential(self.linear1, self.activation))
        x2, regularization[0] = self.conc_drop2(x1, nn.Sequential(self.linear2, self.activation))

        x3, regularization[1] = self.conc_drop3(x2, nn.Sequential(self.linear3, self.activation))

        mean, regularization[2] = self.conc_drop_mu(x3, self.out_mu)
        log_var, regularization[3] = self.conc_drop_logvar(x3, self.out_logvar)

        return mean, log_var, regularization.sum()
    #     # return mean, regularization.sum()

class MCCONCRETEDROP(BaseModel):

    def __init__(self, batch_size=10, num_epochs=500,
                 learning_rate=0.01,
                 adapt_epoch=5000, n_units_1=50, n_units_2=50, n_units_3=50,
                 length_scale = 1e-4, T = 100, regu = False, mc_tau=False,
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
        rng: random seed

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
        self.length_scale = length_scale
        self.gpu = gpu
        self.device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
        self.regu = regu
        self.mc_tau = mc_tau
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
        self.adapt_epoch = adapt_epoch # TODO check
        self.network = None
        self.models = []

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
                      weight_regularizer=wr, dropout_regularizer=dr,  actv=self.actv)
        optimizer = optim.Adam(network.parameters(),
                               lr=self.init_learning_rate)

        if itr > 0:
            model_loading_path = os.path.join(saving_path,
                                              f'concdrop_k={itr-1}_{self.actv}_n{self.n_units_1}_e{self.num_epochs}.pt')
            network.load_state_dict(torch.load(model_loading_path))

        if self.gpu:
            network = network.to(self.device)

        # Start training
        network.train()
        lc = np.zeros([self.num_epochs])
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()

            train_err = 0
            train_batches = 0

            for batch in self.iterate_minibatches(self.X, self.y,
                                                  batch_size, shuffle=True):

                inputs  = Variable(torch.FloatTensor(batch[0]))
                targets = Variable(torch.FloatTensor(batch[1]))
                if self.gpu:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                optimizer.zero_grad()
                output, log_var, regularization = network(inputs)

                if self.regu:
                    loss = heteroscedastic_loss(targets, output, log_var) + regularization
                else:
                    loss = heteroscedastic_loss(targets, output, log_var)

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
                                         f'concdrop_k={itr}_{self.actv}_n{self.n_units_1}_e{self.num_epochs}.pt')
        torch.save(network.state_dict(), model_saving_path)
        print('concdrop model saved')

        # Estimate aleatoric uncertainty (overall tau^-1)
        # X_train_tensor = Variable(torch.FloatTensor(self.X))
        # if self.gpu:
        #     X_train_tensor = X_train_tensor.to(self.device)
        # y_train_mc_samples = [network(X_train_tensor) for _ in range(self.T)]
        # y_train_predict_samples = torch.stack([tup[0] for tup in y_train_mc_samples]).view(self.T, N).cpu().data.numpy()
        # self.aleatoric_uncertainty = np.mean(np.mean((y_train_predict_samples - self.y.flatten())**2 ,0))

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
        X_test: N input test points, np.ndarray (N, D)


        Returns
        ----------
        m: Predictive mean, np.array(N,)

        v: Predictive variance= epistemic uncertainty + aleatoric uncertainty, np.array(N,)

        e_v: Epistemic variance, np.array(N,)

        a_v: Aleatoric variance, np.array(N,)

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
        # model.eval()
        # MC_samples : list T x N x 1
        # Yt_hat = np.array([model(torch.Tensor(X_)).data.numpy() for _ in range(T)])
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
            # MC_samples = [model(Variable(torch.FloatTensor(X_))) for _ in range(T)]
            # means = torch.stack([model(Variable(torch.FloatTensor(X_)))[0] for _ in range(T)]).view(T, X_.shape[0]).data.numpy()

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

        return m, v, e_v, a_v

    @BaseModel._check_shapes_predict
    def validate(self, X_test, Y_test):
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.
        As well as the test log likelihood per point and rmse on test data

        Parameters
        ----------
        X_test: N input test points, np.ndarray (N, D)

        Y_test: N input test values, np.ndarray (N, 1)

        Returns
        ----------
        m: Predictive mean, np.array(N,)

        v: Predictive variance = epistemic uncertainty + aleatoric uncertainty, np.array(N,)

        e_v: Epistemic variance, np.array(N,)

        a_v: Aleatoric variance, np.array(N,)

        ppp: Per point predictive probability = test log likelihood /N , scalar


        rmse: Root mean square error, scalar


        """

        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Perform MC dropout
        model = self.model
        model.eval()
        T = self.T
        # MC_samples : list T x N x 1
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

        # compute uncertainties: aleatoric and epistemic (means_var)
        aleatoric_uncertainty = np.exp(logvar).mean(0)

        MC_pred_mean = np.mean(means, 0)
        means_var = np.var(means, 0)
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

