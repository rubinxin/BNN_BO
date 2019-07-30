import time
import logging
import os
import sys
import numpy as np
from keras.layers import Input, Dense, Lambda, merge, concatenate
from keras.models import Model
import keras.backend as K
from keras import initializers
from keras.engine import InputSpec
from keras.layers import Dense, Lambda, Wrapper
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from tensorflow import set_random_seed


class ConcreteDropout(Wrapper):
    def __init__(self, layer, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):

        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        input_dim = np.prod(input_shape[-1])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
                K.log(self.p + eps)
                - K.log(1. - self.p + eps)
                + K.log(unif_noise + eps)
                - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))

            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)

# class Net():
#     def __init__(self, n_inputs, n_units=[50, 50, 50],
#                  weight_regularizer=1e-6, dropout_regularizer=1e-5, actv='tanh',
#                  p_min=0.1, p_max=0.1):
#         super(Net, self).__init__()
#         self.linear1 = nn.Linear(n_inputs, n_units[0])
#         self.linear2 = nn.Linear(n_units[0], n_units[1])
#         self.linear3 = nn.Linear(n_units[1], n_units[2])
#         self.out_mu = nn.Linear(n_units[2], 1)
#         self.out_logvar = nn.Linear(n_units[2], 1)
#
#         self.conc_drop1 = ConcreteDropout(weight_regularizer=weight_regularizer,
#                                           dropout_regularizer=dropout_regularizer,
#                                           init_min=p_min, init_max=p_max)
#         self.conc_drop2 = ConcreteDropout(weight_regularizer=weight_regularizer,
#                                           dropout_regularizer=dropout_regularizer,
#                                           init_min=p_min, init_max=p_max)
#         self.conc_drop3 = ConcreteDropout(weight_regularizer=weight_regularizer,
#                                           dropout_regularizer=dropout_regularizer,
#                                           init_min=p_min, init_max=p_max)
#         self.conc_drop_mu = ConcreteDropout(weight_regularizer=weight_regularizer,
#                                             dropout_regularizer=dropout_regularizer,
#                                             init_min=p_min, init_max=p_max)
#         self.conc_drop_logvar = ConcreteDropout(weight_regularizer=weight_regularizer,
#                                                 dropout_regularizer=dropout_regularizer,
#                                                 init_min=p_min, init_max=p_max)
#
        # regularization = torch.empty(4, device=x.device)
        # x1 = self.activation(self.linear1(x))
        # x2, regularization[0] = self.conc_drop2(x1, nn.Sequential(self.linear2, self.activation))
        #
        # x3, regularization[1] = self.conc_drop3(x2, nn.Sequential(self.linear3, self.activation))
        #
        # mean, regularization[2] = self.conc_drop_mu(x3, self.out_mu)
        # log_var, regularization[3] = self.conc_drop_logvar(x3, self.out_logvar)

        # return mean, log_var, regularization.sum()
    #     # return mean, regularization.sum()

def Net(n_inputs, n_units=[50, 50, 50],
                 weight_regularizer=1e-6, dropout_regularizer=1e-5, actv='tanh',
                 p_min=0.1, p_max=0.1):

    # define the model
    D = 1
    wr = weight_regularizer
    dr = dropout_regularizer
    inp = Input(shape=(n_inputs,))  #
    x = inp
    x = Dense(n_units[0], activation=actv)(x)
    x = ConcreteDropout(Dense(n_units[1], activation=actv), weight_regularizer=wr, dropout_regularizer=dr, init_min=p_min, init_max=p_max)(x)
    x = ConcreteDropout(Dense(n_units[2], activation=actv), weight_regularizer=wr, dropout_regularizer=dr, init_min=p_min, init_max=p_max)(x)
    mean = ConcreteDropout(Dense(D), weight_regularizer=wr, dropout_regularizer=dr, init_min=p_min, init_max=p_max)(x)
    log_var = ConcreteDropout(Dense(D), weight_regularizer=wr, dropout_regularizer=dr, init_min=p_min, init_max=p_max)(x)
    out = concatenate([mean, log_var])
    model = Model(inp, out)

    return model

class MCCONCRETEDROP_KERAS(object):

    def __init__(self, batch_size=10, num_epochs=500,
                 learning_rate=0.01,
                 adapt_epoch=5000, n_units_1=50, n_units_2=50, n_units_3=50,
                 length_scale = 1e-4, T = 100, regu = False, mc_tau=False,
                 normalize_input=True, normalize_output=True, rng=None, gpu=True, actv='tanh',
                 init_p_min = 0.1, init_p_max = 0.1):
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

        np.random.seed(self.seed)
        set_random_seed(self.seed)

        self.X = None
        self.y = None
        self.network = None
        self.length_scale = length_scale
        self.gpu = gpu
        self.regu = regu
        self.mc_tau = mc_tau
        self.T = T
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.actv = actv
        self.p_min = init_p_min
        self.p_max = init_p_max

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_learning_rate = learning_rate

        self.n_units_1 = n_units_1
        self.n_units_2 = n_units_2
        self.n_units_3 = n_units_3
        self.adapt_epoch = adapt_epoch # TODO check
        self.network = None
        self.models = []

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
        N = self.X.shape[0]
        features = X.shape[1]
        wr = self.length_scale ** 2. / N
        dr = 2. / N
        D = self.X.shape[1]
        self.model = Net(n_inputs=features, n_units=[self.n_units_1, self.n_units_1, self.n_units_3],
                      weight_regularizer=wr, dropout_regularizer=dr,  actv=self.actv, p_min=self.p_min, p_max=self.p_max)

        if itr > 0:
            model_loading_path = os.path.join(saving_path,
                                              f'concdrop_k={itr-1}_{self.actv}_n{self.n_units_1}_e{self.num_epochs}.h5')
            self.model.load_weights(model_loading_path)

        def heteroscedastic_loss(true, pred):
            mean = pred[:, :D]
            log_var = pred[:, D:]
            precision = K.exp(-log_var)
            return K.sum(precision * (true - mean) ** 2. + log_var, -1)

        self.model.compile(optimizer='adam', loss=heteroscedastic_loss)




        hist = self.model.fit(self.X, self.y, epochs=self.num_epochs, batch_size=batch_size, verbose=0)
        loss = hist.history['loss'][-1]

        model_saving_path = os.path.join(saving_path,
                                         f'concdrop_k={itr}_{self.actv}_n{self.n_units_1}_e{self.num_epochs}.h5')
        self.model.save_weights(model_saving_path)
        print('keras concdrop model saved')

        return -0.5 * loss

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
        D = self.X_.shape[1]
        model = self.model
        model.eval()
        T     = self.T

        MC_samples = np.array([model.predict(X_) for _ in range(T)])
        T = MC_samples.shape[0]
        means = MC_samples[:, :, :self.D]  # K x N x D
        logvar = MC_samples[:, :, self.D:]


        # mc_time = time.time() - start_mc
        # print(f'mc_time={mc_time}')
        # compute uncertainties: aleatoric and epistemic (means_var)
        aleatoric_uncertainty = np.exp(logvar).mean(0)
        MC_pred_mean = np.mean(means, 0)
        means_var  = np.var(means, 0)
        MC_pred_var = means_var + aleatoric_uncertainty

        m = MC_pred_mean.flatten()
        v = MC_pred_var.flatten()
        e_v = means_var.flatten()
        a_v = aleatoric_uncertainty.flatten()

        if self.normalize_output:
            m = zero_mean_unit_var_denormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2

        return m, v, e_v, a_v

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
        D = X_.shape[1]
        N = Y_test.shape[0]
        model = self.model
        T     = self.T

        MC_samples = np.array([model.predict(X_) for _ in range(T)])
        T = MC_samples.shape[0]
        means = MC_samples[:, :, :D]  # K x N x D
        logvar = MC_samples[:, :, D:]

        # compute uncertainties: aleatoric and epistemic (means_var)
        aleatoric_uncertainty = np.exp(logvar).mean(0)
        MC_pred_mean = np.mean(means, 0)
        means_var  = np.var(means, 0)
        MC_pred_var = means_var + aleatoric_uncertainty

        test_ll = -0.5 * np.exp(-logvar) * (means - Y_test[None]) ** 2. - 0.5 * logvar - 0.5 * np.log(2 * np.pi)
        test_ll = np.sum(np.sum(test_ll, -1), -1)
        test_ll = logsumexp(test_ll) - np.log(T)
        ppp = test_ll / N  # per point predictive probability
        rmse = np.mean((np.mean(means, 0) - Y_test) ** 2.) ** 0.5

        m = MC_pred_mean.flatten()
        v = MC_pred_var.flatten()
        e_v = means_var.flatten()
        a_v = aleatoric_uncertainty.flatten()

        if self.normalize_output:
            m = zero_mean_unit_var_denormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2

        return m, v, e_v, a_v, ppp, rmse

def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max