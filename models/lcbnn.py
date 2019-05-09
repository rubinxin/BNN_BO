from pybnn.lcbnn import LCBNN
from .base import BaseModel
import numpy as np

class LCBNNWarp(BaseModel):
    """
    A Wrapper for Loss-Calibrated BNN.
    """
    def __init__(self, mini_batch_size=10,num_epochs=500,
                 n_units=[50, 50, 50],
                 dropout = 0.05, length_scale = 1e-2, T = 1000, util_type='se_y',
                 normalize_input=True, normalize_output=True, seed=42, actv='tanh'):
        # self.model = \
        self.model = LCBNN(batch_size=mini_batch_size,num_epochs=num_epochs,
                 n_units_1=n_units[0], n_units_2=n_units[1], n_units_3=n_units[2],
                 dropout_p = dropout, length_scale = length_scale, T = T,
                 normalize_input=normalize_input, normalize_output=normalize_output, rng=seed,
                 weights=None, util_type=util_type, actv=actv)

    def _create_model(self, X, Y):
        Y = Y.flatten()
        train_mes_loss, train_util = self.model.train(X, Y)

    def _update_model(self,  X_all, Y_all, itr=0, seed = None):
        """
        Updates the model with new observations.
        """
        Y_all = Y_all.flatten()

        if self.model is None:
            train_mes_loss, train_util = self._create_model(X_all, Y_all)
        else:
            train_mes_loss, train_util = self.model.train(X_all, Y_all)

        np.save(f'lcbnn_train_mse_loss_i{itr}',train_mes_loss)
        np.save(f'lcbnn_train_util_i{itr}',train_util)


    def predict(self, X):
        """
        Predictions with the model. Returns predictive means and standard deviations at X.
        """
        X = np.atleast_2d(X)
        m, v = self.model.predict(X)
        # m and v have shape (N,)
        s = np.sqrt(v)
        return m[:,None], s[:,None]

    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        return print('Not Implemented')