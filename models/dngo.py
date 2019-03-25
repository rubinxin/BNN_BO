from pybnn.dngo import DNGO
from .base import BaseModel
import numpy as np
class DNGOWrap(BaseModel):
    """
    A Wrapper for MC Dropout for a fully connected
    feed forward neural network..
    """
    def __init__(self, mini_batch_size=10,
                 n_units=[50, 50, 50],
                 alpha=1.0, beta=1000, prior=None, do_mcmc=True,
                 n_hypers=20, chain_length=2000, burnin_steps=2000,
                 normalize_input=True, normalize_output=True, rng=None):

        self.model = DNGO(batch_size=mini_batch_size, n_units_1=n_units[0], n_units_2=n_units[1],
                          n_units_3=n_units[2],alpha=alpha, beta=beta, prior=prior, do_mcmc=do_mcmc,
                 n_hypers=n_hypers, chain_length=chain_length, burnin_steps=burnin_steps,
                 normalize_input=normalize_input, normalize_output=normalize_output, rng=rng)

    def _create_model(self, X, Y):
        Y = Y.flatten()
        self.model.train(X, Y, do_optimize=True)

    def _update_model(self,  X_all, Y_all):
        """
        Updates the model with new observations.
        """
        Y_all = Y_all.flatten()

        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.train(X_all, Y_all, do_optimize=True)

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