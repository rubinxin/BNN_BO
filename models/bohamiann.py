from pybnn.bohamiann import Bohamiann
from .base import BaseModel
import numpy as np

class BOHAMIANNWarp(BaseModel):
    """
    A Wrapper for MC Dropout for a fully connected
    feed forward neural network..
    """
    def __init__(self, num_samples=6000, keep_every=50, lr=1e-2,
                 normalize_input: bool = True, normalize_output: bool = True, verbose=True):

        self.verbose = verbose
        self.num_samples = num_samples
        self.keep_every = keep_every
        self.lr = lr
        self.model = Bohamiann(normalize_input=normalize_input, normalize_output=normalize_output)

    def _create_model(self, X, Y):
        Y = Y.flatten()
        num_burn_in_steps = X.shape[0] * 100
        num_steps = X.shape[0] * 100 + self.num_samples

        self.model.train(X, Y, num_steps=num_steps, num_burn_in_steps=num_burn_in_steps,
                         keep_every=self.keep_every, lr=self.lr, verbose=self.verbose)

    def _update_model(self,  X_all, Y_all):
        """
        Updates the model with new observations.
        """
        Y_all = Y_all.flatten()
        num_burn_in_steps = X_all.shape[0] * 100
        num_steps = X_all.shape[0] * 100 + self.num_samples

        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.train(X_all, Y_all, num_steps=num_steps, num_burn_in_steps=num_burn_in_steps,
                         keep_every=self.keep_every, lr=self.lr, verbose=self.verbose)

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