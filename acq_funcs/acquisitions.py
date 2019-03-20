#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
based on GPyOpt
"""

import numpy as np
from scipy.stats import norm
from scipy.special import erfc
from .base import BaseAcquisition
from GPyOpt.util.general import get_quantiles

""""All acquisition functions are to be maximised"""

class EI(BaseAcquisition):
    """
    Expected improvement acquisition function
    """

    def __init__(self, model, jitter=0.01):
        self.model = model
        self.jitter = jitter

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement
        """
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative
        """
        fmin = self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        return f_acqu, df_acqu

class LCB(BaseAcquisition):
    """
    Lower Confidence Bound acquisition function
    """

    def __init__(self, model, beta=3):
        self.model = model
        self.beta = beta

    def _compute_acq(self, x):
        """
        Computes the LCB
        """
        m, s = self.model.predict(x)
        f_acqu = - (m - self.beta * s)
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the LCB and its derivative
        """
        fmin = self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        f_acqu = -m + self.beta * s
        df_acqu = -dmdx + self.beta * dsdx
        return f_acqu, df_acqu

class MES(BaseAcquisition):
    """
    Max-value Entropy Search acquisition function
    -> require fmin samples
    """

    def __init__(self, model, beta=3):
        self.model = model

    def _compute_acq(self, x):
        x = np.atleast_2d(x)

        f_acqu_at_x = 0
        fmin_samples = self.model.fmin_samples.flatten()
        n_fmin_samples = len(fmin_samples)

        for i in range(n_fmin_samples):
            m, s = self.model.predict(x)

            gamma = (fmin_samples[i] - m)/s
            pdfgamma = np.exp(-0.5 * gamma**2) / np.sqrt(2*np.pi)
            cdfgamma = 0.5 * erfc(-gamma / np.sqrt(2))
            Z = 1 - cdfgamma
            f_acqu_at_x += - gamma * pdfgamma/(2*Z) - np.log(Z)

        mean_f_acqu = f_acqu_at_x / (n_fmin_samples)
        return mean_f_acqu