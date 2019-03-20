#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
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

    analytical_gradient_prediction = True

    def __init__(self, model, jitter=0.01):
        self.model = model
        self.jitter = jitter

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        """
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
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

    analytical_gradient_prediction = True

    def __init__(self, model, beta=3):
        self.model = model
        self.beta = beta

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        """
        m, s = self.model.predict(x)
        f_acqu = - (m - self.beta * s)
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        fmin = self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        f_acqu = -m + self.beta * s
        df_acqu = -dmdx + self.beta * dsdx
        return f_acqu, df_acqu