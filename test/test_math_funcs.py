#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from exps_tasks.math_functions import get_function
from models.gp import GPModel
from utilities.utilities import global_minimiser_cheap
import os
'''
Test
'''
func_name = 'ackley-10d'
f, x_bounds, _, true_fmin = get_function(func_name)
d = x_bounds.shape[0]
n_init = 20
var_noise = 1.0e-10
lb = x_bounds[:,0]
hb = x_bounds[:,1]

def test_gp():
    #  Specify the objective function and parameters (noise variance, input dimension, initial observation
    X_ob = np.random.rand(3,d)*2-1
    X_opt, f_opt = global_minimiser_cheap(f, lb, hb, X_ob, maximise= False, func_gradient=None, gridSize=10000, n_start=3)
    print(f'X_opt={X_opt},f_opt={f_opt}, true_fmin={true_fmin}')

    return X_opt, f_opt

if __name__ == '__main__':
    test_gp()
