#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
"""

import numpy as np
from BayesOpt_General import Bayes_opt
from exps_tasks.math_functions import get_function
from utilities.utilities import get_init_data
import os
import argparse
import pickle
'''
Test
'''

obj_func = 'egg-2d'
model_type = 'GP'
batch_size = 1
bo_method = 'LCB'

X_query_file_name = obj_func + '/X_query' + model_type + bo_method + str(batch_size)
Y_query_file_name = obj_func + '/Y_query' + model_type + bo_method + str(batch_size)
X_opt_file_name = obj_func + '/X_opt' + model_type + bo_method + str(batch_size)
Y_opt_file_name = obj_func + '/Y_opt' + model_type + bo_method + str(batch_size)

with open(X_query_file_name, 'rb') as file:
    X_query_all_seeds = pickle.load(file)
with open(Y_query_file_name, 'rb') as file:
    Y_query_all_seeds = pickle.load(file)

with open(X_opt_file_name, 'rb') as file:
    X_opt_all_seeds = pickle.load(file)
with open(Y_opt_file_name, 'rb') as file:
    Y_opt_all_seeds = pickle.load(file)

print('finished')

