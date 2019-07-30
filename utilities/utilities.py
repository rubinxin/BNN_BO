#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 24 Sept 2018 based MES paper (Zi Wang, 2017)

@author: robin
"""

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import sys
from scipy.special import erfc
import pdb

'''Utility Functions'''
# -----------------------------------------
# The first method for sampling global optima (Recommended)
# -----------------------------------------
def sample_fmin_Gumble(X, Y, model, bounds, nMs = 10, MC=False):
    # x is n x d
    gridSize = 10000
    x_ob = np.copy(X)
    y_ob = - np.copy(Y)

    d = bounds.shape[0]
    # bounds: d x 2
    Xgrid = np.tile(bounds[:, 0], (gridSize, 1)) + np.tile((bounds[:, 1] - bounds[:, 0]),
                                                           (gridSize, 1)) * np.random.rand(gridSize, d)
    # get lb and hb ,each are dx1
    Xgrid = np.vstack((Xgrid, x_ob))
    sx = Xgrid.shape[0]

    if MC == True:
        Nsamples = len(model.params[:, 0])
    else:
        Nsamples = 1

    noise_var = 1e-6

    f_max_samples = np.zeros([Nsamples, nMs])
    for i in range(Nsamples):
        if MC == True:
            # muVector, varVector = model.GP_normal[i].predict(Xgrid)
            'Not implemented'
        else:
            muVector, stdVector = model.predict(Xgrid)

        muVector = - muVector

        def probf(m0):
            z = (m0 - muVector)/stdVector
            cdf = 0.5 * erfc(-z / np.sqrt(2))
            return np.prod(cdf)

        left = np.max(y_ob)

        if probf(left) < 0.25:
            right = np.max(muVector + 5 *stdVector)
            while probf(right) < 0.75:
                right = right + right - left

            mgrid = np.linspace(left,right,100)
            # mgrid = 1x100 , muVector = sx x 1, stdVector = sx x 1
            z_grid = (np.tile(mgrid,(sx,1)) - np.tile(muVector,(1,100))) / np.tile(stdVector,(1,100))
            z_cdf = 0.5 * erfc(-z_grid / np.sqrt(2))
            prob = np.prod(z_cdf, axis=0) [None,:]
            # find the median and quartiles
            med = find_between(0.5, probf, prob, mgrid, 0.01)
            q1 = find_between(0.25, probf, prob, mgrid, 0.01)
            q2 = find_between(0.75, probf, prob, mgrid, 0.01)

            # Approximate the Gumbel parameters alpha and beta.
            beta = (q1 - q2) / (np.log( np.log(4 / 3)) - np.log(np.log(4)))
            alpha = med + beta * np.log(np.log(2));
            assert beta > 0 # check this line
            # Sample from the Gumbel distribution.
            maxes_samples = - np.log(-np.log(np.random.rand(1, nMs)))* beta + alpha;
            maxes_samples[np.where( maxes_samples < 0.5)] = left + 5.0 * np.sqrt(noise_var)
            f_max_samples[i, :] = maxes_samples
        else:
            f_max_samples[i, :] = left + 5.0 * np.sqrt(noise_var)

        f_opt_samples = - f_max_samples

    return f_opt_samples

def find_between(val, func, funcvals, mgrid, thres):
    t2 = np.argmin(abs(funcvals - val))
    check_diff = abs(funcvals[0,t2] - val)
    if abs(funcvals[0,t2] - val) < thres:
        res = mgrid[t2]
        return res

    assert funcvals[0,0] < val and funcvals[0,-1] > val
    if funcvals[0,t2] > val:
        left = mgrid[t2 - 1]
        right = mgrid[t2]
    else:
        left = mgrid[t2]
        right = mgrid[t2 + 1]

    mid = (left + right) / 2.0
    midval = func(mid)
    cnt = 1
    while abs(midval - val) > thres:
        if midval > val:
            right = mid
        else:
            left = mid

        mid = (left + right) / 2
        midval = func(mid)
        cnt = cnt + 1
        if cnt > 10000:
            pdb.set_trace()
    res = mid
    return res

# -----------------------------------------
# The second method for sampling global optima # ToDO to be updated
# -----------------------------------------
def sampl_fmin_samples(model, bounds_handler,nMs = 10, nFeatures = 1000, MC=False):
    # x is n x d
    d= model.GP_model.X.shape[1]
    if MC == True:
        l = model.params[:, 1:2]
        variance = model.params[:, 0]
    else:
        l = np.atleast_2d(model.GP_model.param_array[1:2])
        variance = np.atleast_2d(model.GP_model.param_array[0])

    Nsamples = len(variance)
    f_opt_samples = np.zeros([Nsamples,nMs])

    noise_var = 1e-6
    x_ob = model.GP_model.X
    y_ob = model.GP_model.Y

    for j in range(nMs):
        for i in range(Nsamples):

            # Draw weights for the random features
            W = np.random.randn(nFeatures,d) * np.tile(np.sqrt(l[i,:]),(nFeatures,1))
            b = 2 * np.pi * np.random.randn(nFeatures,1)

            # compute the features for x
            Z = np.sqrt(2* variance[i] /nFeatures) * np.cos( np.dot(W, x_ob.T) + np.tile(b,(1,x_ob.shape[0])))
            # Draw a coefficient theta
            random_noise = np.random.randn(nFeatures,1)

            if x_ob.shape[0] < nFeatures:
                Sigma = np.dot(Z.T,Z) + noise_var * np.eye(x_ob.shape[0])
                mu = np.dot( Z, np.linalg.solve(Sigma, y_ob) )
                D,U = np.linalg.eigh(Sigma)
                R = ( np.sqrt(D) * (np.sqrt(D) + np.sqrt(noise_var)) )**(-1)
                uznoise = np.dot(U.T, np.dot(Z.T, random_noise))
                ruznoise = R [:,None] * uznoise
                uruznoise = np.dot(U, ruznoise)
                zuruznoise = np.dot(Z, uruznoise)
                theta = random_noise - zuruznoise + mu
            else:
                Sigma = (np.dot(Z, Z.T)/noise_var + np.eye(nFeatures))
                mu = np.linalg.solve(Sigma, Z) * y_ob / noise_var
                theta = mu + random_noise * np.linalg.cholesky(Sigma)

            # obtain a function sampled from the posterior GP
            def targetVector(x):
                x = np.atleast_2d(x)
                t1 = theta.T * np.sqrt(2 * variance[i] / nFeatures)
                t2 = np.cos(W.dot(x.T) + np.tile(b, (1, x.shape[0])))
                y = np.dot( t1,t2 ).T
                return y

            def targetVectorGradient(x):
                x = np.atleast_2d(x)
                t1 = -theta.T * np.sqrt(2 * variance[i] / nFeatures)
                t2 = np.sin(np.dot(W, x.T) + b)
                t3 = np.tile(t2, (1, d)) * W
                dy = np.dot(t1, t3)
                return dy[0]

            sample_xopt, sample_yopt = global_optimiser(targetVector, bounds_handler, func_gradient=targetVectorGradient, find_max=False)
            if sample_yopt < model._min_of_Y: # TODO
                f_opt_samples[i,j] = sample_yopt
            else:
                if variance[i] > noise_var:
                    std_r = np.sqrt(variance[i])
                else:
                    std_r = np.sqrt(3*noise_var)
                f_opt_samples[i,j] = model._min_of_Y - abs(std_r * np.random.randn())
                # model._min_of_Y - 3 * np.sqrt(variance[i])

    return f_opt_samples

def global_optimiser(func, bounds_handler, func_gradient=None, find_max = False):

    """
    :param acq_func: acquisition function to be minimised
    :param random_starts: 2d array
    :param bounds: the same format as required by fmin_l_bfgs_b
    :param approx_grad: True if acq_func_gradient is not provided
    :return: max_location, max_acq_value
    """
    f_min = sys.float_info.max
    grid_fineness_level = max(int(np.ceil(np.power(5000, 1 / len(bounds_handler.get_expanded_bounds())))), 2)
    all_domains = [np.linspace(*(bound['domain']), grid_fineness_level) for bound in
                   bounds_handler.get_expanded_bounds()]
    all_guesses = np.array(np.meshgrid(*all_domains)).T.reshape(-1, len(bounds_handler.get_expanded_bounds()))
    # randomly choose 10
    random_candidates = all_guesses[np.random.choice(all_guesses.shape[0], 2000, replace=False), :]
    bounds = [bound['domain'] for bound in bounds_handler.get_expanded_bounds()]
    if find_max == True:
        target = lambda x: -func(x)
        target_gradient = lambda x: -func_gradient(x)
    else:
        target = lambda x: func(x)
        target_gradient = lambda x: func_gradient(x)

    results = target(random_candidates)
    top_candidates_idx = results.flatten().argsort()[:3]
    random_starts = random_candidates[top_candidates_idx]
    f_min = results[top_candidates_idx[0]]
    min_location = random_starts[0]
    for random_start in random_starts:
        if func_gradient:
            x, f_at_x, info = fmin_l_bfgs_b(target, random_start, bounds=bounds, fprime=target_gradient, approx_grad=False)
        else:
            x, f_at_x, info = fmin_l_bfgs_b(target, random_start, bounds=bounds, approx_grad=True)
        if f_at_x < f_min:
            f_min = f_at_x
            min_location = x

    if find_max == True:
        f_opt= -f_min
    else:
        f_opt = f_min
    return np.array([min_location]), f_opt


def global_minimiser_cheap(func, lb, hb, X_ob, maximise= False, func_gradient=None, gridSize=2000, n_start=3):
    bounds = list((li, ui) for li, ui in zip(lb, hb))
    d = len(bounds)
    # get lb and hb ,each are dx1
    Xgrid = np.tile(lb, (gridSize, 1)) + np.tile((hb - lb), (gridSize, 1)) * np.random.rand(gridSize, d)
    Xgrid = np.vstack((Xgrid, X_ob))

    if maximise == True:
        target_func = lambda x: -func(x)
        target_func_gradient = lambda x: -func_gradient(x)
    else:
        target_func = lambda x: func(x)
        target_func_gradient = lambda x: func_gradient(x)

    results = target_func(Xgrid)
    top_candidates_idx = results.flatten().argsort()[:n_start] # give the smallest n_start values in the ascending order
    random_starts = Xgrid[top_candidates_idx]
    f_min = results[top_candidates_idx[0]]
    opt_location = random_starts[0]

    for random_start in random_starts:
        if func_gradient:
            x, f_at_x, info = fmin_l_bfgs_b(func, random_start, bounds=bounds, fprime=target_func_gradient, approx_grad=False)
            # x, f_at_x, info = fmin_l_bfgs_b(func, random_start, bounds=bounds, approx_grad=False)

        else:
            x, f_at_x, info = fmin_l_bfgs_b(func, random_start, bounds=bounds, approx_grad=True)

        if f_at_x < f_min:
            f_min = f_at_x
            opt_location = x

    if maximise == True:
        f_opt = -f_min
    else:
        f_opt = f_min

    return np.array([opt_location]), f_opt

def optimise_acqu_func(acqu_func, bounds, X_ob, func_gradient=False, gridSize=10000, num_chunks = 10, n_start=1):

    # Turn acqu_func to be - acqu_func for minimisation
    target_func = lambda x: - acqu_func._compute_acq(x)

    def target_func_with_gradient(x):
        acqu_f, dacqu_f = acqu_func._compute_acq_withGradients(x)
        return -acqu_f, -dacqu_f

    bounds_opt = list(bounds)
    d = bounds.shape[0]
    # bounds: d x 2
    Xgrid = np.tile(bounds[:,0], (gridSize, 1)) + np.tile((bounds[:,1] - bounds[:,0]), (gridSize, 1)) * np.random.rand(gridSize, d)
    # Xgrid = np.vstack((Xgrid, X_ob))
    X_chunks = np.split(Xgrid, num_chunks)
    x_ob_chunk = X_ob[-200:,:]
    X_chunks.append(x_ob_chunk)

    results_list = []
    for i, x_chunk in enumerate(X_chunks):
        f_chunk = target_func(x_chunk)
        results_list.append(f_chunk)

    results = np.vstack(results_list)
    # results = target_func(Xgrid)
    top_candidates_idx = results.flatten().argsort()[:n_start] # give the smallest n_start values in the ascending order
    random_starts = Xgrid[top_candidates_idx]
    f_min = results[top_candidates_idx[0]]
    opt_location = random_starts[0]

    for random_start in random_starts:
        if func_gradient:
            x, f_at_x, info = fmin_l_bfgs_b(target_func_with_gradient, random_start, bounds=bounds_opt, approx_grad=False)
        else:
            x, f_at_x, info = fmin_l_bfgs_b(target_func, random_start, bounds=bounds_opt, approx_grad=True)

        if f_at_x < f_min:
            f_min = f_at_x
            opt_location = x

    f_opt = -f_min

    return np.array([opt_location]), f_opt

def get_init_data(obj_func, noise_var, n_init, bounds):
    """
    :param n_init: number of initial data points
    :param bounds: lower bounds: bounds[:,0], upper bounds: bounds[:,1]
    :return:
    """
    d = bounds.shape[0]
    x_init = np.random.uniform(bounds[:,0], bounds[:,1], (n_init, d))
    f_init = obj_func(x_init)
    y_init = f_init + np.sqrt(noise_var) * np.random.randn(n_init, 1)
    return x_init, y_init