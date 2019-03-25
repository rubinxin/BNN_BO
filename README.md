# Bayesian Optimisation with Bayesian Neural Network
The code package implemented the surrogate models: 
  * GP 
  * DNGO (NN + Bayesian Linear Regression)
  * MC Dropout
  * BOHAMIAN (HMC-based BNN) 


## Running a BO experiment
Run Bayesian optimisation experiments: `python bo_general_exps.py` followed by the following flags:
  * `-f` Objective function: default=`'egg-2d'`
  * `-m` Surrogate model: `'GP'`(default), `'MCDROP'`, `'DNGO'` or `'BOHAM'`
  * `-acq` Acquisition function: `'LCB'`(default) or `'EI'`
  * `-bm` Batch option: `'CL'`(default) or `'KB'`
  * `-b` BO Batch size: default = `1`                    
  * `-nitr` Max BO iterations: default = `40`
  * `-s` Number of random initialisation: default = `20`
  E.g. `python bo_general_exps.py -f='egg-2d' -m='GP' -acq='LCB' -bm='CL' -b=1 -nitr=60 -s=10`

## Requirement
 * python 3
 * torch
 * torchvision
 * emcee
 * gpy
 * gpyopt




