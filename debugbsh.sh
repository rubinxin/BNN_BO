#!/bin/bash

python bo_general_exps.py -f='egg-2d' -m='LCBNN' -acq='LCB' -bm='CL' -b=1 -nitr=60 -s=1 -uo='se_y'
python bo_general_exps.py -f='egg-2d' -m='LCBNN' -acq='LCB' -bm='CL' -b=1 -nitr=60 -s=1 -uo='se_yclip'
python bo_general_exps.py -f='egg-2d' -m='LCBNN' -acq='LCB' -bm='CL' -b=1 -nitr=60 -s=1 -uo='se_prod_y'

python bo_general_exps.py -f='egg-2d' -m='LCBNN' -acq='LCB' -bm='CL' -b=1 -nitr=60 -s=1 -a='relu' -uo='se_y'
python bo_general_exps.py -f='egg-2d' -m='LCBNN' -acq='LCB' -bm='CL' -b=1 -nitr=60 -s=1 -a='relu' -uo='se_yclip'
python bo_general_exps.py -f='egg-2d' -m='LCBNN' -acq='LCB' -bm='CL' -b=1 -nitr=60 -s=1 -a='relu' -uo='se_prod_y'