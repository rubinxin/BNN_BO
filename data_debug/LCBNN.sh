#!/usr/bin/bash

python bo_general_exps.py -f='egg-2d' -m='LCBNN' -acq='LCB' -bm='CL' -b=1 -nitr=60 -s=3 -uo='se_y' &
python bo_general_exps.py -f='egg-2d' -m='LCBNN' -acq='LCB' -bm='CL' -b=1 -nitr=60 -s=3 -uo='se_prod_y' &
python bo_general_exps.py -f='egg-2d' -m='LCBNN' -acq='LCB' -bm='CL' -b=1 -nitr=60 -s=3 -uo='se_yclip' &

wait
