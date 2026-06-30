import numpy as np

'''
ALL
'''
constant_friction = 0.8

time_slip_start = 100
time_slip_end = 150

time_slip_start_2 = 350
time_slip_end_2 = 450

time_slip_start_3 = 800
time_slip_end_3 = 900
use_dyn_friction = False
number_of_laps = 1
start_point = 1  # index on the trajectory to start from
weight_slip = 1.
weight_slip_2 = 1.0

start_point = 1  # index on the trajectory to start from

v_x_init = 8.0


'''
PaIL
'''
MASS   = 1225.887  # Vehicle mass
CM_norm= 0.9459

CF_norm= 1.6411 
DF_norm= 5331.253505960108*2/(9.81*MASS) 
BF_norm= 11.325229013060346

CR_norm= 1.6411 
DR_norm= 9099.898225690527*2/(9.81*MASS) 
BR_norm= 11.325229013060348

weight_K_gt = 0.05*np.array([1/BR_norm, 1/CR_norm, 1/DR_norm, 1/BF_norm, 1/CF_norm, 1/DF_norm, 1/CM_norm]) ** 2
T_horizon = 10
