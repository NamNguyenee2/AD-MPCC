import time
import yaml
import gym
from argparse import Namespace
from regulators.pure_pursuit import *
# from regulators.path_follow_mpcc_casadi import *
from regulators.path_follow_mpcc_casadi import *
from regulators.get_look_table import *
from models.dynamic import DynamicBicycleModel
from helpers.closest_point import *
import numpy as np

from pyglet.gl import GL_POINTS
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize_scalar
import pandas as pd
from scenarios_11 import *

@dataclass
class MPCConfigDYN:
    NXK: int = 7  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    NU:  int  = 2   # length of input vector: u = = [acceleration, steering speed]
    TK:  int  = 20  # finite time horizon length kinematic

    '''
    Parameters for MPCC objective
    '''
    Rk_ca: list = field(
        default_factory=lambda: np.diag([0.0005, 0.01, 0.01]))  # input cost matrix, penalty for inputs - [accel, steering_speed, vi_speed]
    Rdk_ca: list = field(
        default_factory=lambda: np.diag([0.0001, .01, 0.01]))  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed, vi_speed]

    q_contour: float = 1.0
    q_lag: float     = 1000.0
    q_theta: float   = 180.

    '''
    Learning parameters for predictive model
    '''
    num_param: int = 7
    '''
    Model's parameters
    '''

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float    = 0.05 # time step [s] kinematic
    scale: float  = 0.25
    dlk: float    = 3.0*0.25  # dist step [m] kinematic
    LENGTH: float = 4.298  # Length of the vehicle [m]
    WIDTH: float  = 1.674  # Width of the vehicle [m]
    LR: float = 1.50876
    LF: float = 0.88392
    WB: float = 0.88392 + 1.50876  # Wheelbase [m]  
    MAX_THETA: float = np.inf  # maximum a virtual theta for MPCC
    MIN_THETA: float = -np.inf # minimum a virtual theta for MPCC
    MAX_VI: float = 50.0 # maximum a virtual control input vi for MPCC
    MIN_VI: float = 0.0 # minimum a virtual control input vi for MPCC
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum ste
    # MIN_STEER: float = -0.8189  # maximum steering angle [rad]
    # MAX_STEER: float = 0.8189  # maximum ste
    MAX_ACCEL: float = 50  # maximum acceleration [m/ss]
    MAX_DECEL: float = -50  # maximum acceleration [m/ss]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 50.0  # maximum speed [m/s]
    MIN_SPEED: float = 2.0  # minimum backward speed [m/s]
    MIN_POS_X: float = -np.inf  # minimum horizontal direction (x) 
    MAX_POS_X: float = np.inf  # maximum horizontal direction (x) 
    MIN_POS_Y: float = -np.inf  # minimum vertical direction (y) 
    MAX_POS_Y: float = np.inf  # maximum vertical direction (y) 
    MIN_SPEED_LAT: float = -np.inf  # minimum latteral speed (m/s)
    MAX_SPEED_LAT: float = np.inf # maximum latteral speed (m/s)

    # model parameters
    MASS: float = 1225.887  # Vehicle mass
    I_Z: float = 1560.3729  # Vehicle inertia
    TORQUE_SPLIT: float = 0.0  # Torque distribution

    # https://arxiv.org/pdf/1905.05150.pdf - equation (7)
    CR0: float = 2.3451
    CR2: float = 0.0095

    

def draw_point(e, point, colour):
    scaled_point = 50. * point
    ret = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_point[0], scaled_point[1], 0]), ('c3B/stream', colour))
    return ret


class DrawDebug:
    def __init__(self):
        self.reference_traj_show = np.array([[0, 0]])
        self.predicted_traj_show = np.array([[0, 0]])
        self.dyn_obj_drawn = []
        self.f = 0

    def draw_debug(self, e):
        # delete dynamic objects
        while len(self.dyn_obj_drawn) > 0:
            if self.dyn_obj_drawn[0] is not None:
                self.dyn_obj_drawn[0].delete()
            self.dyn_obj_drawn.pop(0)

        # spawn new objects
        for p in self.reference_traj_show:
            self.dyn_obj_drawn.append(draw_point(e, p, [255, 0, 0]))

        for p in self.predicted_traj_show:
            self.dyn_obj_drawn.append(draw_point(e, p, [0, 255, 0]))


def main():  
    # Choose program parameters
    model_to_use = 'dynamic'  
    map_name = 'Oschersleben' 
    rotate_map = True  # !!!! If the car is spawning with bad orientation change value here !!!! TODO Fix here so this is not needed anymore
    use_dyn_friction = False
    slip_mode = True
    
    control_step = 100.0  # ms
    render_every = 200.0  # render graphics every n simulation steps
    constant_speed = False
    constant_speed_value = 8.0
    velocity_profile_multiplier = 0.9

    dyn_config = MPCConfigDYN()
    work = {'mass': 1225.88, 'lf': 0.80597534362552312, 'tlad': 10.6461887897713965, 'vgain': 1.0}

    # Load map config file
    with open('configs/config_%s.yaml' % map_name) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    if use_dyn_friction:
        if map_name == 'l_shape':
            tpamap_name = './maps/l_shape/friction_data/l_shape_l720_track_tpamap.csv'
            tpadata_name = './maps/l_shape/friction_data/l_shape_l720_track_tpadata.json'
        if map_name == 'DualLaneChange':
            # tpamap_name = './maps/DualLaneChange/friction_data/DualLaneChange_track_tpamap.csv'
            # tpamap_name = './maps/DualLaneChange/friction_data/DualLaneChange5z_track_tpamap.csv'
            tpamap_name = './maps/DualLaneChange/friction_data/DualLaneChange3zv2_track_tpamap.csv'
            # tpadata_name = './maps/DualLaneChange/friction_data/DualLaneChange_track_tpadata.json'
            # tpadata_name = './maps/DualLaneChange/friction_data/DualLaneChange5z_track_tpadata.json'
            tpadata_name = './maps/DualLaneChange/friction_data/DualLaneChange3zv2_track_tpadata.json'
        if map_name == 'SaoPaulo':
            
            tpamap_name = './maps/SaoPaulo/friction_data/SaoPaulo_track_tpamap.csv'
            tpadata_name = './maps/SaoPaulo/friction_data/SaoPaulo_track_tpadata.json'
        if map_name == 'Nuerburgring':
            tpamap_name = './maps/Nuerburgring/friction_data/Nuerburgring_tpamap.csv'
            tpadata_name = './maps/Nuerburgring/friction_data/Nuerburgring_tpadata.json'

        tpamap = np.loadtxt(tpamap_name, delimiter=';', skiprows=1)
        print("I am here!")
        tpadata = {}
        with open(tpadata_name) as f:
            tpadata = json.load(f)

    raceline = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
    waypoints = np.array(raceline)

    # Scale = 0.25
    # X = waypoints[:, 1]*Scale
    # Y = waypoints[:, 2]*Scale

    # data_save = pd.DataFrame({
    #     "X": X,
    #     "Y": Y
    # })

    # data_save.to_csv(f"scale{Scale}_{map_name}_waypoints.csv", index=False)

    # return 0
    waypoints[:, 1] *= dyn_config.scale
    waypoints[:, 2] *= dyn_config.scale
    if rotate_map == True:
        waypoints[:, 3] += 1.5707963268

    if constant_speed:
        waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * constant_speed_value
    else:
        waypoints[:, 5] *= velocity_profile_multiplier

    # init controllers
    planner_pp = PurePursuitPlanner(conf, 0.805975 + 1.50876)
    planner_pp.waypoints = waypoints
    # planner_ekin_mpc = STMPCPlanner(model=DynamicBicycleModel(config=dyn_config), waypoints=waypoints,
    #                                 config=dyn_config)
    
    ini = np.array([[waypoints[start_point, 1], waypoints[start_point, 2], (waypoints[start_point, 3]
                        + np.pi) % (2*np.pi) - np.pi, 0.0, v_x_init, 0.0, 0.0]])
    # ini = np.array([[1.,0., (waypoints[start_point, 3] 
    #                     + np.pi) % (2*np.pi) - np.pi, 0.0, 6.0, 0.0, 0.0]])
    planner_dyn_mpc = STMPCCPlannerCasadi(model=DynamicBicycleModel(config=dyn_config), waypoints=waypoints,
                                   config=dyn_config, index=start_point)
    # planner_dyn_mpc = STMPCPlanner(model=DynamicBicycleModel(config=dyn_config), waypoints=waypoints,
    #                                config=dyn_config)
    find_theta = ThetaLookupTable(spline_x, spline_y, theta_min, theta_max, n_samples=10000000)
    draw = DrawDebug()

    def render_callback(env_renderer):

        e = env_renderer
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 10000
        e.left = left - 5000
        e.right = right + 5000
        e.top = top + 8000
        e.bottom = bottom - 5000
        planner_pp.render_waypoints(e)
        draw.draw_debug(e)

    # MB - reference point: center of mass
    # dynamic_ST - reference point: center of mass

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                   num_agents=1, timestep=0.001, model='MB', drive_control_mode='acc',
                   steering_control_mode='vel')

    env.add_render_callback(render_callback)
    # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    obs, step_reward, done, info = env.reset(
        ini)
    env.render()

    laptime = 0.0
    start = time.time()
    last_render = 0

    # init logger
    log = {'time': [], 'x_ref': [], 'y_ref':[],
            'x': [], 'y': [], 'vx': [], 'yaw': [], 'vy': [], 'yaw_rate': [], 'steer_angle': [], 'lap_n': [],
           'acce': [], 'steering_rate':[],
           'theta': [], 'v_ref': [], 'tracking_error': [], 'time_compute': [],
            'BR': [], 'CR': [], 'DR': [], 'BF': [], 'CF': [], 'DF': [], 'CM': [], 'mu_x': [], 'mu_y': []}

    # calc number of sim steps per one control step
    num_of_sim_steps = int(control_step / (env.timestep * 1000.0))


    # print('Model used: %s' % model_to_use)
    count = 0
    while not done:
        
        
        # Regulator step MPC
        vehicle_state = np.array([env.sim.agents[0].state[0],  # x
                                  env.sim.agents[0].state[1],  # y
                                  env.sim.agents[0].state[3],  # vx
                                  env.sim.agents[0].state[4] ,  # yaw angle
                                  env.sim.agents[0].state[10],  # vy
                                  env.sim.agents[0].state[5],  # yaw rate
                                  env.sim.agents[0].state[2],  # steering angle
                                  ]) + np.random.randn(7) * 0.00001
        
        if len(log['theta']) == 0:
                    theta_cur = find_theta.query_near_prev(
                        env.sim.agents[0].state[0],
                        env.sim.agents[0].state[1],
                        0.0,
                        k_neighbors=20,
                        forward_only=True,
                    )
        else:
            theta_cur = find_theta.query_near_prev(
                env.sim.agents[0].state[0],
                env.sim.agents[0].state[1],
                log['theta'][-1],
                k_neighbors=20,
                forward_only=True,
            )        # print("=============")
        # # else:
        # print("=============")
        # print("step:", count)
        # print("=============")
       

        # CF= 1.6411 
        # DF= 5331.253505960108*2/(9.81*dyn_config.MASS) 
        # BF= 11.325229013060346

        # CR= 1.6411 
        # DR= 9099.898225690527*2/(9.81*dyn_config.MASS) 
        # BR= 11.325229013060348
        # CM = 0.9459

        p_cx1 = 1.6411
        p_ex1 = 0.46403
        p_kx1 = 22.303

        gravity = 9.81
        F_zf = (dyn_config.LR / (dyn_config.LF + dyn_config.LR)) * dyn_config.MASS * gravity
        F_zr = (dyn_config.LF / (dyn_config.LF + dyn_config.LR)) * dyn_config.MASS * gravity 

        muF = constant_friction
        CF = p_cx1
        DF = muF * F_zf
        EF = p_ex1
        KF = F_zf * p_kx1
        BF = KF / (CF * DF)
        DF = DF/(dyn_config.MASS*gravity/2)

        muF = constant_friction
        CR = p_cx1
        DR = muF * F_zr
        ER = p_ex1
        KR = F_zr * p_kx1
        BR = KR / (CR * DR)
        DR = DR/(dyn_config.MASS*gravity/2)
        CM = 0.9459

        compute_start_time = time.time()
        K_gt = np.array([BR, CR, DR, BF, CF, DF, CM])
        u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, _ = planner_dyn_mpc.plan(vehicle_state, K_gt)
        compute_end_time = time.time()
        
        # u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, _, _ = planner_dyn_mpc.plan(vehicle_state)
        u[0] = u[0] / planner_dyn_mpc.config.MASS  # Force to acceleration
        # u[0] = 0.0
        print("vx:", env.sim.agents[0].state[3])
        print("vy:", env.sim.agents[0].state[10])
        print("acceleration:", u[0])
        
        # draw predicted states and reference trajectory
        draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
        draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

        _, tracking_error, _, _, n_point = nearest_point_on_trajectory(np.array([env.sim.agents[0].state[0], env.sim.agents[0].state[1]]),
                                                                       np.array([waypoints[:, 1], waypoints[:, 2]]).T)

        

        # set correct friction to the environment
        if slip_mode == True:
            if ((theta_cur>=time_slip_start) and (theta_cur<=time_slip_end)) or ((theta_cur>=time_slip_start_2) and (theta_cur<=time_slip_end_2)):
                env.params['tire_p_dy1'] = constant_friction* weight_slip
                env.params['tire_p_dx1'] = constant_friction * weight_slip

            elif ((theta_cur>time_slip_start_3) and (theta_cur<time_slip_end_3)):
                env.params['tire_p_dy1'] = constant_friction* weight_slip_2
                env.params['tire_p_dx1'] = constant_friction * weight_slip_2
            else:    
                env.params['tire_p_dy1'] = constant_friction 
                env.params['tire_p_dx1'] = constant_friction 
        # env.params['tire_p_dy1'] = constant_friction 
        # env.params['tire_p_dx1'] = constant_friction 
        # print('Model used: %s' % model_to_use)
        # Simulation step
        step_reward = 0.0
        sim_time = 0.0
        log['time_compute'].append(compute_end_time - compute_start_time)
        log['time'].append(laptime)
        log['x'].append(env.sim.agents[0].state[0])
        log['y'].append(env.sim.agents[0].state[1])
        log['vx'].append(env.sim.agents[0].state[3])
        log['vy'].append(env.sim.agents[0].state[10])
        log['yaw'].append(env.sim.agents[0].state[4])
        log['yaw_rate'].append(env.sim.agents[0].state[5])
        log['steer_angle'].append(env.sim.agents[0].state[2])
        log['theta'].append(theta_cur)
        log['acce'].append(u[0])
        log['steering_rate'].append(u[1])

        
        

        for i in range(num_of_sim_steps):
            obs, rew, done, info = env.step(np.array([[u[1], u[0]]]))
            step_reward += rew
            sim_time += env.timestep

            # Rendering
            last_render += 1
            if last_render >= render_every:
                last_render = 0
                env.render(mode='human_fast')

        laptime += step_reward
        log['x_ref'].append(waypoints[:, 1][n_point])
        log['y_ref'].append(waypoints[:, 2][n_point])
        log['v_ref'].append(waypoints[:, 5][n_point])
        log['tracking_error'].append(tracking_error)
        log['lap_n'].append(obs['lap_counts'][0])
        log['BR'].append(BR)
        log['CR'].append(CR)
        log['DR'].append(DR)
        log['BF'].append(BF)
        log['CF'].append(CF)
        log['DF'].append(DF)
        log['CM'].append(CM)

        log['mu_x'].append(env.params['tire_p_dx1'])
        log['mu_y'].append(env.params['tire_p_dy1'])
        
        if tracking_error > 3.5:
            done = 1
            break

        if obs['lap_counts'][0] == number_of_laps:  
            done = 1
    print('Lap finished! Lap time: %.2f seconds' % laptime)
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)

    with open(f'scale{dyn_config.scale}_TK{dyn_config.TK}_log_{map_name}_full_Vinit_{v_x_init}friction{constant_friction}', 'w') as f:
        json.dump(log, f)


if __name__ == '__main__':
    main()


