from dataclasses import dataclass, field

import numpy as np
import pyglet
from pyglet.gl import GL_POINTS


@dataclass
class MPCConfigDYNBase:
    NXK: int = 7
    NU: int = 2
    TK: int = 20

    Rk_ca: list = field(
        default_factory=lambda: np.diag([0.0005, 2.0, 0.01]))
    Rdk_ca: list = field(
        default_factory=lambda: np.diag([0.01, 2.0, 0.01]))

    q_contour: float = 1.0
    q_lag: float     = 1000.0
    q_theta: float   = 250.0

    num_param: int = 7

    scale: float = 0.25
    N_IND_SEARCH: int = 20
    DTK: float = 0.05
    dlk: float = 3.0 * 0.25
    LENGTH: float = 4.298
    WIDTH: float = 1.674
    LR: float = 1.50876
    LF: float = 0.88392
    WB: float = 0.88392 + 1.50876
    MAX_THETA: float = np.inf
    MIN_THETA: float = -np.inf
    MAX_VI: float = 50.0
    MIN_VI: float = 0.0
    MIN_STEER: float = -0.4189
    MAX_STEER: float = 0.4189
    MAX_ACCEL: float = 50.0
    MAX_DECEL: float = -50.0
    MAX_STEER_V: float = 3.2
    MAX_SPEED: float = 50.0
    MIN_SPEED: float = 2.0
    MIN_POS_X: float = -np.inf
    MAX_POS_X: float = np.inf
    MIN_POS_Y: float = -np.inf
    MAX_POS_Y: float = np.inf
    MIN_SPEED_LAT: float = -np.inf
    MAX_SPEED_LAT: float = np.inf

    MASS: float = 1225.887
    I_Z: float = 1560.3729
    TORQUE_SPLIT: float = 0.0

    CR0: float = 2.3451
    CR2: float = 0.0095


def ground_truth_pacejka(mass):
    CF = 1.6411
    DF = 5331.253505960108 * 2 / (9.81 * mass)
    BF = 11.325229013060346

    CR = 1.6411
    DR = 9099.898225690527 * 2 / (9.81 * mass)
    BR = 11.325229013060348
    CM = 0.9459

    return np.array([BR, CR, DR, BF, CF, DF, CM])


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
        while len(self.dyn_obj_drawn) > 0:
            if self.dyn_obj_drawn[0] is not None:
                self.dyn_obj_drawn[0].delete()
            self.dyn_obj_drawn.pop(0)

        for p in self.reference_traj_show:
            self.dyn_obj_drawn.append(draw_point(e, p, [255, 0, 0]))

        for p in self.predicted_traj_show:
            self.dyn_obj_drawn.append(draw_point(e, p, [0, 255, 0]))


def make_render_callback(planner_pp, draw):
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
    return render_callback


def capture_frame_from_renderer(renderer):
    buffer = pyglet.image.get_buffer_manager().get_color_buffer()
    image_data = buffer.get_image_data()
    data = image_data.get_data('RGB', image_data.width * 3)
    frame = np.frombuffer(data, dtype=np.uint8)
    frame = frame.reshape(image_data.height, image_data.width, 3)
    frame = np.flipud(frame)
    return frame


def make_log_dict():
    return {'time': [], 'x_ref': [], 'y_ref': [],
            'x': [], 'y': [], 'vx': [], 'yaw': [], 'vy': [], 'yaw_rate': [], 'steer_angle': [], 'lap_n': [],
            'acce': [], 'steering_rate': [],
            'theta': [], 'v_ref': [], 'tracking_error': [], 'time_compute': [],
            'BR': [], 'CR': [], 'DR': [], 'BF': [], 'CF': [], 'DF': [], 'CM': [], 'mu_x': [], 'mu_y': []}


def apply_slip_friction(env, theta_cur, constant_friction, weight_slip, weight_slip_2,
                         time_slip_start, time_slip_end, time_slip_start_2, time_slip_end_2,
                         time_slip_start_3, time_slip_end_3, track_length=None):
    if track_length is not None:
        in_slip_1 = (
            ((theta_cur >= time_slip_start) and (theta_cur <= time_slip_end))
            or ((theta_cur >= time_slip_start + track_length / 20) and (theta_cur <= time_slip_end + track_length / 20))
            or ((theta_cur >= time_slip_start_2) and (theta_cur <= time_slip_end_2))
            or ((theta_cur >= time_slip_start_2 + track_length / 20) and (theta_cur <= time_slip_end_2 + track_length / 20))
        )
        in_slip_2 = (
            ((theta_cur > time_slip_start_3) and (theta_cur < time_slip_end_3))
            or ((theta_cur > time_slip_start_3 + track_length / 20) and (theta_cur < time_slip_end_3 + track_length / 20))
        )
    else:
        in_slip_1 = (
            ((theta_cur >= time_slip_start) and (theta_cur <= time_slip_end))
            or ((theta_cur >= time_slip_start_2) and (theta_cur <= time_slip_end_2))
        )
        in_slip_2 = (theta_cur > time_slip_start_3) and (theta_cur < time_slip_end_3)

    if in_slip_1:
        env.params['tire_p_dy1'] = constant_friction * weight_slip
        env.params['tire_p_dx1'] = constant_friction * weight_slip
    elif in_slip_2:
        env.params['tire_p_dy1'] = constant_friction * weight_slip_2
        env.params['tire_p_dx1'] = constant_friction * weight_slip_2
    else:
        env.params['tire_p_dy1'] = constant_friction
        env.params['tire_p_dx1'] = constant_friction


def get_mu_numeric(dynamic, state, param):
    param = np.asarray(param, dtype=float).reshape(-1)
    state = np.asarray(state, dtype=float).reshape(-1)

    BR = param[0]
    CR = param[1]
    DR = param[2]
    BF = param[3]
    CF = param[4]
    DF = param[5]

    vx = float(state[2])
    vy = float(state[4])
    yaw_rate = float(state[5])
    steering_angle = float(state[6])

    vx_abs = max(abs(vx), 0.05)
    vx_safe = np.sign(vx) * vx_abs if vx != 0.0 else 0.05

    alfa_f = steering_angle - np.arctan2(yaw_rate * dynamic.LF + vy, vx_safe)
    alfa_r = np.arctan2(yaw_rate * dynamic.LR - vy, vx_safe)

    mu_F = DF * np.sin(CF * np.arctan(BF * alfa_f))
    mu_R = DR * np.sin(CR * np.arctan(BR * alfa_r))
    return float(mu_F), float(mu_R)


def compute_stage_weights(dynamic, states, theta_arr, param, theta_min, theta_max, round_theta, lookup_q):
    track_length = float(theta_max - theta_min)
    theta_t = (float(theta_arr) - theta_min) % track_length + theta_min
    mu_F_t, mu_R_t = get_mu_numeric(dynamic, states, param)
    step = track_length / round_theta
    k_temp = int(theta_t // step)
    k_temp = min(k_temp, round_theta - 1)
    theta_t = theta_t - k_temp * step
    qc_t, ql_t, qt_t = lookup_q([theta_t, states[2], states[4], mu_F_t, mu_R_t], k_neighbors=20)
    return qc_t, ql_t, qt_t
