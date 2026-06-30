import numpy as np
from scipy.optimize import minimize
from numba import njit


@njit(cache=True)
def _predictive_model_np(
    state,
    control_input,
    param,
    MASS,
    I_Z,
    LF,
    LR,
    TORQUE_SPLIT,
    CR0,
    CR2,
    MAX_DECEL,
    MAX_ACCEL,
    MAX_STEER_V,
    MIN_SPEED,
    MAX_SPEED,
    MIN_STEER,
    MAX_STEER,
):
    BR, CR, DR, BF, CF, DF, CM = param

    vx = state[2]
    if vx < MIN_SPEED:
        vx = MIN_SPEED
    elif vx > MAX_SPEED:
        vx = MAX_SPEED

    steering_angle = state[6]
    if steering_angle < MIN_STEER:
        steering_angle = MIN_STEER
    elif steering_angle > MAX_STEER:
        steering_angle = MAX_STEER

    Fxr = control_input[0]
    low_fxr = MAX_DECEL * MASS
    high_fxr = MAX_ACCEL * MASS
    if Fxr < low_fxr:
        Fxr = low_fxr
    elif Fxr > high_fxr:
        Fxr = high_fxr

    delta_v = control_input[1]
    if delta_v < -MAX_STEER_V:
        delta_v = -MAX_STEER_V
    elif delta_v > MAX_STEER_V:
        delta_v = MAX_STEER_V

    yaw = state[3]
    vy = state[4]
    yaw_rate = state[5]

    vx_safe = np.maximum(np.abs(vx), 0.05)

    alfa_f = steering_angle - np.arctan2(yaw_rate * LF + vy, vx_safe)
    alfa_r = np.arctan2(yaw_rate * LR - vy, vx_safe)

    FzR = 0.5 * MASS * 9.81
    FzF = 0.5 * MASS * 9.81

    Ffy = FzF * DF * np.sin(CF * np.arctan(BF * alfa_f))
    Fry = FzR * DR * np.sin(CR * np.arctan(BR * alfa_r))

    Fx = CM * Fxr - CR0 - CR2 * vx_safe * vx_safe
    Frx = Fx * (1.0 - TORQUE_SPLIT)
    Ffx = Fx * TORQUE_SPLIT

    out = np.empty(7, dtype=np.float64)
    out[0] = vx_safe * np.cos(yaw) - vy * np.sin(yaw)
    out[1] = vx_safe * np.sin(yaw) + vy * np.cos(yaw)
    out[2] = (Frx - Ffy * np.sin(steering_angle) + Ffx * np.cos(steering_angle) + vy * yaw_rate * MASS) / MASS
    out[3] = yaw_rate
    out[4] = (Fry + Ffy * np.cos(steering_angle) + Ffx * np.sin(steering_angle) - vx_safe * yaw_rate * MASS) / MASS
    out[5] = (Ffy * LF * np.cos(steering_angle) - Fry * LR) / I_Z
    out[6] = delta_v
    return out


@njit(cache=True)
def _rk4_step_np(
    x,
    u,
    param,
    dt,
    MASS,
    I_Z,
    LF,
    LR,
    TORQUE_SPLIT,
    CR0,
    CR2,
    MAX_DECEL,
    MAX_ACCEL,
    MAX_STEER_V,
    MIN_SPEED,
    MAX_SPEED,
    MIN_STEER,
    MAX_STEER,
):
    k1 = _predictive_model_np(x, u, param, MASS, I_Z, LF, LR, TORQUE_SPLIT, CR0, CR2, MAX_DECEL, MAX_ACCEL, MAX_STEER_V, MIN_SPEED, MAX_SPEED, MIN_STEER, MAX_STEER)
    k2 = _predictive_model_np(x + 0.5 * dt * k1, u, param, MASS, I_Z, LF, LR, TORQUE_SPLIT, CR0, CR2, MAX_DECEL, MAX_ACCEL, MAX_STEER_V, MIN_SPEED, MAX_SPEED, MIN_STEER, MAX_STEER)
    k3 = _predictive_model_np(x + 0.5 * dt * k2, u, param, MASS, I_Z, LF, LR, TORQUE_SPLIT, CR0, CR2, MAX_DECEL, MAX_ACCEL, MAX_STEER_V, MIN_SPEED, MAX_SPEED, MIN_STEER, MAX_STEER)
    k4 = _predictive_model_np(x + dt * k3, u, param, MASS, I_Z, LF, LR, TORQUE_SPLIT, CR0, CR2, MAX_DECEL, MAX_ACCEL, MAX_STEER_V, MIN_SPEED, MAX_SPEED, MIN_STEER, MAX_STEER)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=True)
def _weighted_mse_np(
    x_cur,
    u_cur,
    x_next_real,
    param,
    dt,
    MASS,
    I_Z,
    LF,
    LR,
    TORQUE_SPLIT,
    CR0,
    CR2,
    MAX_DECEL,
    MAX_ACCEL,
    MAX_STEER_V,
    MIN_SPEED,
    MAX_SPEED,
    MIN_STEER,
    MAX_STEER,
):
    decay = 0.8
    n_steps = x_cur.shape[0]
    weight = decay ** (n_steps - 1)
    weighted_se_sum = 0.0

    for i in range(n_steps):
        x_next_pred = _rk4_step_np(
            x_cur[i], u_cur[i], param, dt, MASS, I_Z, LF, LR, TORQUE_SPLIT, CR0, CR2,
            MAX_DECEL, MAX_ACCEL, MAX_STEER_V, MIN_SPEED, MAX_SPEED, MIN_STEER, MAX_STEER
        )
        err2 = x_next_pred[2] - x_next_real[i, 2]
        err4 = x_next_pred[4] - x_next_real[i, 4]
        err5 = x_next_pred[5] - x_next_real[i, 5]
        weighted_se_sum += (err2 * err2 + err4 * err4 + err5 * err5) * weight
        weight /= decay

    return weighted_se_sum / n_steps

class Pacejka:
    def __init__(self, config, K_gt=None, weight_K_gt=None, T_horizon=None):
        self.config = config
        self.K_gt = K_gt if K_gt is not None else np.zeros(7)
        self.weight_K_gt = weight_K_gt if weight_K_gt is not None else np.zeros(7)
        
        # Vehicle parameters from config 
        self.DTK = self.config.DTK
        self.MASS = self.config.MASS
        self.I_Z = self.config.I_Z
        self.LF = self.config.LF
        self.LR = self.config.LR
        self.TORQUE_SPLIT = self.config.TORQUE_SPLIT

        self.CR0 = self.config.CR0
        self.CR2 = self.config.CR2

        # Initialize parameters
        self.BR = self.K_gt[0]
        self.CR = self.K_gt[1]
        self.DR = self.K_gt[2]
        self.BF = self.K_gt[3]
        self.CF = self.K_gt[4]
        self.DF = self.K_gt[5]
        self.CM = self.K_gt[6]

        self.T_horizon = T_horizon
        self.last_opt_result = None
        self.bounds = self._build_bounds()
        # Emphasize dynamic states that carry tire-parameter information.
    def find_param(self, x_hist, u_hist):
        """
        Find optimal Pacejka tire model parameters
        
        Args:
            x_hist: State history, shape (N, 7)
            u_hist: Control input history, shape (N, 2)
        
        Returns:
            param: Optimal parameters [BR, CR, DR, BF, CF, DF, CM]
        """
        x_hist = np.asarray(x_hist)
        u_hist = np.asarray(u_hist)

        if self.T_horizon is not None and x_hist.shape[0] > self.T_horizon:
            x_hist = x_hist[-self.T_horizon:]
            u_hist = u_hist[-self.T_horizon:]

        # Fix indexing: predict from current to next state
        x_cur = x_hist[:-1, :]  # Current states
        u_cur = u_hist[:-1, :]  # Current controls
        x_next_real = x_hist[1:, :]  # Next states (ground truth)
        if x_cur.shape[0] == 0:
            return np.array([self.BR, self.CR, self.DR, self.BF, self.CF, self.DF, self.CM], dtype=float)
        
        # Initial parameter guess: [BR, CR, DR, BF, CF, DF, CM]
        param_init = np.array([self.BR, self.CR, self.DR, self.BF, self.CF, self.DF, self.CM], dtype=float)
        
        # Optimization
        compute_loss = self.compute_loss

        def objective(param):
            return compute_loss(x_cur, u_cur, x_next_real, param)
        
        result = minimize(
            objective,
            param_init,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': 40, 'disp': False}
        )
        self.last_opt_result = result
        opt_param = result.x
        self.BR, self.CR, self.DR, self.BF, self.CF, self.DF, self.CM = opt_param
        return opt_param

    def _build_bounds(self):
        """Build parameter bounds once; they only depend on vehicle mass."""
        CM_norm = 0.9459
        CF_norm = 1.6411
        DF_norm = 5331.253505960108 * 2 / (9.81 * self.MASS)
        BF_norm = 11.325229013060346
        CR_norm = 1.6411
        DR_norm = 9099.898225690527 * 2 / (9.81 * self.MASS)
        BR_norm = 11.325229013060348
        up_norm = 1.2
        low_norm = 0.8
        return [
            (BR_norm * low_norm, BR_norm * up_norm),  # BR
            (CR_norm * low_norm, CR_norm * up_norm),  # CR
            (DR_norm * low_norm, DR_norm * up_norm),  # DR
            (BF_norm * low_norm, BF_norm * up_norm),  # BF
            (CF_norm * low_norm, CF_norm * up_norm),  # CF
            (DF_norm * low_norm, DF_norm * up_norm),  # DF
            (CM_norm * low_norm, CM_norm * up_norm),  # CM
        ]
    
    def compute_loss(self, x_cur, u_cur, x_next_real, param):
        """
        Compute loss between predicted and real next states
        """
        x_cur = np.asarray(x_cur, dtype=np.float64)
        u_cur = np.asarray(u_cur, dtype=np.float64)
        x_next_real = np.asarray(x_next_real, dtype=np.float64)
        param = np.asarray(param, dtype=np.float64)

        mse = _weighted_mse_np(
            x_cur,
            u_cur,
            x_next_real,
            param,
            self.DTK,
            self.MASS,
            self.I_Z,
            self.LF,
            self.LR,
            self.TORQUE_SPLIT,
            self.CR0,
            self.CR2,
            self.config.MAX_DECEL,
            self.config.MAX_ACCEL,
            self.config.MAX_STEER_V,
            self.config.MIN_SPEED,
            self.config.MAX_SPEED,
            self.config.MIN_STEER,
            self.config.MAX_STEER,
        )
        # Regularization term (if K_gt are prior parameters)
        if self.K_gt is not None and self.weight_K_gt is not None:
            regularization = np.sum(self.weight_K_gt * (param - self.K_gt) ** 2)
        else:
            regularization = 0.0
        
        total_loss = mse + regularization
        
        return total_loss
        
    def clip_input(self, u):
        """Clip control inputs to feasible range"""
        # u = [Fxr, delta_v]
        u0 = np.clip(
            u[0],
            self.config.MAX_DECEL * self.config.MASS,
            self.config.MAX_ACCEL * self.config.MASS)
        
        u1 = np.clip(
            u[1],
            -self.config.MAX_STEER_V,
            self.config.MAX_STEER_V)
        
        return np.array([u0, u1])

    def clip_output(self, state):
        """Clip state variables to feasible range"""
        # state = [x, y, vx, yaw, vy, yaw_rate, steering_angle]
        state = np.array(state)
        
        vx = np.clip(
            state[2],
            self.config.MIN_SPEED,
            self.config.MAX_SPEED)
        
        steering = np.clip(
            state[6],
            self.config.MIN_STEER,
            self.config.MAX_STEER)
        
        return np.array([
            state[0],   # x
            state[1],   # y
            vx,
            state[3],   # yaw
            state[4],   # vy
            state[5],   # yaw_rate
            steering])
        
    def predictive_model(self, state, control_input, param):
        """
        Vehicle dynamics using Pacejka tire model
        
        Args:
            state: [x, y, vx, yaw, vy, yaw_rate, steering_angle]
            control_input: [Fxr, delta_v]
            param: [BR, CR, DR, BF, CF, DF, CM]
        
        Returns:
            f: Time derivatives of state
        """
        # Extract parameters
        BR = param[0]
        CR = param[1]
        DR = param[2]
        BF = param[3]
        CF = param[4]
        DF = param[5]
        CM = param[6]
        
        state = self.clip_output(state)
        control_input = self.clip_input(control_input)
        
        x = state[0]
        y = state[1]
        vx = state[2]
        yaw = state[3]
        vy = state[4]
        yaw_rate = state[5]
        steering_angle = state[6]
        
        Fxr = control_input[0]
        delta_v = control_input[1]

        # Safe velocity handling
        vx_safe = np.maximum(np.abs(vx), 0.05)

        # Tire slip angles
        alfa_f = steering_angle - np.arctan2(yaw_rate * self.LF + vy, vx_safe)
        alfa_r = np.arctan2(yaw_rate * self.LR - vy, vx_safe)

        FzR = 1/2 * self.MASS * 9.81
        FzF = 1/2 * self.MASS * 9.81
  
        # Pacejka tire model
        Ffy = FzF *DF * np.sin(CF * np.arctan(BF * alfa_f))
        Fry = FzR *DR * np.sin(CR * np.arctan(BR * alfa_r))

        # Longitudinal forces
        Fx = CM * Fxr - self.CR0 - self.CR2 * vx_safe ** 2.0
        Frx = Fx * (1.0 - self.TORQUE_SPLIT)
        Ffx = Fx * self.TORQUE_SPLIT

        # Vehicle dynamics (7 states)
        dx = vx_safe * np.cos(yaw) - vy * np.sin(yaw)
        dy = vx_safe * np.sin(yaw) + vy * np.cos(yaw)
        dvx = (1.0 / self.MASS) * (Frx - Ffy * np.sin(steering_angle) + Ffx * np.cos(steering_angle) + vy * yaw_rate * self.MASS)
        dyaw = yaw_rate
        dvy = (1.0 / self.MASS) * (Fry + Ffy * np.cos(steering_angle) + Ffx * np.sin(steering_angle) - vx_safe * yaw_rate * self.MASS)
        dyaw_rate = (1.0 / self.I_Z) * (Ffy * self.LF * np.cos(steering_angle) - Fry * self.LR)
        dsteering = delta_v
        
        f = np.array([dx, dy, dvx, dyaw, dvy, dyaw_rate, dsteering])
        
        return f
    
    def rk4_step(self, x, u, param):
        """4th order Runge-Kutta integration step"""
        dt = self.DTK
        k1 = self.predictive_model(x, u, param)
        k2 = self.predictive_model(x + dt/2 * k1, u, param)
        k3 = self.predictive_model(x + dt/2 * k2, u, param)
        k4 = self.predictive_model(x + dt   * k3, u, param)
        x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next
    
    def Euler_step(self, x, u, param):
        return x + self.DTK * self.predictive_model(x, u, param)

