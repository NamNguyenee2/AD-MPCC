import casadi as ca
import numpy as np
import copy
from regulators.get_look_table import *
from diff_look import *
import os

os.environ["PATH"] += r";C:\Users\namng\Downloads\CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5\CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5\bin"

round_theta = 20
class STMPCCPlannerCasadi:
    def __init__(self, model, config, waypoints=None, index=None, x0_opt_prev=None):
        self.waypoints = waypoints
        self.model = model
        self.config = config

        self.look_theta = ThetaLookupTable(spline_x, spline_y, theta_min, theta_max, n_samples=10000000)
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.track_length = float(theta_max - theta_min)
        self.input_o = np.ones((self.config.NU, self.config.TK)) * np.NaN
        self.states_output = np.ones((self.config.NXK, self.config.TK + 1)) * np.NaN

        self.q_contour = self.config.q_contour
        self.q_lag = self.config.q_lag
        self.q_theta = self.config.q_theta

        self.DTK = self.config.DTK
        self.MASS = self.config.MASS
        self.I_Z = self.config.I_Z
        self.LF = self.config.LF
        self.LR = self.config.LR
        self.TORQUE_SPLIT = self.config.TORQUE_SPLIT

        self.CR0 = self.config.CR0
        self.CR2 = self.config.CR2
        self.theta_prev = float(self.theta_min)
        self.u_his = np.tile(np.array([[10.0], [0.0]]), (1, self.config.TK))
        self.mpc_prob_init()

    def _normalize_controls(self, u_seq):
        u_arr = np.asarray(u_seq, dtype=float)
        if u_arr.ndim == 1 and u_arr.size == self.config.NU:
            return np.tile(u_arr[:, None], (1, self.config.TK))
        if u_arr.ndim == 2 and u_arr.shape == (self.config.NU, self.config.TK):
            return u_arr
        if u_arr.ndim == 2 and u_arr.shape[0] == self.config.NU and u_arr.shape[1] >= 1:
            return np.tile(u_arr[:, [0]], (1, self.config.TK))
        return np.zeros((self.config.NU, self.config.TK), dtype=float)

    def plan(self, states, param, weights, waypoints=None):
        if waypoints is not None:
            self.waypoints = waypoints

        u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, done = self.MPCC_Control(states, param, weights)
        return u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, done

    def clip_input(self, u):
        u0 = ca.fmin(
            ca.fmax(u[0], self.config.MAX_DECEL * self.config.MASS),
            self.config.MAX_ACCEL * self.config.MASS
        )

        u1 = ca.fmin(
            ca.fmax(u[1], -self.config.MAX_STEER_V),
            self.config.MAX_STEER_V
        )

        return ca.vertcat(u0, u1)

    def clip_output(self, state):
        vx = ca.fmin(
            ca.fmax(state[2], self.config.MIN_SPEED),
            self.config.MAX_SPEED
        )

        steering = ca.fmin(
            ca.fmax(state[6], self.config.MIN_STEER),
            self.config.MAX_STEER
        )

        return ca.vertcat(
            state[0],
            state[1],
            vx,
            state[3],
            state[4],
            state[5],
            steering
        )

    def predictive_model(self, state, control_input, param):
        self.BR = param[0]
        self.CR = param[1]
        self.DR = param[2]
        self.BF = param[3]
        self.CF = param[4]
        self.DF = param[5]
        self.CM = param[6]

        x = state[0]
        y = state[1]
        vx = state[2]
        yaw = state[3]
        vy = state[4]
        yaw_rate = state[5]
        steering_angle = state[6]

        Fxr = control_input[0]
        delta_v = control_input[1]

        alfa_f = steering_angle - ca.atan2(yaw_rate * self.LF + vy, vx)
        alfa_r = ca.atan2(yaw_rate * self.LR - vy, vx)

        FzR = 1/2 * self.MASS * 9.81
        FzF = 1/2 * self.MASS * 9.81
        Ffy = FzR * self.DF * ca.sin(self.CF * ca.atan(self.BF * alfa_f))
        Fry = FzF * self.DR * ca.sin(self.CR * ca.atan(self.BR * alfa_r))

        Fx = self.CM * Fxr - self.CR0 - self.CR2 * vx ** 2.0
        Frx = Fx * (1.0 - self.TORQUE_SPLIT)
        Ffx = Fx * self.TORQUE_SPLIT

        dx = vx * ca.cos(yaw) - vy * ca.sin(yaw)
        dy = vx * ca.sin(yaw) + vy * ca.cos(yaw)
        dvx = (1.0 / self.MASS) * (Frx - Ffy * ca.sin(steering_angle) + Ffx * ca.cos(steering_angle) + vy * yaw_rate * self.MASS)
        dyaw = yaw_rate
        dvy = (1.0 / self.MASS) * (Fry + Ffy * ca.cos(steering_angle) + Ffx * ca.sin(steering_angle) - vx * yaw_rate * self.MASS)
        dyaw_rate = (1.0 / self.I_Z) * (Ffy * self.LF * ca.cos(steering_angle) - Fry * self.LR)
        dsteering = delta_v

        f = ca.vertcat(dx, dy, dvx, dyaw, dvy, dyaw_rate, dsteering)

        return f

    def RK4(self, x, u, param):
        dt = self.DTK
        k1 = self.predictive_model(x, u, param)
        k2 = self.predictive_model(x + dt/2 * k1, u, param)
        k3 = self.predictive_model(x + dt/2 * k2, u, param)
        k4 = self.predictive_model(x + dt * k3, u, param)
        x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next

    def RK2(self, x, u, param):
        dt = self.DTK
        k1 = self.predictive_model(x, u, param)
        k2 = self.predictive_model(x + (dt / 2.0) * k1, u, param)
        x_next = x + dt * k2
        return x_next

    def Euler(self, x, u, param):
        x_next = x + self.DTK* self.predictive_model(x, u, param)
        return x_next

    def get_initial_guess(self, x0, param, theta0):
        states = np.zeros((self.config.NXK, self.config.TK + 1), dtype=float)
        controls = np.zeros((self.config.NU, self.config.TK), dtype=float)
        theta_arr = np.zeros(self.config.TK + 1, dtype=float)
        vi_arr = np.zeros(self.config.TK, dtype=float)

        states[:, 0] = x0
        theta_arr[0] = theta0

        for t in range(self.config.TK):
            u_t = np.array([0.0, 0.0], dtype=float)
            controls[:, t] = u_t
            x_next = self.Euler(states[:, t], u_t, param)
            states[:, t + 1] = np.array(x_next).astype(float).flatten()

        theta_lookup = np.zeros(self.config.TK + 1, dtype=float)

        theta_lookup[0] = float(theta0)
        for t in range(1, self.config.TK + 1):
            theta_lookup[t] = self.look_theta.query_near_prev(
                states[0, t],
                states[1, t],
                theta_lookup[t - 1],
                k_neighbors=20,
                forward_only=True,
                max_forward_step=8.0,
            )

        theta_unwrap = np.zeros_like(theta_lookup)
        theta_unwrap[0] = float(theta0)
        for t in range(1, self.config.TK + 1):
            theta_unwrap[t] = theta_lookup[t]

        theta_arr[:] = theta_unwrap
        for t in range(self.config.TK):
            vi_arr[t] = (theta_arr[t + 1] - theta_arr[t]) / self.DTK

        x0_opt = np.zeros(self.n_states + self.n_controls + self.n_theta + self.n_vi, dtype=float)
        idx = 0
        x0_opt[idx:idx + self.n_states] = states.T.reshape(-1)
        idx += self.n_states
        x0_opt[idx:idx + self.n_controls] = controls.T.reshape(-1)
        idx += self.n_controls
        x0_opt[idx:idx + self.n_theta] = theta_arr
        idx += self.n_theta
        x0_opt[idx:idx + self.n_vi] = vi_arr

        return x0_opt

    def mpc_prob_init(self):
        self.x0_opt = None
        self.xk = ca.MX.sym('xk', self.config.NXK, self.config.TK + 1)
        self.uk = ca.MX.sym('uk', self.config.NU, self.config.TK)
        self.theta_k = ca.MX.sym('theta_k', self.config.TK + 1)
        self.vik = ca.MX.sym('vik', self.config.TK)

        self.x0k = ca.MX.sym('x0k', self.config.NXK)
        self.theta0 = ca.MX.sym('theta0')
        self.param = ca.MX.sym('param', self.config.num_param)
        self.weights = ca.MX.sym('weights', 3)

        theta_grid = np.array(theta, dtype=float)
        x_grid = np.array(x, dtype=float)
        y_grid = np.array(y, dtype=float)
        phi_grid = np.unwrap(np.array([lookup_phi(t) for t in theta_grid]))

        order = np.argsort(theta_grid)
        theta_grid = theta_grid[order]
        x_grid = x_grid[order]
        y_grid = y_grid[order]
        phi_grid = phi_grid[order]

        mask = np.ones_like(theta_grid, dtype=bool)
        mask[1:] = theta_grid[1:] > theta_grid[:-1]
        theta_grid = theta_grid[mask]
        x_grid = x_grid[mask]
        y_grid = y_grid[mask]
        phi_grid = phi_grid[mask]

        L = float(self.track_length)

        span = theta_grid[-1] - theta_grid[0]
        if np.isclose(span, L, rtol=0.0, atol=1e-8 * max(1.0, L)):
            theta_grid = theta_grid[:-1]
            x_grid = x_grid[:-1]
            y_grid = y_grid[:-1]
            phi_grid = phi_grid[:-1]

        self.ref_x_fun = ca.interpolant('ref_x_fun', 'linear', [theta_grid], x_grid)
        self.ref_y_fun = ca.interpolant('ref_y_fun', 'linear', [theta_grid], y_grid)
        self.ref_phi_fun = ca.interpolant('ref_phi_fun', 'linear', [theta_grid], phi_grid)

        objective = 0.0
        dxy_limit = float(getattr(self.config, "DXY_SOFT_LIMIT", 2.5))
        q_dxy_soft = float(getattr(self.config, "Q_DXY_SOFT", 50.0))

        constraints = []
        lbg = []
        ubg = []

        for t in range(self.config.TK):
            x_next = self.Euler(self.xk[:, t], self.uk[:, t], self.param)
            constraints.append(self.xk[:, t + 1] - x_next)
            lbg.extend([0.0] * self.config.NXK)
            ubg.extend([0.0] * self.config.NXK)

        for t in range(self.config.TK):
            theta_next = self.theta_k[t] + self.DTK * self.vik[t]
            constraints.append(self.theta_k[t + 1] - theta_next)
            lbg.append(0.0)
            ubg.append(0.0)

        for t in range(self.config.TK + 1):
            x_ref = self.ref_x_fun(self.theta_k[t])
            y_ref = self.ref_y_fun(self.theta_k[t])
            phi_t = self.ref_phi_fun(self.theta_k[t])
            sin_phi_t = ca.sin(phi_t)
            cos_phi_t = ca.cos(phi_t)

            dx = self.xk[0, t] - x_ref
            dy = self.xk[1, t] - y_ref

            dxy2 = dx**2 + dy**2
            dxy2_violation = ca.fmax(0.0, dxy2 - dxy_limit**2)
            objective += q_dxy_soft * dxy2_violation**2

            e_c = sin_phi_t * dx - cos_phi_t * dy
            e_l = -cos_phi_t * dx - sin_phi_t * dy
            objective += self.weights[0] * e_c ** 2
            objective += self.weights[1] * e_l ** 2

        for t in range(self.config.TK):
            objective += -self.weights[2] * self.vik[t]

        for t in range(self.config.TK):
            p_u_1 = self.uk[0, t]
            p_u_2 = self.uk[1, t]
            p_vi = self.vik[t]
            p_u = ca.vertcat(p_u_1, p_u_2, p_vi)
            objective += p_u.T@self.config.Rk_ca@ p_u

        for t in range(self.config.TK - 1):
            du_1 = self.uk[0, t + 1] - self.uk[0, t]
            du_2 = self.uk[1, t + 1] - self.uk[1, t]
            dvi = self.vik[t + 1] - self.vik[t]
            du = ca.vertcat(du_1, du_2, dvi)
            objective += du.T@self.config.Rdk_ca@ du

        constraints.append(self.xk[:, 0] - self.x0k)
        lbg.extend([0.0] * self.config.NXK)
        ubg.extend([0.0] * self.config.NXK)

        constraints.append(self.theta_k[0] - self.theta0)
        lbg.append(0.0)
        ubg.append(0.0)

        g = ca.vertcat(*constraints)

        opt_variables = ca.vertcat(
            ca.reshape(self.xk, -1, 1),
            ca.reshape(self.uk, -1, 1),
            self.theta_k,
            self.vik
        )

        opt_params = ca.vertcat(
            self.x0k,
            self.theta0,
            self.param,
            self.weights
        )

        nlp = {
            'x': opt_variables,
            'f': objective,
            'g': g,
            'p': opt_params
        }

        linear_solver = getattr(self.config, "IPOPT_LINEAR_SOLVER", "mumps")

        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
            "ipopt.acceptable_tol": 1e-1,
            "print_time": 0,
            "ipopt.linear_solver": linear_solver,
            "ipopt.print_level": 0,
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.n_states   = self.config.NXK * (self.config.TK + 1)
        self.n_controls = self.config.NU * self.config.TK
        self.n_theta    = self.config.TK + 1
        self.n_vi       = self.config.TK

        self.lbx = []
        self.ubx = []

        self.lbx.extend([
            self.config.MIN_POS_X,
            self.config.MIN_POS_Y,
            self.config.MIN_SPEED,
            -np.inf,
            self.config.MIN_SPEED_LAT,
            -np.inf,
            self.config.MIN_STEER
        ] * (self.config.TK + 1))
        self.ubx.extend([
            self.config.MAX_POS_X,
            self.config.MAX_POS_Y,
            self.config.MAX_SPEED,
            np.inf,
            self.config.MAX_SPEED_LAT,
            np.inf,
            self.config.MAX_STEER
        ] * (self.config.TK + 1))

        self.lbx.extend([
            self.config.MAX_DECEL * self.MASS,
            -self.config.MAX_STEER_V
        ] * self.config.TK)
        self.ubx.extend([
            self.config.MAX_ACCEL * self.MASS,
            self.config.MAX_STEER_V
        ] * self.config.TK)

        self.lbx.extend([self.config.MIN_THETA] * (self.config.TK + 1))
        self.ubx.extend([self.config.MAX_THETA] * (self.config.TK + 1))

        self.lbx.extend([self.config.MIN_VI] * self.config.TK)
        self.ubx.extend([self.config.MAX_VI] * self.config.TK)

        self.lbg = lbg
        self.ubg = ubg

        self.x0_opt = np.zeros(self.n_states + self.n_controls + self.n_theta + self.n_vi)

    def mpc_prob_solve(self, x0, u_his, param, weights):
        x0 = x0.copy()
        if hasattr(self, "theta_output") and self.theta_output is not None:
            prev = self.theta_output[0]
        else:
            prev = self.theta_prev

        max_theta_step = 12.0
        theta_0 = self.look_theta.query_near_prev(
            x0[0],
            x0[1],
            prev,
            k_neighbors=20,
            forward_only=True,
            max_forward_step=max_theta_step,
        )
        self.theta_prev = theta_0

        if self.x0_opt is not None and np.any(self.x0_opt != 0):
            prev_yaw = self.x0_opt[3]
            diff = x0[3] - prev_yaw
            x0[3] -= np.round(diff / (2.0 * np.pi)) * 2.0 * np.pi

        p = np.concatenate([
            x0,
            [theta_0],
            param,
            weights
        ])
        if self.x0_opt is None or np.any(np.isnan(self.x0_opt)):
            self.x0_opt = self.get_initial_guess(x0, param, theta_0)
        else:
            idx_theta = self.n_states + self.n_controls
            theta_ws = self.x0_opt[idx_theta:idx_theta + self.n_theta].copy()
            if theta_ws.size > 0:
                k_lap = np.round((theta_0 - theta_ws[0]) / self.track_length)
                theta_shift = k_lap * self.track_length
                self.x0_opt[idx_theta:idx_theta + self.n_theta] = theta_ws + theta_shift

        try:
            sol = self.solver(
                x0=self.x0_opt,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p)

            solver_stats = self.solver.stats()
            is_successful = solver_stats['success']

            if is_successful:
                done = 0
                print("Solver succeeded with status: ", solver_stats['return_status'])
            else:
                done = 1
            x_opt = sol['x'].full().flatten()

            idx = 0
            states = x_opt[idx:idx + self.n_states].reshape((self.config.TK + 1, self.config.NXK)).T
            idx += self.n_states
            controls = x_opt[idx:idx + self.n_controls].reshape((self.config.TK, self.config.NU)).T
            idx += self.n_controls
            theta = x_opt[idx:idx + self.n_theta]
            yaw_offset = np.round(states[3, 0] / (2.0 * np.pi)) * 2.0 * np.pi
            for t in range(self.config.TK + 1):
                x_opt[t * self.config.NXK + 3] -= yaw_offset
            self.x0_opt = x_opt
            return controls, states, theta, done

        except Exception as e:
            done = 1
            controls = self._normalize_controls(u_his)
            states = np.tile(x0[:, None], (1, self.config.TK + 1))
            theta = np.full(self.n_theta, np.nan)
            return controls, states, theta, done

    def MPCC_Control(self, x0_full, param, weights):
        x0 = x0_full[:self.config.NXK]
        input_o, states_output, theta_output, done = self.mpc_prob_solve(x0, self.u_his, param, weights)
        if not np.any(np.isnan(states_output)):
            self.states_output = states_output
            self.input_o = input_o
            self.theta_output = theta_output
        else:
            raise ValueError("MPCC solver failed to find a valid solution.")

        self.u_his = self._normalize_controls(input_o)
        self.input_o = self._normalize_controls(self.input_o)
        u = self.input_o[:, 0]

        ref_path_x = np.zeros(self.config.TK + 1)
        ref_path_y = np.zeros(self.config.TK + 1)
        for t in range(self.config.TK + 1):
            theta_t = (self.theta_output[t] - self.theta_min) % self.track_length + self.theta_min
            ref_path_x[t], ref_path_y[t] = lookup_xy(theta_t)
        pred_x = states_output[0, :]
        pred_y = states_output[1, :]
        return u, ref_path_x, ref_path_y, pred_x, pred_y, done
