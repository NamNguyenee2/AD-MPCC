import os
import time
from dataclasses import dataclass, field
import casadi as ca
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from numba import njit
import copy
from scipy.optimize import minimize_scalar
from scipy.interpolate import make_interp_spline

import pandas as pd
from scipy.spatial import KDTree

_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'main', 'scale0.25_Oschersleben_waypoints.csv')
data = pd.read_csv(_csv_path)
df = data[["X", "Y"]].copy()

x = df["X"].to_numpy()
y = df["Y"].to_numpy()


dx = np.diff(x)
dy = np.diff(y)
segment_lengths = np.sqrt(dx**2 + dy**2)

track_length = np.sum(segment_lengths)  # ✅ Actual arc length
theta = np.concatenate([[0], np.cumsum(segment_lengths)])

# Previous cubic interpolation (kept for reference):
# spline_x = CubicSpline(theta, x, bc_type='natural')
# spline_y = CubicSpline(theta, y, bc_type='natural')


spline_x = make_interp_spline(theta, x, k=1)
spline_y = make_interp_spline(theta, y, k=1)

theta_min = float(theta.min())
theta_max = float(theta.max())

# print("theta_min:", theta_min)
print("theta_max:", theta_max)


@njit
def _unwrap_theta_candidates(theta_candidates, theta_prev, track_length, forward_only):
    n = theta_candidates.size
    theta_cont = np.empty(n, dtype=np.float64)
    for i in range(n):
        k = np.round((theta_prev - theta_candidates[i]) / track_length)
        th = theta_candidates[i] + k * track_length
        if forward_only and th < theta_prev:
            th += track_length
        theta_cont[i] = th
    return theta_cont


@njit
def _argmin_feasible_score(theta_cont, dists, theta_prev, forward_only, has_cap, theta_cap):
    best_idx = -1
    best_score = 1e300
    for i in range(theta_cont.size):
        th = theta_cont[i]
        if forward_only and has_cap and th > theta_cap:
            continue
        score = dists[i] + 1e-6 * np.abs(th - theta_prev)
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx

class ThetaLookupTable:
    def __init__(self, spline_x, spline_y, theta_min, theta_max, n_samples=50000):
        # Dense sampling of the path
        self.spline_x = spline_x
        self.spline_y = spline_y
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.track_length = float(theta_max - theta_min)
        self.theta_samples = np.linspace(theta_min, theta_max, n_samples)
        self.x_samples = spline_x(self.theta_samples)
        self.y_samples = spline_y(self.theta_samples)
        
        # Build KD-tree for fast nearest neighbor search
        self.positions = np.column_stack([self.x_samples, self.y_samples])
        self.kdtree = KDTree(self.positions)
    
    # def query(self, x_query, y_query, k_neighbors=1):
    #     query_point = np.array([[x_query, y_query]])
    #     if k_neighbors == 1:
    #         # Simple nearest neighbor
    #         dist, idx = self.kdtree.query(query_point)
    #         return float(self.theta_samples[idx[0]])
    #     else:
    #         distances, indices = self.kdtree.query(query_point, k=k_neighbors)
    #         weights = 1.0 / (distances[0] + 1e-10)
    #         weights /= weights.sum()
    #         theta_weighted = np.sum(self.theta_samples[indices[0]] * weights)
    #         return float(theta_weighted)

    def query_near_prev(
        self,
        x_query,
        y_query,
        theta_prev,
        k_neighbors=50,
        forward_only=True,
        max_forward_step=None,
    ):
        """
        Continuous theta projection with continuity constraints:
        minimize geometric distance to the path while staying near theta_prev.
        """
        k_neighbors = max(1, int(k_neighbors))
        distances, indices = self.kdtree.query([x_query, y_query], k=k_neighbors)
        idx = np.atleast_1d(indices).astype(int)
        dists = np.atleast_1d(distances).astype(float)
        theta_candidates = np.asarray(self.theta_samples[idx], dtype=np.float64)
        dists = np.asarray(dists, dtype=np.float64)

        L = float(self.track_length)
        theta_cont = _unwrap_theta_candidates(theta_candidates, float(theta_prev), L, bool(forward_only))

        has_cap = bool(forward_only and max_forward_step is not None)
        theta_cap = float(theta_prev + float(max_forward_step)) if has_cap else 0.0
        best_idx = _argmin_feasible_score(
            theta_cont,
            dists,
            float(theta_prev),
            bool(forward_only),
            has_cap,
            theta_cap,
        )
        if best_idx < 0:
            # No candidate in window -> clamp progress.
            return float(theta_cap)

        seed = float(theta_cont[best_idx])

        # Local continuous refinement around seed.
        span = 2.0
        lo = seed - span
        hi = seed + span
        if forward_only:
            lo = max(lo, theta_prev)
            if max_forward_step is not None:
                hi = min(hi, theta_prev + float(max_forward_step))
        if hi <= lo:
            return float(seed)

        def dist2(th):
            xb = float(self.spline_x(th))
            yb = float(self.spline_y(th))
            dx = xb - x_query
            dy = yb - y_query
            return dx * dx + dy * dy

        res = minimize_scalar(dist2, bounds=(lo, hi), method='bounded')
        return float(res.x if res.success else seed)
    
    # def query_batch(self, x_queries, y_queries, k_neighbors=1):

    #     query_points = np.column_stack([x_queries, y_queries])
        
    #     if k_neighbors == 1:
    #         distances, indices = self.kdtree.query(query_points)
    #         return self.theta_samples[indices].astype(float)
    #     else:
    #         distances, indices = self.kdtree.query(query_points, k=k_neighbors)
    #         # Weighted average for each query point
    #         thetas = []
    #         for i in range(len(query_points)):
    #             weights = 1.0 / (distances[i] + 1e-10)
    #             weights /= weights.sum()
    #             theta_weighted = np.sum(self.theta_samples[indices[i]] * weights)
    #             thetas.append(theta_weighted)
    #         return np.array(thetas)
    
def lookup_xy(theta_query):
    return float(spline_x(theta_query)), float(spline_y(theta_query))

def lookup_phi(theta_query):
    dx = spline_x(theta_query, 1)
    dy = spline_y(theta_query, 1)
    return float(np.arctan2(dy, dx))
    
