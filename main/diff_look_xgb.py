import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor
# from scipy.spatial import KDTree
import os

DIFF_DATA_DIR = os.path.join(os.path.dirname(__file__), "diff_data")

LOG_NAMES = [
    "friction0.6_outer_steps100_pg_iters10_lr0.1",
    "friction0.7_outer_steps100_pg_iters10_lr0.1",
    "friction0.8_outer_steps100_pg_iters10_lr0.1",
    "friction0.9_outer_steps100_pg_iters10_lr0.1",
    "friction1.0_outer_steps100_pg_iters10_lr0.1",
    "friction1.1_outer_steps100_pg_iters10_lr0.1",
    "friction1.2_outer_steps100_pg_iters10_lr0.1",
    "friction0.6_outer_steps100_pg_iters10_lr0.1_2",
    "friction0.7_outer_steps100_pg_iters10_lr0.1_2",
    "friction0.8_outer_steps100_pg_iters10_lr0.1_2",
    "friction0.9_outer_steps100_pg_iters10_lr0.1_2",
    "friction1.0_outer_steps100_pg_iters10_lr0.1_2",
    "friction1.1_outer_steps100_pg_iters10_lr0.1_2",
    "friction1.2_outer_steps100_pg_iters10_lr0.1_2",
]

LOG_PATHS = [os.path.join(DIFF_DATA_DIR, name) for name in LOG_NAMES]

LF = 0.88392
LR = 1.50876


def _load_arrays(log_path: str):
    with Path(log_path).open("r", encoding="utf-8") as f:
        log = json.load(f)

    theta = np.asarray(log["theta"], dtype=float)
    vx = np.asarray(log["vx"], dtype=float)
    vy = np.asarray(log["vy"], dtype=float)
    steering_angle = np.asarray(log["steer_angle"], dtype=float)
    yaw_rate = np.asarray(log["yaw_rate"], dtype=float)

    df = np.asarray(log["DF"], dtype=float)
    cf = np.asarray(log["CF"], dtype=float)
    bf = np.asarray(log["BF"], dtype=float)

    dr = np.asarray(log["DR"], dtype=float)
    cr = np.asarray(log["CR"], dtype=float)
    br = np.asarray(log["BR"], dtype=float)

    vx_safe = np.where(np.abs(vx) < 1e-3, np.sign(vx) * 1e-3 + (vx == 0) * 1e-3, vx)
    alfa_f = steering_angle - np.arctan2(yaw_rate * LF + vy, vx_safe)
    alfa_r = np.arctan2(yaw_rate * LR - vy, vx_safe)

    mu_f = df * np.sin(cf * np.arctan(bf * alfa_f))
    mu_r = dr * np.sin(cr * np.arctan(br * alfa_r))

    q_contour = np.asarray(log["q_contour_next"], dtype=float)
    q_lag = np.asarray(log["q_lag_next"], dtype=float)
    q_theta = np.asarray(log["q_theta_next"], dtype=float)

    if not (
        theta.size
        == vx.size
        == vy.size
        == mu_f.size
        == mu_r.size
        == q_contour.size
        == q_lag.size
        == q_theta.size
    ):
        raise ValueError(f"All vectors must have same length in {log_path}.")

    features = np.column_stack([theta, vx, vy, mu_f, mu_r])
    return features, q_contour, q_lag, q_theta


all_features = []
all_q_contour = []
all_q_lag = []
all_q_theta = []

for path in LOG_PATHS:
    features_i, q_contour_i, q_lag_i, q_theta_i = _load_arrays(path)
    all_features.append(features_i)
    all_q_contour.append(q_contour_i)
    all_q_lag.append(q_lag_i)
    all_q_theta.append(q_theta_i)

features = np.vstack(all_features)
q_contour = np.concatenate(all_q_contour)
q_lag = np.concatenate(all_q_lag)
q_theta = np.concatenate(all_q_theta)


def _build_regressor():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )


_model_q_contour = _build_regressor()
_model_q_lag = _build_regressor()
_model_q_theta = _build_regressor()

_model_q_contour.fit(features, q_contour)
_model_q_lag.fit(features, q_lag)
_model_q_theta.fit(features, q_theta)


def lookup_q(query, k_neighbors=8):
    q = np.asarray(query, dtype=float).reshape(-1)
    if q.size != 5:
        raise ValueError("query must have 5 elements: [theta, vx, vy, mu_F, mu_R].")

    # Keep signature stable for callers still passing k_neighbors.
    _ = k_neighbors
    feature_row = q.reshape(1, -1)

    qc = float(_model_q_contour.predict(feature_row)[0])
    ql = float(_model_q_lag.predict(feature_row)[0])
    qt = float(_model_q_theta.predict(feature_row)[0])
    return qc, ql, qt
