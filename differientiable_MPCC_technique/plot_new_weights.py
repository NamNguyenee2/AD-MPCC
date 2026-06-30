import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_NAME = "friction0.9_outer_steps40_pg_iters1_lr0.2_clipped"
SAVE_CLIPPED = True
CLIPPED_SUFFIX = "_clipped"

base_dir = Path(__file__).resolve().parent
log_path = base_dir / LOG_NAME

with log_path.open("r", encoding="utf-8") as f:
    log = json.load(f)

q_theta_cur = np.asarray(log["q_theta_cur"], dtype=float)
q_theta_next_raw = np.asarray(log["q_contour_next"], dtype=float)
if q_theta_next_raw.shape != q_theta_cur.shape:
    raise ValueError(
        f"Shape mismatch: q_theta_next={q_theta_next_raw.shape}, "
        f"q_theta_cur={q_theta_cur.shape}"
    )

print("keys:", log.keys())

plt.plot(q_theta_next_raw, label="q_theta_next_raw", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
