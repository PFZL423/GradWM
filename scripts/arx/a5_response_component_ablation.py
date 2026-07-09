"""Ablate which subset of the 9D response label gives the cleanest linear fit.

Tests hypothesis: on A5's mesh-contact, angular velocity components are
inherently nonlinear, dragging the joint local-state RMSE up. Position and
linear velocity components should fit much better in isolation.
"""
import csv
import json
from pathlib import Path

import numpy as np

CSV = Path(
    "analysis/2026-07-09_arx_pusher/stage2_short1_eps3e3_shard1/a5_stage2_short1_eps3e3_shard1.csv"
)


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append(
            {
                "anchor_id": int(r["anchor_id"]),
                "v": np.asarray(json.loads(r["direction_vec"]), dtype=float),
                "y_full": np.asarray(json.loads(r["local_state_response"]), dtype=float),
                "restore_mm": float(r["restored_nominal_local_state_max_abs"]),
            }
        )
    return rows


def fit_rel(v, y):
    coef, _, _, _ = np.linalg.lstsq(v, y, rcond=None)
    err = v @ coef - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    trms = float(np.sqrt(np.mean(y ** 2)))
    return rmse / (trms + 1e-15)


def summarize(rows, mask, mask_label):
    print(f"\n=== label = {mask_label} (dim={mask.sum()}) ===")
    per = []
    for a in sorted({r["anchor_id"] for r in rows}):
        grp = [r for r in rows if r["anchor_id"] == a]
        v = np.stack([r["v"] for r in grp])
        y = np.stack([r["y_full"][mask] for r in grp])
        rel = fit_rel(v, y)
        per.append(rel)
        print(f"  anchor {a}: rel_rmse = {rel:.4f}")
    print(f"  per_anchor_mean = {np.mean(per):.4f}")


def main():
    rows = load(CSV)
    print(f"Loaded {len(rows)} rows.")
    # 9D layout: [px, py, pz, vx, vy, vz, wx, wy, wz]
    variants = {
        "9D full state": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], bool),
        "6D pos + lin vel (exclude angvel)": np.array([1, 1, 1, 1, 1, 1, 0, 0, 0], bool),
        "3D linear velocity only": np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], bool),
        "3D position only": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], bool),
        "3D angular velocity only": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], bool),
        "1D vy only (push direction)": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], bool),
    }
    for label, mask in variants.items():
        summarize(rows, mask, label)


if __name__ == "__main__":
    raise SystemExit(main())
