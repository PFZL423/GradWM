"""Quick restore-gate ablation on the existing A5 short-window CSV.

Filters the stage2_short1 dataset by `restored_nominal_local_state_max_abs`
and refits per-anchor + global linear response matrix. No new simulation is
run; this is pure post-hoc analysis of the CSV that already exists.
"""
import csv
import json
from pathlib import Path

import numpy as np

CSV = Path("analysis/2026-07-09_arx_pusher/stage2_short1_eps3e3_shard1/a5_stage2_short1_eps3e3_shard1.csv")
TARGET = "local_state_response"


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append(
            {
                "anchor_id": int(r["anchor_id"]),
                "v": np.asarray(json.loads(r["direction_vec"]), dtype=float),
                "y": np.asarray(json.loads(r[TARGET]), dtype=float),
                "restore_mm": float(r["restored_nominal_local_state_max_abs"]),
            }
        )
    return rows


def fit(rows):
    x = np.stack([r["v"] for r in rows])
    y = np.stack([r["y"] for r in rows])
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    err = pred - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    trms = float(np.sqrt(np.mean(y ** 2)))
    return rmse / (trms + 1e-12)


def summarize(rows, label):
    n_anchors = len({r["anchor_id"] for r in rows})
    print(f"\n=== {label} ({len(rows)} rows, {n_anchors} anchors) ===")
    if not rows:
        print("  (no rows)")
        return
    print(f"  global fit rel_rmse: {fit(rows):.4f}")
    per = []
    for a in sorted({r["anchor_id"] for r in rows}):
        grp = [r for r in rows if r["anchor_id"] == a]
        rel = fit(grp)
        per.append(rel)
        rm = grp[0]["restore_mm"]
        print(f"  anchor {a} (n={len(grp):3d}, restore_mm={rm:.2e}): rel_rmse = {rel:.4f}")
    print(f"  per_anchor_mean = {np.mean(per):.4f}")


def main():
    rows_all = load(CSV)
    print(f"Loaded {len(rows_all)} rows from {CSV.name}")
    summarize(rows_all, "BASELINE (no gate)")
    for gate in [1e-4, 1e-5, 1e-6]:
        kept = [r for r in rows_all if r["restore_mm"] <= gate]
        summarize(kept, f"GATED (restore_mm <= {gate:.0e})")


if __name__ == "__main__":
    raise SystemExit(main())
