"""Deeper diagnostic: for each A5 anchor, break down which components of the
9D response label the linear fit fails on.

If some components (e.g. position) are dominated by noise floor while others
(velocity) carry the true signal, the joint local-state RMSE is uninformative.
"""
import csv
import json
from pathlib import Path

import numpy as np

CSV = Path(
    "analysis/2026-07-09_arx_pusher/stage2_short1_eps3e3_shard1/a5_stage2_short1_eps3e3_shard1.csv"
)
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


def fit_component(v, y_col):
    coef, _, _, _ = np.linalg.lstsq(v, y_col, rcond=None)
    pred = v @ coef
    err = pred - y_col
    rmse = float(np.sqrt(np.mean(err ** 2)))
    signal_rms = float(np.sqrt(np.mean(y_col ** 2)))
    rel = rmse / (signal_rms + 1e-15)
    return rel, signal_rms, rmse


def main():
    rows = load(CSV)
    print(f"Loaded {len(rows)} rows. Response dim = {rows[0]['y'].shape[0]}")
    labels = ["px", "py", "pz", "vx", "vy", "vz", "wx", "wy", "wz"]

    for a in sorted({r["anchor_id"] for r in rows}):
        grp = [r for r in rows if r["anchor_id"] == a]
        v = np.stack([r["v"] for r in grp])
        y = np.stack([r["y"] for r in grp])
        print(f"\n=== anchor {a} (n={len(grp)}, restore_mm={grp[0]['restore_mm']:.2e}) ===")
        for j in range(y.shape[1]):
            rel, signal, rmse = fit_component(v, y[:, j])
            print(f"  {labels[j]:>4s}: signal_rms={signal:.3e}  fit_rmse={rmse:.3e}  rel_rmse={rel:.4f}")

        # also: which SVD rank does the response matrix have?
        coef, _, _, _ = np.linalg.lstsq(v, y, rcond=None)
        # coef is (6, 9)
        s = np.linalg.svd(coef, compute_uv=False)
        print(f"  A_full singular values: {s.tolist()}")

    # cross-anchor Jacobian similarity
    print("\n=== Per-anchor A matrix cosine similarity (flatten) ===")
    mats = {}
    for a in sorted({r["anchor_id"] for r in rows}):
        grp = [r for r in rows if r["anchor_id"] == a]
        v = np.stack([r["v"] for r in grp])
        y = np.stack([r["y"] for r in grp])
        coef, _, _, _ = np.linalg.lstsq(v, y, rcond=None)
        mats[a] = coef.flatten()
    ids = sorted(mats.keys())
    print("       " + "  ".join(f"a{a:<2d}" for a in ids))
    for a in ids:
        row = []
        for b in ids:
            u, v = mats[a], mats[b]
            cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-15))
            row.append(f"{cos:+.3f}")
        print(f"  a{a}: " + "  ".join(row))


if __name__ == "__main__":
    raise SystemExit(main())
