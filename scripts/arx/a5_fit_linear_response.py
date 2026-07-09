"""Fit a single-anchor linear action-to-object response matrix from FD labels."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_DATA = Path("analysis/2026-07-09_arx_pusher/a5_fd_response_dataset_clean_anchor.csv")
DEFAULT_OUT = Path("analysis/2026-07-09_arx_pusher/a5_linear_response_fit_clean_anchor.json")


def _loads_vec(text):
    return np.asarray(json.loads(text), dtype=float)


def _read_rows(path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "direction": row["direction"],
                    "epsilon": float(row["epsilon"]),
                    "v": _loads_vec(row["direction_vec"]),
                    "local": _loads_vec(row["local_response"]),
                    "final": _loads_vec(row["final_response"]),
                }
            )
    return rows


def _fit(rows, target_name):
    x = np.stack([r["v"] for r in rows], axis=0)
    y = np.stack([r[target_name] for r in rows], axis=0)
    coef, residuals, rank, singular_values = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    err = pred - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    target_rms = float(np.sqrt(np.mean(y ** 2)))
    mae = float(np.mean(np.abs(err)))
    rel_rmse = rmse / (target_rms + 1e-12)
    return {
        "target": target_name,
        "matrix_shape": [3, 6],
        "A_object_by_action": coef.T.tolist(),
        "rank": int(rank),
        "singular_values": singular_values.tolist(),
        "residuals": residuals.tolist(),
        "rmse": rmse,
        "mae": mae,
        "target_rms": target_rms,
        "relative_rmse": rel_rmse,
        "row_errors": [
            {
                "direction": rows[i]["direction"],
                "epsilon": rows[i]["epsilon"],
                "target": y[i].tolist(),
                "pred": pred[i].tolist(),
                "error": err[i].tolist(),
                "error_norm": float(np.linalg.norm(err[i])),
                "target_norm": float(np.linalg.norm(y[i])),
            }
            for i in range(len(rows))
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epsilon", type=float, default=0.0, help="0 uses all epsilon values")
    args = parser.parse_args()

    rows = _read_rows(args.data)
    if args.epsilon:
        rows = [r for r in rows if abs(r["epsilon"] - args.epsilon) < 1e-12]
    if not rows:
        raise RuntimeError(f"no rows selected from {args.data}")

    payload = {
        "description": "Single-anchor least-squares action-to-object response fit",
        "data": str(args.data),
        "epsilon_filter": args.epsilon,
        "num_rows": len(rows),
        "local_fit": _fit(rows, "local"),
        "final_fit": _fit(rows, "final"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[a5-linear-fit] rows={len(rows)} wrote {args.out}")
    print(
        "[a5-linear-fit] local rel_rmse="
        f"{payload['local_fit']['relative_rmse']:.4f} final rel_rmse={payload['final_fit']['relative_rmse']:.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
