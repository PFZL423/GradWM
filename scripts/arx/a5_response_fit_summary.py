"""Summarize global vs per-anchor linear response fits."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_DATA = Path("analysis/2026-07-09_arx_pusher/a5_multi_anchor_fd_dataset_grid9.csv")
DEFAULT_OUT = Path("analysis/2026-07-09_arx_pusher/a5_response_fit_summary_grid9.json")


def _loads_vec(text):
    return np.asarray(json.loads(text), dtype=float)


def _load(path, target):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "anchor_id": int(row["anchor_id"]),
                    "direction": row["direction"],
                    "v": _loads_vec(row["direction_vec"]),
                    "target": _loads_vec(row[target]),
                }
            )
    return rows


def _metrics(pred, target):
    err = pred - target
    rmse = float(np.sqrt(np.mean(err ** 2)))
    target_rms = float(np.sqrt(np.mean(target ** 2)))
    return {
        "rmse": rmse,
        "target_rms": target_rms,
        "relative_rmse": rmse / (target_rms + 1e-12),
    }


def _fit(rows):
    x = np.stack([r["v"] for r in rows], axis=0)
    y = np.stack([r["target"] for r in rows], axis=0)
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    return {
        "num_rows": len(rows),
        "rank": int(rank),
        "metrics": _metrics(x @ coef, y),
        "A_object_by_action": coef.T.tolist(),
    }


def _mean(values):
    return float(sum(values) / len(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--target",
        choices=(
            "local_response",
            "final_response",
            "local_vel_response",
            "final_vel_response",
            "local_state_response",
            "final_state_response",
        ),
        default="final_response",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = _load(args.data, args.target)
    global_fit = _fit(rows)
    per_anchor = []
    for anchor_id in sorted({r["anchor_id"] for r in rows}):
        anchor_rows = [r for r in rows if r["anchor_id"] == anchor_id]
        result = _fit(anchor_rows)
        result["anchor_id"] = anchor_id
        per_anchor.append(result)

    payload = {
        "description": "Global vs per-anchor linear fit summary",
        "data": str(args.data),
        "target": args.target,
        "num_rows": len(rows),
        "num_anchors": len(per_anchor),
        "global_fit": global_fit,
        "per_anchor": per_anchor,
        "per_anchor_summary": {
            "relative_rmse_mean": _mean([p["metrics"]["relative_rmse"] for p in per_anchor]),
            "relative_rmse_min": min(p["metrics"]["relative_rmse"] for p in per_anchor),
            "relative_rmse_max": max(p["metrics"]["relative_rmse"] for p in per_anchor),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[fit-summary] target={args.target} global_rel={global_fit['metrics']['relative_rmse']:.4f} "
        f"per_anchor_mean={payload['per_anchor_summary']['relative_rmse_mean']:.4f}"
    )
    print(f"[fit-summary] wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
