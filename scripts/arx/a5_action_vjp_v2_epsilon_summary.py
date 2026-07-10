"""Summarize linear-velocity action matrices across FD epsilon values."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _vec(value, length=None):
    output = np.asarray(json.loads(value), dtype=np.float64)
    return output if length is None else output[:length]


def _rms(value):
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value))) if value.size else 0.0


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denom < 1e-12 else float(np.dot(a, b) / denom)


def _random_index(name):
    if not name.startswith("random"):
        return None
    try:
        return int(name[len("random"):])
    except ValueError:
        return None


def _fit(rows):
    v = np.stack([_vec(row["direction_vec"], 6) for row in rows])
    y = np.stack([_vec(row["local_vel_response"], 6)[:3] for row in rows])
    coef, _, rank, singular = np.linalg.lstsq(v, y, rcond=None)
    pred = v @ coef
    return {
        "coef": coef,
        "rank": int(rank),
        "condition": float(singular[0] / max(singular[-1], 1e-12)),
        "relative_rmse": _rms(pred - y) / (_rms(y) + 1e-12),
        "y_relative_rmse": _rms(pred[:, 1] - y[:, 1]) / (_rms(y[:, 1]) + 1e-12),
    }


def _summarize_file(path, fit_random_max):
    grouped = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            grouped[float(row["epsilon"])].append(row)
    epsilon_rows = []
    for epsilon, rows in sorted(grouped.items()):
        fit_rows = []
        hold_rows = []
        for row in rows:
            random_idx = _random_index(row["direction"])
            if random_idx is not None and random_idx > fit_random_max:
                hold_rows.append(row)
            else:
                fit_rows.append(row)
        fit = _fit(fit_rows)
        coef = fit.pop("coef")
        hold_v = np.stack([_vec(row["direction_vec"], 6) for row in hold_rows])
        hold_y = np.stack([_vec(row["local_vel_response"], 6)[:3] for row in hold_rows])
        hold_pred = hold_v @ coef
        all_y = np.stack([_vec(row["local_vel_response"], 6)[:3] for row in rows])
        epsilon_rows.append(
            {
                "epsilon": epsilon,
                "num_rows": len(rows),
                "num_fit_rows": len(fit_rows),
                "num_hold_rows": len(hold_rows),
                **fit,
                "hold_relative_rmse": _rms(hold_pred - hold_y) / (_rms(hold_y) + 1e-12),
                "hold_cosine": _cosine(hold_pred, hold_y),
                "hold_y_relative_rmse": _rms(hold_pred[:, 1] - hold_y[:, 1])
                / (_rms(hold_y[:, 1]) + 1e-12),
                "hold_y_cosine": _cosine(hold_pred[:, 1], hold_y[:, 1]),
                "signal_rms": np.sqrt(np.mean(all_y * all_y, axis=0)).tolist(),
                "matrix": coef.T.tolist(),
                "y_vjp": coef[:, 1].tolist(),
                "y_vjp_norm": float(np.linalg.norm(coef[:, 1])),
            }
        )

    cross = []
    for i, left in enumerate(epsilon_rows):
        for right in epsilon_rows[i + 1:]:
            cross.append(
                {
                    "epsilon_a": left["epsilon"],
                    "epsilon_b": right["epsilon"],
                    "y_vjp_cosine": _cosine(left["y_vjp"], right["y_vjp"]),
                    "y_vjp_norm_ratio": left["y_vjp_norm"] / max(right["y_vjp_norm"], 1e-12),
                }
            )
    return {"csv": str(path), "epsilon_rows": epsilon_rows, "cross_epsilon": cross}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--fit-random-max", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = [_summarize_file(path, args.fit_random_max) for path in args.csv]
    payload = {
        "description": "A5 action-side linear-velocity epsilon audit",
        "target": "local_vel_response[:3]",
        "fit_random_max": args.fit_random_max,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    for result in results:
        print(f"[a5-vjp-v2-epsilon] {result['csv']}")
        for row in result["epsilon_rows"]:
            print(
                f"  eps={row['epsilon']:.1e} rank={row['rank']} "
                f"fit_y_rel={row['y_relative_rmse']:.3f} "
                f"hold_y_rel={row['hold_y_relative_rmse']:.3f} "
                f"hold_y_cos={row['hold_y_cosine']} "
                f"signal_y={row['signal_rms'][1]:.3e}"
            )
        print("  cross-epsilon y-VJP:")
        for row in result["cross_epsilon"]:
            print(
                f"    {row['epsilon_a']:.1e} vs {row['epsilon_b']:.1e}: "
                f"cos={row['y_vjp_cosine']} norm_ratio={row['y_vjp_norm_ratio']:.3e}"
            )
    print(f"[a5-vjp-v2-epsilon] wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
