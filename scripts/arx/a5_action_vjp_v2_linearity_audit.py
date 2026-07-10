"""Audit whether A5 velocity FD rows admit one linear action matrix per anchor."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_DATA = Path(
    "analysis/2026-07-09_arx_pusher/stage2_phase_grid_full132/"
    "a5_stage2_phase_grid_full132.csv"
)
DEFAULT_OUT_DIR = Path(
    "analysis/2026-07-09_arx_pusher/stage2_phase_grid_full132/action_vjp_v2_phase_a/linearity_audit"
)
ACTION_DIM = 6
TARGET_DIM = 3


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


def _trace_delta(row):
    plus = _vec(row["plus_contact_trace"])
    minus = _vec(row["minus_contact_trace"])
    if plus.size != minus.size:
        return float("inf"), float("inf"), False
    delta = np.abs(plus - minus)
    return float(np.mean(delta)), float(np.max(delta, initial=0.0)), bool(np.array_equal(plus, minus))


def _gates():
    return {
        "all": lambda row: True,
        "trace_equal": lambda row: _trace_delta(row)[2],
        "trace_mean_le_0p5_max_le_1": lambda row: _trace_delta(row)[0] <= 0.5
        and _trace_delta(row)[1] <= 1.0,
        "trace_mean_le_1_max_le_2": lambda row: _trace_delta(row)[0] <= 1.0
        and _trace_delta(row)[1] <= 2.0,
        "trace_mean_le_2_max_le_4": lambda row: _trace_delta(row)[0] <= 2.0
        and _trace_delta(row)[1] <= 4.0,
    }


def _fit_rows(rows):
    v = np.stack([_vec(row["direction_vec"], ACTION_DIM) for row in rows])
    y = np.stack([_vec(row["local_vel_response"], 6)[:TARGET_DIM] for row in rows])
    coef, _, rank, singular = np.linalg.lstsq(v, y, rcond=None)
    pred = v @ coef
    condition = float(singular[0] / max(singular[-1], 1e-12)) if singular.size else float("inf")
    return coef, int(rank), condition, _rms(pred - y) / (_rms(y) + 1e-12)


def _audit_anchor(anchor_id, rows, gate_name, gate, fit_random_max, min_hold_rows):
    selected = [row for row in rows if gate(row)]
    fit_rows = []
    hold_rows = []
    for row in selected:
        random_idx = _random_index(row["direction"])
        if random_idx is not None and random_idx > fit_random_max:
            hold_rows.append(row)
        else:
            fit_rows.append(row)

    result = {
        "anchor_id": anchor_id,
        "gate": gate_name,
        "obj_y": float(rows[0]["obj_y"]),
        "speed": float(rows[0]["speed"]),
        "anchor_step": int(rows[0]["anchor_step"]),
        "num_rows_total": len(rows),
        "num_rows_selected": len(selected),
        "num_fit_rows": len(fit_rows),
        "num_hold_rows": len(hold_rows),
        "rank_all": 0,
        "rank_fit": 0,
        "condition_all": None,
        "condition_fit": None,
        "all_fit_relative_rmse": None,
        "fit_relative_rmse": None,
        "hold_relative_rmse": None,
        "hold_cosine": None,
        "signal_rms_y": None,
        "status": "insufficient_rows",
    }
    if len(selected) < ACTION_DIM:
        return result
    coef_all, rank_all, condition_all, all_rel = _fit_rows(selected)
    result.update(
        {
            "rank_all": rank_all,
            "condition_all": condition_all,
            "all_fit_relative_rmse": all_rel,
        }
    )
    selected_y = np.stack([_vec(row["local_vel_response"], 6)[:TARGET_DIM] for row in selected])
    result["signal_rms_y"] = _rms(selected_y[:, 1])
    if len(fit_rows) < ACTION_DIM or len(hold_rows) < min_hold_rows:
        return result
    coef, rank, condition, fit_rel = _fit_rows(fit_rows)
    result.update({"rank_fit": rank, "condition_fit": condition, "fit_relative_rmse": fit_rel})
    if rank < ACTION_DIM:
        result["status"] = "rank_deficient"
        return result
    hold_v = np.stack([_vec(row["direction_vec"], ACTION_DIM) for row in hold_rows])
    hold_y = np.stack([_vec(row["local_vel_response"], 6)[:TARGET_DIM] for row in hold_rows])
    hold_pred = hold_v @ coef
    result.update(
        {
            "hold_relative_rmse": _rms(hold_pred - hold_y) / (_rms(hold_y) + 1e-12),
            "hold_cosine": _cosine(hold_pred, hold_y),
            "status": "ok",
        }
    )
    return result


def _finite(rows, key):
    output = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            output.append(value)
    return output


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "median": None, "mean": None, "p25": None, "p75": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
    }


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fit-random-max", type=int, default=16)
    parser.add_argument("--min-hold-rows", type=int, default=6)
    args = parser.parse_args()

    grouped = defaultdict(list)
    with args.data.open(newline="") as f:
        for row in csv.DictReader(f):
            grouped[int(row["anchor_id"])].append(row)

    rows = []
    summary = {}
    for gate_name, gate in _gates().items():
        gate_rows = [
            _audit_anchor(anchor_id, anchor_rows, gate_name, gate, args.fit_random_max, args.min_hold_rows)
            for anchor_id, anchor_rows in sorted(grouped.items())
        ]
        rows.extend(gate_rows)
        ok = [row for row in gate_rows if row["status"] == "ok"]
        full_rank_all = [row for row in gate_rows if row["rank_all"] == ACTION_DIM]
        hold_cosines = _finite(ok, "hold_cosine")
        summary[gate_name] = {
            "num_anchors": len(gate_rows),
            "num_rows_selected": int(sum(row["num_rows_selected"] for row in gate_rows)),
            "num_full_rank_all": len(full_rank_all),
            "num_ok_with_holdout": len(ok),
            "all_fit_relative_rmse": _stats(_finite(full_rank_all, "all_fit_relative_rmse")),
            "fit_relative_rmse": _stats(_finite(ok, "fit_relative_rmse")),
            "hold_relative_rmse": _stats(_finite(ok, "hold_relative_rmse")),
            "hold_cosine": _stats(hold_cosines),
            "hold_cosine_gt_0": sum(value > 0.0 for value in hold_cosines),
            "hold_cosine_ge_0p7": sum(value >= 0.7 for value in hold_cosines),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "a5_action_vjp_v2_linearity_audit.csv"
    json_path = args.out_dir / "a5_action_vjp_v2_linearity_audit.json"
    _write_csv(csv_path, rows)
    payload = {
        "description": "A5 action-side linear-velocity matrix consistency audit",
        "data": str(args.data),
        "target": "local_vel_response[:3]",
        "fit_random_max": args.fit_random_max,
        "min_hold_rows": args.min_hold_rows,
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    for gate_name, metrics in summary.items():
        print(
            f"[a5-vjp-v2-linearity] {gate_name:31s} rows={metrics['num_rows_selected']:4d} "
            f"rank6={metrics['num_full_rank_all']:2d} ok={metrics['num_ok_with_holdout']:2d} "
            f"all_rel_med={metrics['all_fit_relative_rmse']['median']} "
            f"hold_cos_med={metrics['hold_cosine']['median']} "
            f"cos>=0.7={metrics['hold_cosine_ge_0p7']}"
        )
    print(f"[a5-vjp-v2-linearity] wrote {json_path}")
    print(f"[a5-vjp-v2-linearity] wrote {csv_path}")


if __name__ == "__main__":
    raise SystemExit(main())
