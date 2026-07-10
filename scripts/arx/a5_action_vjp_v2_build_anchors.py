"""Build trusted per-anchor matrices from A5 action-side VJP v2 rows."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _vec(value):
    return np.asarray(json.loads(value), dtype=np.float64)


def _rms(value):
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value))) if value.size else 0.0


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denom < 1e-12 else float(np.dot(a, b) / denom)


def _as_bool(value):
    return str(value).strip().lower() not in ("", "0", "false", "no", "none")


def _axis_index(name):
    if not name.startswith("joint") or not name.endswith("+"):
        return None
    try:
        return int(name[len("joint"):-1]) - 1
    except ValueError:
        return None


def _matrix_from_axes(rows):
    columns = {}
    for row in rows:
        index = _axis_index(row["direction"])
        if index is not None:
            columns[index] = _vec(row["linear_velocity_response"])
    if sorted(columns) != list(range(6)):
        return None
    return np.stack([columns[index] for index in range(6)], axis=1)


def _random_hold_metrics(rows, matrix):
    random_rows = [row for row in rows if row["direction"].startswith("random")]
    if not random_rows:
        return {
            "num_random": 0,
            "relative_rmse": None,
            "response_cosine": None,
            "y_relative_rmse": None,
            "y_cosine": None,
            "y_target_rms": 0.0,
        }
    directions = np.stack([_vec(row["direction_vec"]) for row in random_rows])
    target = np.stack([_vec(row["linear_velocity_response"]) for row in random_rows])
    pred = directions @ matrix.T
    return {
        "num_random": len(random_rows),
        "relative_rmse": _rms(pred - target) / (_rms(target) + 1e-12),
        "response_cosine": _cosine(pred, target),
        "y_relative_rmse": _rms(pred[:, 1] - target[:, 1]) / (_rms(target[:, 1]) + 1e-12),
        "y_cosine": _cosine(pred[:, 1], target[:, 1]),
        "y_target_rms": _rms(target[:, 1]),
    }


def _float_or_none(value):
    if value in (None, ""):
        return None
    return float(value)


def _close_epsilon(value, target):
    return abs(float(value) - float(target)) <= max(abs(float(target)) * 1e-8, 1e-12)


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-epsilon", type=float, default=0.01)
    parser.add_argument("--reference-epsilon", type=float, default=0.003)
    parser.add_argument("--min-y-vjp-norm", type=float, default=1e-5)
    parser.add_argument("--min-random-y-rms", type=float, default=1e-6)
    parser.add_argument("--min-hold-y-cosine", type=float, default=0.7)
    parser.add_argument("--min-cross-epsilon-y-cosine", type=float, default=0.7)
    parser.add_argument("--min-signature-equal-rate", type=float, default=0.8)
    parser.add_argument("--max-repeat-state-diff", type=float, default=1e-7)
    parser.add_argument("--allow-missing-contact-event", action="store_true")
    parser.add_argument("--allow-missing-reference", action="store_true")
    args = parser.parse_args()

    anchor_manifest = {}
    with args.anchors.open(newline="") as f:
        for row in csv.DictReader(f):
            anchor_manifest[int(row["anchor_id"])] = row
    grouped = defaultdict(list)
    if args.rows.exists() and args.rows.stat().st_size:
        with args.rows.open(newline="") as f:
            for row in csv.DictReader(f):
                if _as_bool(row.get("keep", True)):
                    grouped[int(row["anchor_id"])].append(row)

    output = []
    for anchor_id, anchor in sorted(anchor_manifest.items()):
        rows = grouped.get(anchor_id, [])
        by_epsilon = defaultdict(list)
        for row in rows:
            by_epsilon[float(row["epsilon"])].append(row)
        matrices = {}
        metrics = {}
        for epsilon, epsilon_rows in sorted(by_epsilon.items()):
            matrix = _matrix_from_axes(epsilon_rows)
            if matrix is None:
                continue
            matrices[epsilon] = matrix
            metrics[epsilon] = _random_hold_metrics(epsilon_rows, matrix)

        target_key = next(
            (epsilon for epsilon in matrices if _close_epsilon(epsilon, args.target_epsilon)), None
        )
        reference_key = next(
            (epsilon for epsilon in matrices if _close_epsilon(epsilon, args.reference_epsilon)), None
        )
        target_matrix = None if target_key is None else matrices[target_key]
        reference_matrix = None if reference_key is None else matrices[reference_key]
        target_metrics = {} if target_key is None else metrics[target_key]
        cross_y_cosine = (
            None
            if target_matrix is None or reference_matrix is None
            else _cosine(target_matrix[1], reference_matrix[1])
        )
        target_rows = [] if target_key is None else by_epsilon[target_key]
        nominal_contact_geometry_trace = (
            "[]"
            if not target_rows
            else target_rows[0].get("nominal_contact_geometry_trace", "[]")
        )
        signature_equal_rate = (
            0.0
            if not target_rows
            else sum(_as_bool(row["contact_signature_trace_equal"]) for row in target_rows)
            / len(target_rows)
        )
        y_vjp_norm = 0.0 if target_matrix is None else float(np.linalg.norm(target_matrix[1]))
        repeat_diff = float(anchor.get("repeat_state_max_abs") or float("inf"))
        arm_contact_events = int(float(anchor.get("arm_contact_events") or 0))
        reasons = []
        if anchor.get("status") != "ok":
            reasons.append(anchor.get("status") or "anchor_status")
        if arm_contact_events < 1 and not args.allow_missing_contact_event:
            reasons.append("no_arm_object_contact")
        if repeat_diff > args.max_repeat_state_diff:
            reasons.append("repeat_state_diff")
        if target_matrix is None:
            reasons.append("missing_target_matrix")
        if reference_matrix is None and not args.allow_missing_reference:
            reasons.append("missing_reference_matrix")
        if y_vjp_norm < args.min_y_vjp_norm:
            reasons.append("weak_y_vjp")
        if (target_metrics.get("y_target_rms") or 0.0) < args.min_random_y_rms:
            reasons.append("weak_random_y_signal")
        if (target_metrics.get("y_cosine") or -1.0) < args.min_hold_y_cosine:
            reasons.append("hold_y_cosine")
        if (
            not args.allow_missing_reference
            and (cross_y_cosine or -1.0) < args.min_cross_epsilon_y_cosine
        ):
            reasons.append("cross_epsilon_y_cosine")
        if signature_equal_rate < args.min_signature_equal_rate:
            reasons.append("contact_signature_switch")
        reasons = list(dict.fromkeys(reasons))

        output.append(
            {
                "anchor_id": anchor_id,
                "split": anchor["split"],
                "status": anchor.get("status"),
                "branch_mode": anchor.get("branch_mode", ""),
                "usable": not reasons,
                "gate_reasons": "ok" if not reasons else "|".join(reasons),
                "obj_pos": anchor["obj_pos"],
                "speed": anchor["speed"],
                "anchor_step": anchor["anchor_step"],
                "arm_contact_events": arm_contact_events,
                "repeat_state_max_abs": repeat_diff,
                "target_epsilon": args.target_epsilon,
                "reference_epsilon": args.reference_epsilon,
                "target_matrix": "" if target_matrix is None else json.dumps(target_matrix.tolist()),
                "reference_matrix": "" if reference_matrix is None else json.dumps(reference_matrix.tolist()),
                "y_vjp_norm": y_vjp_norm,
                "hold_relative_rmse": target_metrics.get("relative_rmse"),
                "hold_response_cosine": target_metrics.get("response_cosine"),
                "hold_y_relative_rmse": target_metrics.get("y_relative_rmse"),
                "hold_y_cosine": target_metrics.get("y_cosine"),
                "hold_y_target_rms": target_metrics.get("y_target_rms"),
                "cross_epsilon_y_cosine": cross_y_cosine,
                "contact_signature_equal_rate": signature_equal_rate,
                "anchor_object_state": anchor["anchor_object_state"],
                "anchor_arm_state": anchor["anchor_arm_state"],
                "anchor_contact": anchor["anchor_contact"],
                "nominal_object_state": anchor["nominal_object_state"],
                "nominal_contact_trace": anchor["nominal_contact_trace"],
                "nominal_contact_geometry_trace": nominal_contact_geometry_trace,
            }
        )

    split_counts = defaultdict(Counter)
    reason_counts = Counter()
    contact_reason_counts = Counter()
    contact_reason_combinations = Counter()
    for row in output:
        split_counts[row["split"]]["total"] += 1
        split_counts[row["split"]]["contact"] += int(row["arm_contact_events"] > 0)
        split_counts[row["split"]]["usable"] += int(row["usable"])
        if not row["usable"]:
            row_reasons = row["gate_reasons"].split("|")
            reason_counts.update(row_reasons)
            if row["arm_contact_events"] > 0:
                contact_reason_counts.update(row_reasons)
                contact_reason_combinations[row["gate_reasons"]] += 1
    summary = {
        "description": "A5 action-side VJP v2 trusted anchor build",
        "rows": str(args.rows),
        "anchors": str(args.anchors),
        "gates": {
            "target_epsilon": args.target_epsilon,
            "reference_epsilon": args.reference_epsilon,
            "min_y_vjp_norm": args.min_y_vjp_norm,
            "min_random_y_rms": args.min_random_y_rms,
            "min_hold_y_cosine": args.min_hold_y_cosine,
            "min_cross_epsilon_y_cosine": args.min_cross_epsilon_y_cosine,
            "min_signature_equal_rate": args.min_signature_equal_rate,
            "max_repeat_state_diff": args.max_repeat_state_diff,
            "allow_missing_contact_event": args.allow_missing_contact_event,
            "allow_missing_reference": args.allow_missing_reference,
        },
        "num_anchors": len(output),
        "num_contact_anchors": sum(int(row["arm_contact_events"] > 0) for row in output),
        "num_usable_anchors": sum(int(row["usable"]) for row in output),
        "split_counts": {key: dict(value) for key, value in sorted(split_counts.items())},
        "gate_reason_counts": dict(reason_counts),
        "contact_gate_reason_counts": dict(contact_reason_counts),
        "contact_gate_reason_combinations": dict(contact_reason_combinations),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "a5_action_vjp_v2_anchor_matrices.csv"
    json_path = args.out_dir / "a5_action_vjp_v2_anchor_summary.json"
    _write_csv(csv_path, output)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[a5-vjp-v2-build] anchors={summary['num_anchors']} "
        f"contact={summary['num_contact_anchors']} usable={summary['num_usable_anchors']}"
    )
    print(f"[a5-vjp-v2-build] split={summary['split_counts']}")
    print(f"[a5-vjp-v2-build] reasons={summary['gate_reason_counts']}")
    print(f"[a5-vjp-v2-build] contact_reasons={summary['contact_gate_reason_counts']}")
    print(f"[a5-vjp-v2-build] wrote {json_path}")
    print(f"[a5-vjp-v2-build] wrote {csv_path}")


if __name__ == "__main__":
    raise SystemExit(main())
