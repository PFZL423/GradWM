"""Join A5 restore-prefilter and replay-trusted action VJP matrices."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _as_bool(value):
    return str(value).strip().lower() not in ("", "0", "false", "no", "none")


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denom < 1e-12 else float(np.dot(a, b) / denom)


def _relative(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b * b)) + 1e-12))


def _load(path):
    with path.open(newline="") as f:
        return {int(row["anchor_id"]): row for row in csv.DictReader(f)}


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-y-cosine", type=float, default=0.9)
    parser.add_argument("--min-y-norm-ratio", type=float, default=0.25)
    parser.add_argument("--max-y-norm-ratio", type=float, default=4.0)
    args = parser.parse_args()

    restore = _load(args.restore)
    replay = _load(args.replay)
    output = []
    for anchor_id, replay_row in sorted(replay.items()):
        row = dict(replay_row)
        restore_row = restore.get(anchor_id)
        replay_usable = _as_bool(replay_row.get("usable"))
        restore_usable = restore_row is not None and _as_bool(restore_row.get("usable"))
        y_cosine = None
        matrix_cosine = None
        y_relative = None
        matrix_relative = None
        y_norm_ratio = None
        if restore_row is not None and replay_row.get("target_matrix") and restore_row.get("target_matrix"):
            replay_matrix = np.asarray(json.loads(replay_row["target_matrix"]), dtype=np.float64)
            restore_matrix = np.asarray(json.loads(restore_row["target_matrix"]), dtype=np.float64)
            y_cosine = _cosine(replay_matrix[1], restore_matrix[1])
            matrix_cosine = _cosine(replay_matrix, restore_matrix)
            y_relative = _relative(replay_matrix[1], restore_matrix[1])
            matrix_relative = _relative(replay_matrix, restore_matrix)
            y_norm_ratio = float(
                np.linalg.norm(replay_matrix[1]) / (np.linalg.norm(restore_matrix[1]) + 1e-12)
            )
        branch_consistent = (
            y_cosine is not None
            and y_cosine >= args.min_y_cosine
            and y_norm_ratio is not None
            and args.min_y_norm_ratio <= y_norm_ratio <= args.max_y_norm_ratio
        )
        final_usable = replay_usable and restore_usable and branch_consistent
        row.update(
            {
                "usable": final_usable,
                "replay_usable": replay_usable,
                "restore_usable": restore_usable,
                "branch_consistent": branch_consistent,
                "restore_replay_y_cosine": y_cosine,
                "restore_replay_matrix_cosine": matrix_cosine,
                "restore_replay_y_relative_rmse": y_relative,
                "restore_replay_matrix_relative_rmse": matrix_relative,
                "restore_replay_y_norm_ratio": y_norm_ratio,
                "restore_gate_reasons": (
                    "missing_restore" if restore_row is None else restore_row.get("gate_reasons", "")
                ),
            }
        )
        output.append(row)

    split_counts = defaultdict(Counter)
    for row in output:
        split_counts[row["split"]]["replay_total"] += 1
        split_counts[row["split"]]["replay_usable"] += int(row["replay_usable"])
        split_counts[row["split"]]["branch_consistent"] += int(row["branch_consistent"])
        split_counts[row["split"]]["final_usable"] += int(row["usable"])
    summary = {
        "description": "A5 action VJP v2 restore/replay branch comparison",
        "restore": str(args.restore),
        "replay": str(args.replay),
        "gates": {
            "min_y_cosine": args.min_y_cosine,
            "min_y_norm_ratio": args.min_y_norm_ratio,
            "max_y_norm_ratio": args.max_y_norm_ratio,
        },
        "num_replay": len(output),
        "num_final_usable": sum(int(row["usable"]) for row in output),
        "split_counts": {key: dict(value) for key, value in sorted(split_counts.items())},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "a5_action_vjp_v2_final_matrices.csv"
    json_path = args.out_dir / "a5_action_vjp_v2_branch_summary.json"
    _write_csv(csv_path, output)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[a5-vjp-v2-branches] replay={summary['num_replay']} "
        f"final_usable={summary['num_final_usable']}"
    )
    print(f"[a5-vjp-v2-branches] split={summary['split_counts']}")
    print(f"[a5-vjp-v2-branches] wrote {json_path}")
    print(f"[a5-vjp-v2-branches] wrote {csv_path}")


if __name__ == "__main__":
    raise SystemExit(main())
