"""Sweep response window / epsilon for the simple pusher FD labels.

This orchestrates many calls to ``pusher_restore_response_dataset.py`` and
summarizes whether each label is locally linear:

    local_response       object position response
    local_vel_response   object qvel response
    local_state_response position + qvel response

The script intentionally runs each condition in a subprocess, because Genesis
is much happier with one scene build per process.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_SCRIPT = REPO_ROOT / "scripts" / "pusher" / "pusher_restore_response_dataset.py"
DEFAULT_OUT_DIR = Path("analysis/2026-07-09_arx_pusher/simple_pusher_window_sweep")
TARGETS = ("local_response", "local_vel_response", "local_state_response")


def _parse_list(text, cast=str):
    return [cast(x) for x in text.split(",") if x.strip()]


def _slug(text):
    return (
        text.replace("+", "plus")
        .replace("-", "minus")
        .replace(".", "p")
        .replace(",", "_")
    )


def _loads_vec(text):
    return np.asarray(json.loads(text), dtype=np.float64)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "none", "")


def _metrics(pred, target):
    err = pred - target
    rmse = float(np.sqrt(np.mean(err ** 2)))
    target_rms = float(np.sqrt(np.mean(target ** 2)))
    return {
        "rmse": rmse,
        "target_rms": target_rms,
        "relative_rmse": rmse / (target_rms + 1e-12),
    }


def _fit_metric(rows, target, require_keep):
    items = []
    for row in rows:
        if require_keep and not _as_bool(row.get("keep", True)):
            continue
        items.append((row["direction_vec"], row[target]))
    if len(items) < 2:
        return {
            "num_fit_rows": len(items),
            "rank": 0,
            "rmse": float("nan"),
            "target_rms": float("nan"),
            "relative_rmse": float("nan"),
        }
    x = np.stack([_loads_vec(v) for v, _ in items], axis=0)
    y = np.stack([_loads_vec(t) for _, t in items], axis=0)
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    metric = _metrics(x @ coef, y)
    return {"num_fit_rows": len(items), "rank": int(rank), **metric}


def _norm_stats(rows):
    values = np.asarray([float(row["local_state_response_norm"]) for row in rows], dtype=np.float64)
    if values.size == 0:
        return {
            "state_norm_min": float("nan"),
            "state_norm_median": float("nan"),
            "state_norm_p90": float("nan"),
            "state_norm_max": float("nan"),
            "state_norm_max_over_median": float("nan"),
        }
    med = float(np.quantile(values, 0.5))
    mx = float(np.max(values))
    return {
        "state_norm_min": float(np.min(values)),
        "state_norm_median": med,
        "state_norm_p90": float(np.quantile(values, 0.9)),
        "state_norm_max": mx,
        "state_norm_max_over_median": mx / (med + 1e-12),
    }


def _read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _tail(text, n=10):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _run_condition(args, program, response_steps, eps):
    stem = (
        f"{args.tag}_prog{_slug(program)}"
        f"_resp{response_steps}_eps{_slug(f'{eps:.1e}')}"
    )
    json_path = args.out_dir / f"{stem}.json"
    csv_path = args.out_dir / f"{stem}.csv"
    cmd = [
        sys.executable,
        str(DATASET_SCRIPT),
        "--program",
        program,
        "--speed",
        str(args.speed),
        "--close-speed",
        str(args.close_speed),
        "--settle-steps",
        str(args.settle_steps),
        "--close-steps",
        str(args.close_steps),
        "--push-steps",
        str(args.push_steps),
        "--response-steps",
        str(response_steps),
        "--scales",
        f"{eps:.12g}",
        "--num-random",
        str(args.num_random),
        "--seed",
        str(args.seed),
        "--max-response-norm",
        str(args.max_response_norm),
        "--json",
        str(json_path),
        "--csv",
        str(csv_path),
    ]
    if args.drop_filtered:
        cmd.append("--drop-filtered")

    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    elapsed = time.perf_counter() - start
    record = {
        "program": program,
        "response_steps": response_steps,
        "epsilon": eps,
        "status": "ok" if proc.returncode == 0 and csv_path.exists() else f"error:{proc.returncode}",
        "elapsed_seconds": elapsed,
        "json": str(json_path),
        "csv": str(csv_path),
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }
    if record["status"] != "ok":
        return record

    payload = json.loads(json_path.read_text())
    rows = _read_rows(csv_path)
    record.update(
        {
            "anchor_keep": payload.get("anchor_keep"),
            "anchor_filter_reason": payload.get("anchor_filter_reason"),
            "anchor_step": payload.get("anchor_step"),
            "horizontal_disp": payload.get("horizontal_disp"),
            "vertical_disp": payload.get("vertical_disp"),
            "restore_diff": payload.get("restored_nominal_local_state_max_abs"),
            "repeat_diff": payload.get("repeat_local_state_max_abs"),
            "num_rows": len(rows),
            "num_kept_rows": sum(1 for row in rows if _as_bool(row.get("keep", True))),
            **_norm_stats(rows),
        }
    )
    for target in TARGETS:
        metric = _fit_metric(rows, target, require_keep=args.fit_kept_only)
        for key, value in metric.items():
            record[f"{target}_{key}"] = value
    return record


def _write_summary_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _jsonable_args(args):
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="simple_pusher_window_sweep")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--programs", default="j2j4j6+")
    parser.add_argument("--response-steps-list", default="1,2,5,10")
    parser.add_argument("--eps-list", default="1e-4,3e-4,1e-3,3e-3")
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--close-speed", type=float, default=1.5)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--push-steps", type=int, default=80)
    parser.add_argument("--num-random", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-response-norm", type=float, default=1e9)
    parser.add_argument("--drop-filtered", action="store_true")
    parser.add_argument("--fit-kept-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    programs = _parse_list(args.programs)
    response_steps_list = _parse_list(args.response_steps_list, int)
    eps_list = _parse_list(args.eps_list, float)

    records = []
    for program in programs:
        for response_steps in response_steps_list:
            for eps in eps_list:
                record = _run_condition(args, program, response_steps, eps)
                records.append(record)
                rel_state = record.get("local_state_response_relative_rmse")
                rel_vel = record.get("local_vel_response_relative_rmse")
                spread = record.get("state_norm_max_over_median")
                print(
                    f"[pusher-sweep] program={program} resp={response_steps} eps={eps:.1e} "
                    f"status={record['status']} rows={record.get('num_rows')} "
                    f"state_rel={rel_state} vel_rel={rel_vel} spread={spread}",
                    flush=True,
                )

    summary_csv = args.out_dir / f"{args.tag}_summary.csv"
    summary_json = args.out_dir / f"{args.tag}_summary.json"
    _write_summary_csv(summary_csv, records)
    summary_json.write_text(
        json.dumps(
            {
                "description": "Simple pusher response-window/epsilon linearity sweep",
                "args": _jsonable_args(args),
                "num_records": len(records),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[pusher-sweep] wrote {summary_csv} and {summary_json}")


if __name__ == "__main__":
    raise SystemExit(main())
