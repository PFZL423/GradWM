"""Collect and aggregate finite-scale FD response labels across A5 anchors."""
import argparse
import concurrent.futures
import csv
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_SCRIPT = REPO_ROOT / "scripts" / "arx" / "a5_fd_response_dataset.py"
RESTORE_DATASET_SCRIPT = REPO_ROOT / "scripts" / "arx" / "a5_restore_response_dataset.py"
DEFAULT_OUT = Path("analysis/2026-07-09_arx_pusher/a5_multi_anchor_fd_dataset.json")
DEFAULT_CSV = Path("analysis/2026-07-09_arx_pusher/a5_multi_anchor_fd_dataset.csv")
DEFAULT_RUN_DIR = Path("analysis/2026-07-09_arx_pusher/a5_multi_anchor_runs")
DEFAULT_QPOS = "0.0,1.4,-0.4,0.5,0.0,0.0"


def _parse_float_list(text):
    return [float(x) for x in text.split(",") if x.strip()]


def _qvel(speed):
    return f"{speed},0.0,0.0,0.0,0.0,0.0"


def _tail(text, n=8):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _append_filter_reason(old_reason, new_reason):
    if not old_reason or old_reason == "ok":
        return new_reason
    parts = old_reason.split("|")
    if new_reason in parts:
        return old_reason
    return f"{old_reason}|{new_reason}"


def _run_anchor(args, anchor_id, obj_y, speed):
    stem = f"anchor_{anchor_id:03d}_y{obj_y:.4f}_s{speed:.2f}"
    json_path = args.run_dir / f"{stem}.json"
    csv_path = args.run_dir / f"{stem}.csv"
    dataset_script = RESTORE_DATASET_SCRIPT if args.sampler == "restore" else DATASET_SCRIPT
    cmd = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "--no-capture-output",
        "python",
        str(dataset_script),
        "--obj-x",
        f"{args.obj_x:.6f}",
        "--obj-y",
        f"{obj_y:.6f}",
        "--obj-z",
        f"{args.obj_z:.6f}",
        "--qpos",
        args.qpos,
        "--qvel",
        _qvel(speed),
        "--settle-steps",
        str(args.settle_steps),
        "--push-steps",
        str(args.push_steps),
        "--response-steps",
        str(args.response_steps),
        "--contact-window",
        str(args.contact_window),
        "--max-contact-mean-delta",
        str(args.max_contact_mean_delta),
        "--max-contact-max-delta",
        str(args.max_contact_max_delta),
        "--min-response-norm",
        str(args.min_response_norm),
        "--max-response-norm",
        str(args.max_response_norm),
        "--scales",
        args.scales,
        "--num-random",
        str(args.num_random),
        "--seed",
        str(args.seed + anchor_id),
        "--json",
        str(json_path),
        "--csv",
        str(csv_path),
    ]
    if args.drop_filtered:
        cmd.append("--drop-filtered")
    if args.sampler == "restore" and args.collect_final:
        cmd.append("--collect-final")
    if args.no_requires_grad:
        cmd.append("--no-requires-grad")
    if args.sampler == "restore":
        cmd.extend(
            [
                "--max-restore-local-state-diff",
                str(args.max_restore_local_state_diff),
                "--max-repeat-local-state-diff",
                str(args.max_repeat_local_state_diff),
                "--min-horizontal-disp",
                str(args.min_horizontal_disp),
                "--max-abs-vertical-disp",
                str(args.max_abs_vertical_disp),
            ]
        )
        if args.drop_bad_anchor:
            cmd.append("--drop-bad-anchor")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    if proc.returncode != 0 or not json_path.exists():
        return {
            "anchor_id": anchor_id,
            "status": f"error:{proc.returncode}",
            "obj_y": obj_y,
            "speed": speed,
            "json": str(json_path),
            "csv": str(csv_path),
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
            "rows": [],
        }

    payload = json.loads(json_path.read_text())
    anchor_keep = payload.get("anchor_keep", True)
    anchor_filter_reason = payload.get("anchor_filter_reason", "ok")
    rows = []
    for row_idx, row in enumerate(payload["rows"]):
        rows.append(
            {
                "anchor_id": anchor_id,
                "row_idx": row_idx,
                "obj_x": args.obj_x,
                "obj_y": obj_y,
                "obj_z": args.obj_z,
                "speed": speed,
                "qpos": payload["qpos"],
                "qvel": payload["qvel"],
                "anchor_step": payload["anchor_step"],
                "response_steps": payload["response_steps"],
                "horizontal_disp": payload["horizontal_disp"],
                "vertical_disp": payload["vertical_disp"],
                "anchor_keep": anchor_keep,
                "anchor_filter_reason": anchor_filter_reason,
                "restored_nominal_local_state_max_abs": payload.get("restored_nominal_local_state_max_abs", 0.0),
                "repeat_local_state_max_abs": (payload.get("repeat_check") or {}).get("local_state_max_abs", 0.0),
                "direction": row["direction"],
                "epsilon": row["epsilon"],
                "direction_vec": row["direction_vec"],
                "local_response": row["local_response"],
                "final_response": row["final_response"],
                "local_vel_response": row.get("local_vel_response", []),
                "final_vel_response": row.get("final_vel_response", []),
                "local_state_response": row.get("local_state_response", []),
                "final_state_response": row.get("final_state_response", []),
                "local_response_norm": row["local_response_norm"],
                "final_response_norm": row["final_response_norm"],
                "local_vel_response_norm": row.get("local_vel_response_norm", 0.0),
                "final_vel_response_norm": row.get("final_vel_response_norm", 0.0),
                "local_state_response_norm": row.get("local_state_response_norm", 0.0),
                "final_state_response_norm": row.get("final_state_response_norm", 0.0),
                "local_qvel_plus": row.get("local_qvel_plus", []),
                "local_qvel_minus": row.get("local_qvel_minus", []),
                "final_qvel_plus": row.get("final_qvel_plus", []),
                "final_qvel_minus": row.get("final_qvel_minus", []),
                "keep": row.get("keep", True),
                "filter_reason": row.get("filter_reason", "ok"),
                "contact_mean_delta": row.get("contact_mean_delta", 0.0),
                "contact_max_delta": row.get("contact_max_delta", 0.0),
                "plus_contact_mean": row.get("plus_contact_mean", 0.0),
                "minus_contact_mean": row.get("minus_contact_mean", 0.0),
                "plus_contact_max": row.get("plus_contact_max", 0),
                "minus_contact_max": row.get("minus_contact_max", 0),
                "plus_contact_trace": row.get("plus_contact_trace", []),
                "minus_contact_trace": row.get("minus_contact_trace", []),
                "nominal_anchor_pos": row.get("nominal_anchor_pos", [0.0, 0.0, 0.0]),
                "nominal_pre_pos": row.get("nominal_pre_pos", [0.0, 0.0, 0.0]),
                "nominal_post_pos": row.get("nominal_post_pos", [0.0, 0.0, 0.0]),
                "nominal_initial_pos": row.get("nominal_initial_pos", [0.0, 0.0, 0.0]),
                "nominal_final_pos": row.get("nominal_final_pos", [0.0, 0.0, 0.0]),
                "nominal_anchor_qvel": row.get("nominal_anchor_qvel", [0.0] * 6),
                "nominal_pre_qvel": row.get("nominal_pre_qvel", [0.0] * 6),
                "nominal_post_qvel": row.get("nominal_post_qvel", [0.0] * 6),
                "nominal_initial_qvel": row.get("nominal_initial_qvel", [0.0] * 6),
                "nominal_final_qvel": row.get("nominal_final_qvel", [0.0] * 6),
                "nominal_pre_disp": row.get("nominal_pre_disp", [0.0, 0.0, 0.0]),
                "nominal_post_disp": row.get("nominal_post_disp", [0.0, 0.0, 0.0]),
                "nominal_total_disp": row.get("nominal_total_disp", [0.0, 0.0, 0.0]),
                "nominal_pre_qvel_delta": row.get("nominal_pre_qvel_delta", [0.0] * 6),
                "nominal_post_qvel_delta": row.get("nominal_post_qvel_delta", [0.0] * 6),
                "nominal_total_qvel_delta": row.get("nominal_total_qvel_delta", [0.0] * 6),
                "nominal_contact_mean": row.get("nominal_contact_mean", 0.0),
                "nominal_contact_max": row.get("nominal_contact_max", 0),
                "nominal_contact_min": row.get("nominal_contact_min", 0),
                "nominal_contact_trace": row.get("nominal_contact_trace", []),
                "max_total_contact_count_plus": row["max_total_contact_count_plus"],
                "max_total_contact_count_minus": row["max_total_contact_count_minus"],
            }
        )
    rows_before_anchor_gate = len(rows)
    kept_rows_before_anchor_gate = sum(1 for row in rows if row.get("keep", True))
    if (
        anchor_keep
        and args.min_kept_rows_per_anchor > 0
        and kept_rows_before_anchor_gate < args.min_kept_rows_per_anchor
    ):
        anchor_keep = False
        anchor_filter_reason = _append_filter_reason(
            anchor_filter_reason, "too_few_kept_rows"
        )
        for row in rows:
            row["anchor_keep"] = anchor_keep
            row["anchor_filter_reason"] = anchor_filter_reason
        if args.drop_bad_anchor:
            rows = []

    return {
        "anchor_id": anchor_id,
        "status": "ok",
        "obj_y": obj_y,
        "speed": speed,
        "sampler": payload.get("sampler", args.sampler),
        "json": str(json_path),
        "csv": str(csv_path),
        "anchor_step": payload["anchor_step"],
        "horizontal_disp": payload["horizontal_disp"],
        "vertical_disp": payload["vertical_disp"],
        "num_rows": len(rows),
        "num_rows_before_anchor_gate": rows_before_anchor_gate,
        "num_kept_rows_before_anchor_gate": kept_rows_before_anchor_gate,
        "anchor_keep": anchor_keep,
        "anchor_filter_reason": anchor_filter_reason,
        "query_seconds": payload.get("query_seconds"),
        "rows_per_second": payload.get("rows_per_second"),
        "estimated_step_reuse_speedup": payload.get("estimated_step_reuse_speedup"),
        "restored_anchor_state_max_abs": payload.get("restored_anchor_state_max_abs"),
        "restored_nominal_local_state_max_abs": payload.get("restored_nominal_local_state_max_abs"),
        "repeat_local_state_max_abs": (payload.get("repeat_check") or {}).get("local_state_max_abs"),
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
        "rows": rows,
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "anchor_id",
        "row_idx",
        "obj_x",
        "obj_y",
        "obj_z",
        "speed",
        "qpos",
        "qvel",
        "anchor_step",
        "response_steps",
        "horizontal_disp",
        "vertical_disp",
        "anchor_keep",
        "anchor_filter_reason",
        "restored_nominal_local_state_max_abs",
        "repeat_local_state_max_abs",
        "direction",
        "epsilon",
        "direction_vec",
        "local_response",
        "final_response",
        "local_vel_response",
        "final_vel_response",
        "local_state_response",
        "final_state_response",
        "local_response_norm",
        "final_response_norm",
        "local_vel_response_norm",
        "final_vel_response_norm",
        "local_state_response_norm",
        "final_state_response_norm",
        "local_qvel_plus",
        "local_qvel_minus",
        "final_qvel_plus",
        "final_qvel_minus",
        "keep",
        "filter_reason",
        "contact_mean_delta",
        "contact_max_delta",
        "plus_contact_mean",
        "minus_contact_mean",
        "plus_contact_max",
        "minus_contact_max",
        "plus_contact_trace",
        "minus_contact_trace",
        "nominal_anchor_pos",
        "nominal_pre_pos",
        "nominal_post_pos",
        "nominal_initial_pos",
        "nominal_final_pos",
        "nominal_anchor_qvel",
        "nominal_pre_qvel",
        "nominal_post_qvel",
        "nominal_initial_qvel",
        "nominal_final_qvel",
        "nominal_pre_disp",
        "nominal_post_disp",
        "nominal_total_disp",
        "nominal_pre_qvel_delta",
        "nominal_post_qvel_delta",
        "nominal_total_qvel_delta",
        "nominal_contact_mean",
        "nominal_contact_max",
        "nominal_contact_min",
        "nominal_contact_trace",
        "max_total_contact_count_plus",
        "max_total_contact_count_minus",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key in (
                "qpos",
                "qvel",
                "direction_vec",
                "local_response",
                "final_response",
                "local_vel_response",
                "final_vel_response",
                "local_state_response",
                "final_state_response",
                "local_qvel_plus",
                "local_qvel_minus",
                "final_qvel_plus",
                "final_qvel_minus",
                "plus_contact_trace",
                "minus_contact_trace",
                "nominal_anchor_pos",
                "nominal_pre_pos",
                "nominal_post_pos",
                "nominal_initial_pos",
                "nominal_final_pos",
                "nominal_anchor_qvel",
                "nominal_pre_qvel",
                "nominal_post_qvel",
                "nominal_initial_qvel",
                "nominal_final_qvel",
                "nominal_pre_disp",
                "nominal_post_disp",
                "nominal_total_disp",
                "nominal_pre_qvel_delta",
                "nominal_post_qvel_delta",
                "nominal_total_qvel_delta",
                "nominal_contact_trace",
            ):
                flat[key] = json.dumps(flat[key])
            writer.writerow(flat)


def _write_anchor_manifest(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "anchor_id",
        "status",
        "anchor_keep",
        "anchor_filter_reason",
        "obj_y",
        "speed",
        "anchor_step",
        "horizontal_disp",
        "vertical_disp",
        "num_rows",
        "num_rows_before_anchor_gate",
        "num_kept_rows_before_anchor_gate",
        "query_seconds",
        "rows_per_second",
        "estimated_step_reuse_speedup",
        "restored_anchor_state_max_abs",
        "restored_nominal_local_state_max_abs",
        "repeat_local_state_max_abs",
        "json",
        "csv",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-env", default="genesis")
    parser.add_argument("--sampler", choices=("slow", "restore"), default="slow")
    parser.add_argument("--obj-x", type=float, default=0.306)
    parser.add_argument("--obj-y-values", default="0.072,0.076")
    parser.add_argument("--obj-z", type=float, default=0.120)
    parser.add_argument("--speeds", default="1.4,1.6")
    parser.add_argument("--qpos", default=DEFAULT_QPOS)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--push-steps", type=int, default=170)
    parser.add_argument("--response-steps", type=int, default=10)
    parser.add_argument("--contact-window", type=int, default=5)
    parser.add_argument("--max-contact-mean-delta", type=float, default=8.0)
    parser.add_argument("--max-contact-max-delta", type=float, default=12.0)
    parser.add_argument("--min-response-norm", type=float, default=1e-8)
    parser.add_argument("--max-response-norm", type=float, default=1e3)
    parser.add_argument("--drop-filtered", action="store_true")
    parser.add_argument("--collect-final", action="store_true")
    parser.add_argument("--no-requires-grad", action="store_true")
    parser.add_argument("--drop-bad-anchor", action="store_true")
    parser.add_argument("--min-kept-rows-per-anchor", type=int, default=0)
    parser.add_argument("--max-restore-local-state-diff", type=float, default=float("inf"))
    parser.add_argument("--max-repeat-local-state-diff", type=float, default=1e-8)
    parser.add_argument("--min-horizontal-disp", type=float, default=0.0)
    parser.add_argument("--max-abs-vertical-disp", type=float, default=float("inf"))
    parser.add_argument("--scales", default="1e-3")
    parser.add_argument("--num-random", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--omit-rows-json", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    obj_y_values = _parse_float_list(args.obj_y_values)
    speeds = _parse_float_list(args.speeds)

    jobs = []
    anchor_id = 0
    for obj_y in obj_y_values:
        for speed in speeds:
            anchor_id += 1
            jobs.append((anchor_id, obj_y, speed))

    anchor_records = []
    all_rows = []
    results = []
    if args.workers <= 1:
        for anchor_id, obj_y, speed in jobs:
            results.append(_run_anchor(args, anchor_id, obj_y, speed))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {
                executor.submit(_run_anchor, args, anchor_id, obj_y, speed): (anchor_id, obj_y, speed)
                for anchor_id, obj_y, speed in jobs
            }
            for future in concurrent.futures.as_completed(future_to_job):
                results.append(future.result())

    for record in sorted(results, key=lambda r: r["anchor_id"]):
        obj_y = record["obj_y"]
        speed = record["speed"]
        anchor_records.append({k: v for k, v in record.items() if k != "rows"})
        all_rows.extend(record["rows"])
        print(
            f"[multi-anchor:{record['anchor_id']:03d}] y={obj_y:.4f} speed={speed:.2f} "
            f"status={record['status']} keep={record.get('anchor_keep')} rows={len(record['rows'])} "
            f"h={record.get('horizontal_disp')} v={record.get('vertical_disp')} "
            f"q_s={record.get('query_seconds')} speedup={record.get('estimated_step_reuse_speedup')}",
            flush=True,
        )

    payload = {
        "description": "Aggregated A5 multi-anchor finite-scale FD response dataset",
        "obj_y_values": obj_y_values,
        "speeds": speeds,
        "obj_x": args.obj_x,
        "obj_z": args.obj_z,
        "qpos": args.qpos,
        "sampler": args.sampler,
        "collect_final": args.collect_final,
        "requires_grad": not args.no_requires_grad,
        "drop_bad_anchor": args.drop_bad_anchor,
        "min_kept_rows_per_anchor": args.min_kept_rows_per_anchor,
        "max_restore_local_state_diff": args.max_restore_local_state_diff,
        "max_repeat_local_state_diff": args.max_repeat_local_state_diff,
        "min_horizontal_disp": args.min_horizontal_disp,
        "max_abs_vertical_disp": args.max_abs_vertical_disp,
        "response_steps": args.response_steps,
        "contact_window": args.contact_window,
        "max_contact_mean_delta": args.max_contact_mean_delta,
        "max_contact_max_delta": args.max_contact_max_delta,
        "drop_filtered": args.drop_filtered,
        "scales": args.scales,
        "num_random": args.num_random,
        "num_anchors": len(anchor_records),
        "num_kept_anchors": sum(
            1
            for record in anchor_records
            if record.get("status") == "ok" and record.get("anchor_keep", True)
        ),
        "num_rows": len(all_rows),
        "num_kept_rows": sum(1 for row in all_rows if row["keep"]),
        "workers": args.workers,
        "anchors": anchor_records,
        "rows_omitted_from_json": args.omit_rows_json,
    }
    if not args.omit_rows_json:
        payload["rows"] = all_rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(args.csv, all_rows)
    if args.manifest_csv is not None:
        _write_anchor_manifest(args.manifest_csv, anchor_records)
    print(f"[multi-anchor] wrote {args.out} and {args.csv}")


if __name__ == "__main__":
    raise SystemExit(main())
