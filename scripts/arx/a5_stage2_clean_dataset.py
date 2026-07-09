"""Stage-2 clean restored-anchor data collection for A5.

This driver wraps ``a5_multi_anchor_fd_dataset.py --sampler restore`` with the
quality gates that should be on by default for larger runs. It shards by speed,
writes per-shard CSV/manifest files, and then merges them into a single dataset
CSV plus an anchor manifest.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MULTI_ANCHOR_SCRIPT = REPO_ROOT / "scripts" / "arx" / "a5_multi_anchor_fd_dataset.py"
DEFAULT_OUT_DIR = Path("analysis/2026-07-09_arx_pusher/stage2_clean")
DEFAULT_TAG = "a5_stage2_clean"
DEFAULT_OBJ_Y_VALUES = "0.068,0.070,0.072,0.074,0.076,0.078"
DEFAULT_SPEEDS = "1.25,1.35,1.45,1.55,1.65,1.75"
DEFAULT_QPOS = "0.0,1.4,-0.4,0.5,0.0,0.0"


def _parse_float_list(text):
    return [float(x) for x in text.split(",") if x.strip()]


def _fmt_values(values):
    return ",".join(f"{value:.6g}" for value in values)


def _tail(text, n=10):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _stage2_paths(args, shard_idx, speed):
    shard_name = f"{args.tag}_shard{shard_idx:03d}_speed{speed:.2f}"
    shard_dir = args.out_dir / "shards" / shard_name
    return {
        "shard_name": shard_name,
        "shard_dir": shard_dir,
        "json": shard_dir / f"{shard_name}.json",
        "csv": shard_dir / f"{shard_name}.csv",
        "manifest": shard_dir / f"{shard_name}_manifest.csv",
        "run_dir": shard_dir / "anchor_runs",
    }


def _global_anchor_id(shard_idx, source_anchor_id, anchors_per_shard):
    return (shard_idx - 1) * anchors_per_shard + int(source_anchor_id)


def _stage2_fieldnames(base_fieldnames):
    extras = ["shard_idx", "shard_speed", "source_anchor_id"]
    return extras + [name for name in base_fieldnames if name not in extras]


def _rewrite_stage2_row(row, record, anchors_per_shard):
    row = dict(row)
    source_anchor_id = row.get("anchor_id")
    row["shard_idx"] = record["shard_idx"]
    row["shard_speed"] = record["speed"]
    row["source_anchor_id"] = source_anchor_id
    if source_anchor_id:
        row["anchor_id"] = _global_anchor_id(
            record["shard_idx"], source_anchor_id, anchors_per_shard
        )
    return row


def _merge_stage2_csvs(records, out, anchors_per_shard, path_key):
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", newline="") as f_out:
        writer = None
        for record in records:
            if record["status"] != "ok":
                continue
            path = Path(record[path_key])
            if not path.exists():
                continue
            with path.open(newline="") as f_in:
                reader = csv.DictReader(f_in)
                if writer is None:
                    fieldnames = _stage2_fieldnames(reader.fieldnames)
                    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                    writer.writeheader()
                for row in reader:
                    row = _rewrite_stage2_row(row, record, anchors_per_shard)
                    writer.writerow(row)
                    total += 1
    return total


def _existing_shard_record(args, shard_idx, speed):
    paths = _stage2_paths(args, shard_idx, speed)
    json_path = paths["json"]
    csv_path = paths["csv"]
    manifest_path = paths["manifest"]
    status = "ok" if json_path.exists() and csv_path.exists() and manifest_path.exists() else "missing"
    record = {
        "shard_idx": shard_idx,
        "speed": speed,
        "status": status,
        "json": str(json_path),
        "csv": str(csv_path),
        "manifest_csv": str(manifest_path),
    }
    if status == "ok":
        payload = json.loads(json_path.read_text())
        record.update(
            {
                "num_anchors": payload["num_anchors"],
                "num_kept_anchors": payload["num_kept_anchors"],
                "num_rows": payload["num_rows"],
                "num_kept_rows": payload["num_kept_rows"],
            }
        )
    return record


def _run_shard(args, shard_idx, speed, obj_y_values):
    paths = _stage2_paths(args, shard_idx, speed)
    shard_dir = paths["shard_dir"]
    json_path = paths["json"]
    csv_path = paths["csv"]
    manifest_path = paths["manifest"]
    run_dir = paths["run_dir"]
    cmd = [
        sys.executable,
        str(MULTI_ANCHOR_SCRIPT),
        "--conda-env",
        args.conda_env,
        "--sampler",
        "restore",
        "--obj-x",
        str(args.obj_x),
        "--obj-y-values",
        _fmt_values(obj_y_values),
        "--obj-z",
        str(args.obj_z),
        "--speeds",
        f"{speed:.6g}",
        "--qpos",
        args.qpos,
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
        "--min-kept-rows-per-anchor",
        str(args.min_kept_rows_per_anchor),
        "--scales",
        args.scales,
        "--num-random",
        str(args.num_random),
        "--seed",
        str(args.seed + shard_idx * 1000),
        "--timeout",
        str(args.anchor_timeout),
        "--workers",
        str(args.workers),
        "--max-restore-local-state-diff",
        str(args.max_restore_local_state_diff),
        "--max-repeat-local-state-diff",
        str(args.max_repeat_local_state_diff),
        "--min-horizontal-disp",
        str(args.min_horizontal_disp),
        "--max-abs-vertical-disp",
        str(args.max_abs_vertical_disp),
        "--out",
        str(json_path),
        "--csv",
        str(csv_path),
        "--manifest-csv",
        str(manifest_path),
        "--run-dir",
        str(run_dir),
        "--drop-filtered",
        "--drop-bad-anchor",
        "--omit-rows-json",
    ]
    if args.no_requires_grad:
        cmd.append("--no-requires-grad")
    if args.collect_final:
        cmd.append("--collect-final")

    if args.dry_run:
        return {
            "shard_idx": shard_idx,
            "speed": speed,
            "status": "dry_run",
            "cmd": cmd,
            "json": str(json_path),
            "csv": str(csv_path),
            "manifest_csv": str(manifest_path),
        }

    shard_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.shard_timeout)
    elapsed = time.perf_counter() - start
    status = "ok" if proc.returncode == 0 and json_path.exists() and csv_path.exists() else f"error:{proc.returncode}"
    record = {
        "shard_idx": shard_idx,
        "speed": speed,
        "status": status,
        "elapsed_seconds": elapsed,
        "json": str(json_path),
        "csv": str(csv_path),
        "manifest_csv": str(manifest_path),
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }
    if status == "ok":
        payload = json.loads(json_path.read_text())
        record.update(
            {
                "num_anchors": payload["num_anchors"],
                "num_kept_anchors": payload["num_kept_anchors"],
                "num_rows": payload["num_rows"],
                "num_kept_rows": payload["num_kept_rows"],
            }
        )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-env", default="genesis")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--obj-x", type=float, default=0.306)
    parser.add_argument("--obj-y-values", default=DEFAULT_OBJ_Y_VALUES)
    parser.add_argument("--obj-z", type=float, default=0.120)
    parser.add_argument("--speeds", default=DEFAULT_SPEEDS)
    parser.add_argument("--qpos", default=DEFAULT_QPOS)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--push-steps", type=int, default=170)
    parser.add_argument("--response-steps", type=int, default=10)
    parser.add_argument("--contact-window", type=int, default=5)
    parser.add_argument("--max-contact-mean-delta", type=float, default=6.0)
    parser.add_argument("--max-contact-max-delta", type=float, default=10.0)
    parser.add_argument("--min-response-norm", type=float, default=1e-8)
    parser.add_argument("--max-response-norm", type=float, default=5.0)
    parser.add_argument("--min-kept-rows-per-anchor", type=int, default=12)
    parser.add_argument("--max-restore-local-state-diff", type=float, default=0.05)
    parser.add_argument("--max-repeat-local-state-diff", type=float, default=1e-8)
    parser.add_argument("--min-horizontal-disp", type=float, default=0.008)
    parser.add_argument("--max-abs-vertical-disp", type=float, default=0.0002)
    parser.add_argument("--scales", default="1e-3")
    parser.add_argument("--num-random", type=int, default=24)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--anchor-timeout", type=int, default=300)
    parser.add_argument("--shard-timeout", type=int, default=1800)
    parser.add_argument("--no-requires-grad", action="store_true")
    parser.add_argument("--collect-final", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Reuse existing shard outputs and only rebuild the final CSV/manifest/JSON.",
    )
    args = parser.parse_args()

    obj_y_values = _parse_float_list(args.obj_y_values)
    speeds = _parse_float_list(args.speeds)
    if args.max_shards > 0:
        speeds = speeds[: args.max_shards]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_records = []
    for shard_idx, speed in enumerate(speeds, start=1):
        if args.merge_only:
            record = _existing_shard_record(args, shard_idx, speed)
        else:
            record = _run_shard(args, shard_idx, speed, obj_y_values)
        shard_records.append(record)
        print(
            f"[stage2:{shard_idx:03d}] speed={speed:.3f} status={record['status']} "
            f"anchors={record.get('num_kept_anchors')}/{record.get('num_anchors')} "
            f"rows={record.get('num_rows')} elapsed={record.get('elapsed_seconds')}",
            flush=True,
        )

    final_csv = args.out_dir / f"{args.tag}.csv"
    final_manifest = args.out_dir / f"{args.tag}_manifest.csv"
    final_json = args.out_dir / f"{args.tag}.json"
    anchors_per_shard = len(obj_y_values)
    total_rows = (
        0
        if args.dry_run
        else _merge_stage2_csvs(shard_records, final_csv, anchors_per_shard, "csv")
    )
    total_manifest_rows = (
        0
        if args.dry_run
        else _merge_stage2_csvs(
            shard_records, final_manifest, anchors_per_shard, "manifest_csv"
        )
    )
    payload = {
        "description": "A5 Stage-2 clean restored-anchor dataset",
        "tag": args.tag,
        "obj_y_values": obj_y_values,
        "speeds": speeds,
        "anchors_per_shard": anchors_per_shard,
        "anchor_id_policy": (
            "anchor_id=(shard_idx-1)*anchors_per_shard+source_anchor_id; "
            "source_anchor_id preserves the per-shard anchor id."
        ),
        "num_candidate_anchors": len(obj_y_values) * len(speeds),
        "num_shards": len(shard_records),
        "num_ok_shards": sum(1 for r in shard_records if r["status"] == "ok"),
        "num_rows": total_rows,
        "num_manifest_rows": total_manifest_rows,
        "final_csv": str(final_csv),
        "final_manifest_csv": str(final_manifest),
        "quality_gates": {
            "max_response_norm": args.max_response_norm,
            "min_kept_rows_per_anchor": args.min_kept_rows_per_anchor,
            "max_restore_local_state_diff": args.max_restore_local_state_diff,
            "max_repeat_local_state_diff": args.max_repeat_local_state_diff,
            "min_horizontal_disp": args.min_horizontal_disp,
            "max_abs_vertical_disp": args.max_abs_vertical_disp,
            "max_contact_mean_delta": args.max_contact_mean_delta,
            "max_contact_max_delta": args.max_contact_max_delta,
        },
        "collection": {
            "scales": args.scales,
            "num_random": args.num_random,
            "response_steps": args.response_steps,
            "workers": args.workers,
            "requires_grad": not args.no_requires_grad,
            "collect_final": args.collect_final,
        },
        "shards": shard_records,
    }
    final_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[stage2] wrote {final_json}")
    if not args.dry_run:
        print(f"[stage2] wrote {final_csv} and {final_manifest}")


if __name__ == "__main__":
    raise SystemExit(main())
