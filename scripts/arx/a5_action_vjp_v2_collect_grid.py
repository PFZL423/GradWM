"""Coordinate batched long-lived workers for A5 action-side VJP v2."""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = REPO_ROOT / "scripts" / "arx" / "a5_action_vjp_v2_collect_worker.py"
DEFAULT_OUT_DIR = Path("analysis/2026-07-09_arx_pusher/action_vjp_v2_dataset")
DEFAULT_QPOS = [0.0, 1.4, -0.4, 0.5, 0.0, 0.0]


def _float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def _int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def _tail(value, lines=12):
    output = [line for line in value.splitlines() if line.strip()]
    return "\n".join(output[-lines:])


def _uniform_hash(seed, *parts):
    text = ":".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / float(2**64)


def _split_name(seed, obj_y, speed, anchor_step, ood_y_min, val_frac, test_frac):
    if obj_y >= ood_y_min:
        return "test_ood"
    value = _uniform_hash(seed, f"{obj_y:.8g}", f"{speed:.8g}", anchor_step)
    if value < val_frac:
        return "val"
    if value < val_frac + test_frac:
        return "test_id"
    return "train"


def _build_jobs(args):
    jobs = []
    anchor_id = 0
    obj_x_values = _float_list(args.obj_x_values) if args.obj_x_values else [args.obj_x]
    for obj_x in obj_x_values:
        for obj_y in _float_list(args.obj_y_values):
            for speed in _float_list(args.speeds):
                for anchor_step in _int_list(args.anchor_steps):
                    anchor_id += 1
                    jobs.append(
                        {
                            "anchor_id": anchor_id,
                            "split": _split_name(
                                args.seed,
                                obj_y,
                                speed,
                                anchor_step,
                                args.ood_y_min,
                                args.val_frac,
                                args.test_frac,
                            ),
                            "obj_pos": [obj_x, obj_y, args.obj_z],
                            "qpos": DEFAULT_QPOS,
                            "qvel": [speed, 0.0, 0.0, 0.0, 0.0, 0.0],
                            "speed": speed,
                            "anchor_step": anchor_step,
                            "seed": args.seed + anchor_id,
                        }
                    )
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    return jobs


def _load_jobs(path):
    jobs = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            jobs.append(
                {
                    "anchor_id": int(row["anchor_id"]),
                    "split": row["split"],
                    "obj_pos": json.loads(row["obj_pos"]),
                    "qpos": json.loads(row["qpos"]),
                    "qvel": json.loads(row["qvel"]),
                    "speed": float(row["speed"]),
                    "anchor_step": int(row["anchor_step"]),
                    "seed": int(row["seed"]),
                }
            )
    return jobs


def _filter_jobs(jobs, anchors_path, field, values):
    accepted = set(values.split(","))
    with anchors_path.open(newline="") as f:
        keep_ids = {
            int(row["anchor_id"])
            for row in csv.DictReader(f)
            if row.get(field) in accepted
        }
    return [job for job in jobs if job["anchor_id"] in keep_ids]


def _limit_jobs_by_split(jobs, specification, seed):
    if not specification:
        return jobs
    limits = {}
    for item in specification.split(","):
        name, value = item.split("=", 1)
        limits[name] = int(value)
    grouped = {}
    for job in jobs:
        grouped.setdefault(job["split"], []).append(job)
    selected_ids = set()
    for split, split_jobs in grouped.items():
        split_jobs.sort(
            key=lambda job: _uniform_hash(seed, "filtered", split, job["anchor_id"])
        )
        limit = limits.get(split, len(split_jobs))
        selected_ids.update(job["anchor_id"] for job in split_jobs[:limit])
    return [job for job in jobs if job["anchor_id"] in selected_ids]


def _chunks(values, size):
    return [values[start:start + size] for start in range(0, len(values), size)]


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(source, output_rows):
    if not source.exists() or source.stat().st_size == 0:
        return
    with source.open(newline="") as f:
        output_rows.extend(csv.DictReader(f))


def _run_batch(args, batch_index, jobs, config):
    batch_dir = args.run_dir / f"batch_{batch_index:04d}"
    request_path = args.request_dir / f"batch_{batch_index:04d}.json"
    worker_json = batch_dir / "worker.json"
    rows_csv = batch_dir / "rows.csv"
    if args.resume and worker_json.exists() and rows_csv.exists():
        payload = json.loads(worker_json.read_text())
        return {
            "batch_index": batch_index,
            "status": "resumed",
            "num_jobs": payload.get("num_jobs", len(jobs)),
            "num_rows": payload.get("num_rows", 0),
            "elapsed_seconds": 0.0,
            "worker_json": str(worker_json),
            "rows_csv": str(rows_csv),
            "anchors_csv": str(batch_dir / "anchors.csv"),
            "stdout_tail": "",
            "stderr_tail": "",
        }

    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps({"config": config, "jobs": jobs}, indent=2) + "\n")
    batch_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "--no-capture-output",
        "python",
        str(WORKER_SCRIPT),
        "--request",
        str(request_path),
        "--out-dir",
        str(batch_dir),
    ]
    start = time.perf_counter()
    last = None
    for _ in range(args.retries + 1):
        try:
            env = None
            if args.gpu_ids:
                gpu_ids = [item for item in args.gpu_ids.split(",") if item.strip()]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[batch_index % len(gpu_ids)]
            last = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.batch_timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            last = exc
            continue
        if last.returncode == 0 and worker_json.exists():
            break
    elapsed = time.perf_counter() - start
    if not worker_json.exists():
        return {
            "batch_index": batch_index,
            "status": "timeout" if isinstance(last, subprocess.TimeoutExpired) else f"error:{getattr(last, 'returncode', None)}",
            "num_jobs": len(jobs),
            "num_rows": 0,
            "elapsed_seconds": elapsed,
            "worker_json": str(worker_json),
            "rows_csv": str(rows_csv),
            "anchors_csv": str(batch_dir / "anchors.csv"),
            "stdout_tail": _tail(getattr(last, "stdout", "") or ""),
            "stderr_tail": _tail(getattr(last, "stderr", "") or ""),
        }
    payload = json.loads(worker_json.read_text())
    return {
        "batch_index": batch_index,
        "status": "ok",
        "num_jobs": payload["num_jobs"],
        "num_rows": payload["num_rows"],
        "elapsed_seconds": elapsed,
        "worker_json": str(worker_json),
        "rows_csv": str(rows_csv),
        "anchors_csv": str(batch_dir / "anchors.csv"),
        "stdout_tail": _tail(getattr(last, "stdout", "") or ""),
        "stderr_tail": _tail(getattr(last, "stderr", "") or ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-env", default="genesis")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="a5_action_vjp_v2")
    parser.add_argument("--jobs-manifest", type=Path)
    parser.add_argument("--filter-anchors", type=Path)
    parser.add_argument("--filter-field", default="status")
    parser.add_argument("--filter-statuses", default="contact_candidate,ok")
    parser.add_argument("--max-filtered-per-split", default="")
    parser.add_argument("--obj-x", type=float, default=0.306)
    parser.add_argument("--obj-x-values", default="")
    parser.add_argument(
        "--obj-y-values",
        default="0.0675,0.0700,0.0725,0.0750,0.0775,0.0800,0.0825,0.0850,0.0875,0.0900,0.0925,0.0950,0.0975,0.1000,0.1025",
    )
    parser.add_argument("--obj-z", type=float, default=0.120)
    parser.add_argument("--speeds", default="0.8,1.0,1.2,1.4,1.6")
    parser.add_argument("--anchor-steps", default="24,32,40,48,56,64,72,80,88,96")
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--response-steps", type=int, default=1)
    parser.add_argument("--branch-mode", choices=("replay", "restore"), default="replay")
    parser.add_argument("--epsilons", default="0.003,0.01")
    parser.add_argument("--num-random", type=int, default=6)
    parser.add_argument("--max-contacts", type=int, default=16)
    parser.add_argument("--max-repeat-state-diff", type=float, default=1e-7)
    parser.add_argument("--require-arm-object-contact", action="store_true", default=True)
    parser.add_argument(
        "--allow-no-arm-object-contact",
        dest="require_arm_object_contact",
        action="store_false",
    )
    parser.add_argument("--min-arm-object-contact-events", type=int, default=1)
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--seed", type=int, default=5201)
    parser.add_argument("--ood-y-min", type=float, default=0.0975)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--batch-timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = _load_jobs(args.jobs_manifest) if args.jobs_manifest else _build_jobs(args)
    if args.filter_anchors:
        jobs = _filter_jobs(
            jobs, args.filter_anchors, args.filter_field, args.filter_statuses
        )
        jobs = _limit_jobs_by_split(jobs, args.max_filtered_per_split, args.seed)
    config = {
        "settle_steps": args.settle_steps,
        "response_steps": args.response_steps,
        "branch_mode": args.branch_mode,
        "epsilons": _float_list(args.epsilons),
        "num_random": args.num_random,
        "max_contacts": args.max_contacts,
        "max_repeat_state_diff": args.max_repeat_state_diff,
        "require_arm_object_contact": args.require_arm_object_contact,
        "min_arm_object_contact_events": args.min_arm_object_contact_events,
        "collect_probes": not args.scan_only,
    }
    args.run_dir = args.out_dir / "runs"
    args.request_dir = args.out_dir / "requests"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = args.out_dir / f"{args.tag}_frozen_split.csv"
    _write_csv(split_manifest, jobs)
    split_counts = {}
    for job in jobs:
        split_counts[job["split"]] = split_counts.get(job["split"], 0) + 1
    batches = _chunks(jobs, args.batch_size)
    print(
        f"[a5-vjp-v2-grid] jobs={len(jobs)} batches={len(batches)} split={split_counts} "
        f"scan_only={args.scan_only} eps={config['epsilons']} dirs={7 + args.num_random}",
        flush=True,
    )
    if args.dry_run:
        print(f"[a5-vjp-v2-grid] wrote frozen split {split_manifest}")
        return 0

    start = time.perf_counter()
    records = []
    if args.workers <= 1:
        for batch_index, batch_jobs in enumerate(batches):
            records.append(_run_batch(args, batch_index, batch_jobs, config))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_run_batch, args, batch_index, batch_jobs, config): batch_index
                for batch_index, batch_jobs in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(futures):
                records.append(future.result())
    records.sort(key=lambda record: record["batch_index"])
    for record in records:
        print(
            f"[a5-vjp-v2-grid:{record['batch_index']:04d}] status={record['status']} "
            f"jobs={record['num_jobs']} rows={record['num_rows']} "
            f"elapsed={record['elapsed_seconds']:.1f}s",
            flush=True,
        )

    aggregate_rows = []
    aggregate_anchors = []
    for record in records:
        if record["status"] not in ("ok", "resumed"):
            continue
        _append_csv(Path(record["rows_csv"]), aggregate_rows)
        _append_csv(Path(record["anchors_csv"]), aggregate_anchors)
    rows_path = args.out_dir / f"{args.tag}_rows.csv"
    anchors_path = args.out_dir / f"{args.tag}_anchors.csv"
    batches_path = args.out_dir / f"{args.tag}_batches.csv"
    _write_csv(rows_path, aggregate_rows)
    _write_csv(anchors_path, aggregate_anchors)
    _write_csv(batches_path, records)
    payload = {
        "description": "A5 action-side VJP v2 batched grid collection",
        "config": config,
        "grid": {
            "obj_x_values": (
                _float_list(args.obj_x_values) if args.obj_x_values else [args.obj_x]
            ),
            "obj_y_values": _float_list(args.obj_y_values),
            "speeds": _float_list(args.speeds),
            "anchor_steps": _int_list(args.anchor_steps),
        },
        "split_counts_frozen": split_counts,
        "selection": {
            "jobs_manifest": None if args.jobs_manifest is None else str(args.jobs_manifest),
            "filter_anchors": None if args.filter_anchors is None else str(args.filter_anchors),
            "filter_field": args.filter_field,
            "filter_values": args.filter_statuses,
            "max_filtered_per_split": args.max_filtered_per_split,
        },
        "num_jobs": len(jobs),
        "num_batches": len(batches),
        "num_successful_jobs": len(aggregate_anchors),
        "num_rows": len(aggregate_rows),
        "elapsed_seconds": time.perf_counter() - start,
        "frozen_split_csv": str(split_manifest),
        "rows_csv": str(rows_path),
        "anchors_csv": str(anchors_path),
        "batches_csv": str(batches_path),
        "batches": records,
    }
    json_path = args.out_dir / f"{args.tag}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[a5-vjp-v2-grid] complete jobs={len(aggregate_anchors)}/{len(jobs)} "
        f"rows={len(aggregate_rows)} elapsed={payload['elapsed_seconds']:.1f}s"
    )
    print(f"[a5-vjp-v2-grid] wrote {json_path}")
    print(f"[a5-vjp-v2-grid] wrote {rows_path}")


if __name__ == "__main__":
    raise SystemExit(main())
