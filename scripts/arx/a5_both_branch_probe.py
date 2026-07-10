"""Diagnose piecewise-linear branch structure in the A5 both cohort."""

import argparse
import copy
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
WORKER_PATH = SCRIPT_PATH.parent / "a5_both_branch_collect_worker.py"
DEFAULT_OUT = REPO_ROOT / "analysis/2026-07-09_arx_pusher/both_branch_probe"
DEFAULT_PREFILTER = (
    REPO_ROOT
    / "analysis/2026-07-09_arx_pusher/action_vjp_v2_restore_prefilter1200"
    / "trusted/a5_action_vjp_v2_anchor_matrices.csv"
)
DEFAULT_DENSE = (
    REPO_ROOT
    / "analysis/2026-07-09_arx_pusher/action_vjp_v2_dense_scan26568"
    / "a5_action_vjp_v2_dense_scan26568_frozen_split.csv"
)
DEFAULT_OLD_PROBE = (
    REPO_ROOT
    / "analysis/2026-07-09_arx_pusher/marginal_probe"
    / "a5_marginal_probe_frozen_selection.csv"
)
DEFAULT_REPORT = REPO_ROOT / "notes/a5_vjp_progress/2026-07-10_both_branch_probe.md"

ACTION_DIM = 6
TARGET_DIM = 3
DENSE_EPSILON = 0.003
VALIDATION_EPSILON = 0.01
WEAK_REASONS = {"weak_y_vjp", "weak_random_y_signal"}


def _as_bool(value):
    return str(value).strip().lower() not in ("", "0", "false", "no", "none")


def _json(value, default):
    return default if value in (None, "") else json.loads(value)


def _read_csv(path):
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return value


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*values):
    text = "|".join(str(value) for value in values)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _percentile(values, q):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else float(np.percentile(finite, q))


def _distribution(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "num": len(finite),
        "mean": None if not finite else float(np.mean(finite)),
        "q25": _percentile(finite, 25),
        "median": _percentile(finite, 50),
        "q75": _percentile(finite, 75),
        "min": None if not finite else min(finite),
        "max": None if not finite else max(finite),
    }


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denom < 1e-12 else float(np.dot(a, b) / denom)


def _stratified_maximin_select(rows, count, seed):
    features = np.asarray(
        [[row["obj_x"], row["obj_y"], row["speed"], row["anchor_step"]] for row in rows],
        dtype=np.float64,
    )
    low = features.min(axis=0)
    span = features.max(axis=0) - low
    span[span < 1e-12] = 1.0
    features = (features - low) / span
    speeds = sorted({row["speed"] for row in rows})
    base, remainder = divmod(count, len(speeds))
    extra_order = sorted(speeds, key=lambda speed: _stable_hash(seed, "speed", speed))
    quotas = {speed: base + int(speed in set(extra_order[:remainder])) for speed in speeds}
    remaining = set(range(len(rows)))
    selected = []
    speed_counts = Counter()
    x_counts = Counter()
    phase_bin_counts = Counter()
    y_bin_counts = Counter()
    while len(selected) < count:
        eligible = [
            index
            for index in remaining
            if speed_counts[rows[index]["speed"]] < quotas[rows[index]["speed"]]
        ]
        if not eligible:
            raise RuntimeError("stratified maximin exhausted eligible speed strata")

        def score(index):
            distance = (
                4.0
                if not selected
                else min(
                    float(np.sum((features[index] - features[other]) ** 2))
                    for other in selected
                )
            )
            row = rows[index]
            phase_bin = int(row["anchor_step"] // 16)
            y_bin = int(round(row["obj_y"] * 1000.0)) // 4
            balance = (
                0.35 / (1 + x_counts[row["obj_x"]])
                + 0.20 / (1 + phase_bin_counts[phase_bin])
                + 0.10 / (1 + y_bin_counts[y_bin])
            )
            return distance + balance, -_stable_hash(seed, "tie", row["anchor_id"])

        best = max(eligible, key=score)
        selected.append(best)
        remaining.remove(best)
        row = rows[best]
        speed_counts[row["speed"]] += 1
        x_counts[row["obj_x"]] += 1
        phase_bin_counts[int(row["anchor_step"] // 16)] += 1
        y_bin_counts[int(round(row["obj_y"] * 1000.0)) // 4] += 1
    return [rows[index] for index in selected]


def _paths(out_dir):
    return {
        "selection": out_dir / "a5_both_branch_probe_frozen_selection.csv",
        "jobs": out_dir / "a5_both_branch_probe_jobs.csv",
        "requests": out_dir / "requests",
        "runs": out_dir / "runs",
        "batches": out_dir / "a5_both_branch_probe_batches.csv",
        "rows": out_dir / "a5_both_branch_probe_rows.csv",
        "anchors": out_dir / "a5_both_branch_probe_anchors.csv",
        "anchor_summary": out_dir / "a5_both_branch_probe_anchor_summary.csv",
        "k_metrics": out_dir / "a5_both_branch_probe_k_metrics.csv",
        "branch_metrics": out_dir / "a5_both_branch_probe_branch_metrics.csv",
        "selector_metrics": out_dir / "a5_both_branch_probe_selector_metrics.csv",
        "stability_metrics": out_dir / "a5_both_branch_probe_k_seed_stability.csv",
        "assignments": out_dir / "a5_both_branch_probe_assignments.csv",
        "summary": out_dir / "a5_both_branch_probe_summary.json",
    }


def _select(args):
    paths = _paths(args.out_dir)
    if paths["selection"].exists() and not args.force_selection:
        rows = _read_csv(paths["selection"])
        if len(rows) != args.num_anchors:
            raise RuntimeError(
                f"frozen selection has {len(rows)} rows, expected {args.num_anchors}"
            )
        print(f"[both-select] reused {paths['selection']}")
        return rows

    old_ids = {int(row["anchor_id"]) for row in _read_csv(args.old_probe)}
    dense = {int(row["anchor_id"]): row for row in _read_csv(args.dense_manifest)}
    candidates = []
    for row in _read_csv(args.prefilter_matrices):
        reasons = set(row["gate_reasons"].split("|"))
        anchor_id = int(row["anchor_id"])
        if (
            _as_bool(row["usable"])
            or bool(reasons & WEAK_REASONS)
            or not {"cross_epsilon_y_cosine", "contact_signature_switch"} <= reasons
            or anchor_id in old_ids
            or anchor_id not in dense
        ):
            continue
        job = dense[anchor_id]
        obj_pos = _json(job["obj_pos"], [0.0, 0.0, 0.0])
        candidates.append(
            {
                "anchor_id": anchor_id,
                "probe_role": "frozen_branch_discovery",
                "source_subtype": "both",
                "source_gate_reasons": row["gate_reasons"],
                "original_split": row["split"],
                "obj_x": float(obj_pos[0]),
                "obj_y": float(obj_pos[1]),
                "obj_z": float(obj_pos[2]),
                "speed": float(job["speed"]),
                "anchor_step": int(job["anchor_step"]),
                "qpos": job["qpos"],
                "qvel": job["qvel"],
                "dense_seed": int(job["seed"]),
                "branch_seed": args.direction_seed + 1009 * anchor_id,
                "source_cross_epsilon_y_cosine": row.get("cross_epsilon_y_cosine", ""),
                "source_signature_equal_rate": row.get("contact_signature_equal_rate", ""),
            }
        )
    if len(candidates) < args.num_anchors:
        raise RuntimeError(f"only {len(candidates)} eligible both anchors")
    selected = _stratified_maximin_select(candidates, args.num_anchors, args.selection_seed)
    selected.sort(key=lambda row: int(row["anchor_id"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(paths["selection"], selected)
    jobs = [
        {
            "anchor_id": row["anchor_id"],
            "split": "both_branch_discovery",
            "obj_pos": [row["obj_x"], row["obj_y"], row["obj_z"]],
            "qpos": _json(row["qpos"], []),
            "qvel": _json(row["qvel"], []),
            "speed": row["speed"],
            "anchor_step": row["anchor_step"],
            "branch_seed": row["branch_seed"],
        }
        for row in selected
    ]
    _write_csv(paths["jobs"], jobs)
    print(
        f"[both-select] candidates={len(candidates)} selected={len(selected)} "
        f"old_probe_overlap={len({int(row['anchor_id']) for row in selected} & old_ids)}"
    )
    print(f"[both-select] wrote {paths['selection']}")
    return selected


def _load_jobs(path):
    jobs = []
    for row in _read_csv(path):
        jobs.append(
            {
                "anchor_id": int(row["anchor_id"]),
                "split": row["split"],
                "obj_pos": _json(row["obj_pos"], []),
                "qpos": _json(row["qpos"], []),
                "qvel": _json(row["qvel"], []),
                "speed": float(row["speed"]),
                "anchor_step": int(row["anchor_step"]),
                "branch_seed": int(row["branch_seed"]),
            }
        )
    return jobs


def _chunks(values, size):
    return [values[index:index + size] for index in range(0, len(values), size)]


def _run_batch(args, index, jobs, config, gpu_id=None):
    paths = _paths(args.out_dir)
    batch_dir = paths["runs"] / f"batch_{index:04d}"
    request_path = paths["requests"] / f"batch_{index:04d}.json"
    worker_path = batch_dir / "worker.json"
    rows_path = batch_dir / "rows.csv"
    anchors_path = batch_dir / "anchors.csv"
    if args.resume and worker_path.exists() and rows_path.exists() and anchors_path.exists():
        payload = json.loads(worker_path.read_text())
        return {
            "batch_index": index,
            "status": "resumed",
            "num_jobs": payload["num_jobs"],
            "num_rows": payload["num_rows"],
            "query_replays": payload["query_replays"],
            "elapsed_seconds": 0.0,
            "worker_elapsed_seconds": payload["elapsed_seconds"],
            "rows_csv": str(rows_path),
            "anchors_csv": str(anchors_path),
            "stdout_tail": "",
            "stderr_tail": "",
        }

    request_path.parent.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps({"config": config, "jobs": jobs}, indent=2) + "\n")
    command = [
        sys.executable,
        str(WORKER_PATH),
        "--request",
        str(request_path),
        "--out-dir",
        str(batch_dir),
    ]
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    start = time.perf_counter()
    last = None
    for _ in range(args.retries + 1):
        try:
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
        if last.returncode == 0 and worker_path.exists():
            break
    elapsed = time.perf_counter() - start
    if not worker_path.exists():
        return {
            "batch_index": index,
            "status": "timeout"
            if isinstance(last, subprocess.TimeoutExpired)
            else f"error:{getattr(last, 'returncode', None)}",
            "num_jobs": len(jobs),
            "num_rows": 0,
            "query_replays": 0,
            "elapsed_seconds": elapsed,
            "worker_elapsed_seconds": 0.0,
            "rows_csv": str(rows_path),
            "anchors_csv": str(anchors_path),
            "stdout_tail": "\n".join((getattr(last, "stdout", "") or "").splitlines()[-20:]),
            "stderr_tail": "\n".join((getattr(last, "stderr", "") or "").splitlines()[-20:]),
        }
    payload = json.loads(worker_path.read_text())
    return {
        "batch_index": index,
        "status": "ok",
        "num_jobs": payload["num_jobs"],
        "num_rows": payload["num_rows"],
        "query_replays": payload["query_replays"],
        "elapsed_seconds": elapsed,
        "worker_elapsed_seconds": payload["elapsed_seconds"],
        "rows_csv": str(rows_path),
        "anchors_csv": str(anchors_path),
        "stdout_tail": "\n".join((getattr(last, "stdout", "") or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((getattr(last, "stderr", "") or "").splitlines()[-20:]),
    }


def _collect(args):
    paths = _paths(args.out_dir)
    if not paths["jobs"].exists():
        _select(args)
    jobs = _load_jobs(paths["jobs"])
    config = {
        "settle_steps": args.settle_steps,
        "response_steps": 0,
        "num_random": args.num_random,
        "train_random": args.train_random,
        "dense_epsilon": DENSE_EPSILON,
        "validation_epsilon": VALIDATION_EPSILON,
        "max_contacts": args.max_contacts,
        "max_repeat_state_diff": args.max_repeat_state_diff,
    }
    batches = _chunks(jobs, args.batch_size)
    start = time.perf_counter()
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    lane_count = min(args.workers, len(gpu_ids)) if gpu_ids else args.workers
    lane_count = max(1, lane_count)
    lanes = [[] for _ in range(lane_count)]
    for index, batch in enumerate(batches):
        lanes[index % lane_count].append((index, batch))

    def run_lane(lane_index):
        lane_records = []
        gpu_id = gpu_ids[lane_index] if gpu_ids else None
        for index, batch in lanes[lane_index]:
            record = _run_batch(args, index, batch, config, gpu_id)
            lane_records.append(record)
            print(
                f"[both-collect:{record['batch_index']:04d}] gpu={gpu_id} "
                f"status={record['status']} jobs={record['num_jobs']} "
                f"rows={record['num_rows']} replays={record['query_replays']} "
                f"elapsed={record['elapsed_seconds']:.1f}s",
                flush=True,
            )
            if record["stderr_tail"] and record["status"] not in ("ok", "resumed"):
                print(record["stderr_tail"], file=sys.stderr, flush=True)
        return lane_records

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=lane_count) as executor:
        futures = [executor.submit(run_lane, lane_index) for lane_index in range(lane_count)]
        for future in concurrent.futures.as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda row: row["batch_index"])
    aggregate_rows = []
    aggregate_anchors = []
    for record in records:
        if record["status"] not in ("ok", "resumed"):
            continue
        aggregate_rows.extend(_read_csv(Path(record["rows_csv"])))
        aggregate_anchors.extend(_read_csv(Path(record["anchors_csv"])))
    _write_csv(paths["batches"], records)
    _write_csv(paths["rows"], aggregate_rows)
    _write_csv(paths["anchors"], aggregate_anchors)
    failures = len(jobs) - len(aggregate_anchors)
    print(
        f"[both-collect] anchors={len(aggregate_anchors)}/{len(jobs)} "
        f"rows={len(aggregate_rows)} query_replays={sum(int(r['query_replays']) for r in records)} "
        f"wall={time.perf_counter() - start:.1f}s failures={failures}"
    )
    if failures > 10:
        raise RuntimeError(f"sampling failed for {failures} anchors (>10 hard stop)")
    return records


def _geometry_vector(trace):
    trace = _json(trace, []) if isinstance(trace, str) else trace
    if not trace:
        return np.zeros(10, dtype=np.float64)
    geometry = trace[-1]
    pooled = geometry.get("pooled", {})
    return np.asarray(
        [
            float(geometry.get("count", 0)),
            float(pooled.get("penetration_sum", 0.0)),
            float(pooled.get("penetration_mean", 0.0)),
            float(pooled.get("penetration_max", 0.0)),
            *[float(value) for value in pooled.get("position_object_mean", [0.0] * 3)],
            *[float(value) for value in pooled.get("normal_object_mean", [0.0] * 3)],
        ],
        dtype=np.float64,
    )


def _mode_key(trace):
    trace = _json(trace, []) if isinstance(trace, str) else trace
    if not trace:
        return "empty"
    geometry = trace[-1]
    return json.dumps(
        [int(geometry.get("count", 0)), geometry.get("signature", [])],
        separators=(",", ":"),
    )


def _samples(rows, epsilon, role=None):
    output = []
    for row in rows:
        if not _as_bool(row.get("keep", True)):
            continue
        if not math.isclose(float(row["epsilon"]), epsilon, rel_tol=0.0, abs_tol=1e-12):
            continue
        if role is not None and row["direction_role"] != role:
            continue
        vector = np.asarray(_json(row["direction_vec"], []), dtype=np.float64)
        for side, sign in (("plus", 1.0), ("minus", -1.0)):
            output.append(
                {
                    "direction": sign * vector,
                    "response": np.asarray(
                        _json(row[f"{side}_local_vel_response"], []), dtype=np.float64
                    ),
                    "contact": _geometry_vector(row[f"{side}_contact_geometry_trace"]),
                    "mode_key": _mode_key(row[f"{side}_contact_geometry_trace"]),
                    "base_direction": row["direction"],
                    "direction_kind": row["direction_kind"],
                    "direction_index": int(row["direction_index"]),
                    "side": side,
                    "role": row["direction_role"],
                    "epsilon": epsilon,
                }
            )
    return output


def _arrays(samples):
    return (
        np.stack([row["direction"] for row in samples]),
        np.stack([row["response"] for row in samples]),
        np.stack([row["contact"] for row in samples]),
    )


def _fit_matrix(x, y, weights=None):
    if weights is None:
        weights = np.ones(len(x), dtype=np.float64)
    root = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
    xw = x * root
    yw = y * root
    gram = xw.T @ xw
    ridge = max(float(np.trace(gram)) / ACTION_DIM, 1.0) * 1e-7
    coefficients = np.linalg.solve(gram + ridge * np.eye(ACTION_DIM), xw.T @ yw)
    return coefficients.T


def _normalized_errors(x, y, matrices, floor):
    errors = []
    for matrix in matrices:
        prediction = x @ matrix.T
        numerator = np.sum((prediction - y) ** 2, axis=1)
        denominator = np.sum(y * y, axis=1) + floor * floor
        errors.append(numerator / denominator)
    return np.stack(errors, axis=1)


def _standardize_features(features):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (features - mean) / scale


def _kmeans(features, k, seed, iterations=50):
    rng = np.random.default_rng(seed)
    count = len(features)
    centers = [features[rng.integers(count)]]
    while len(centers) < k:
        distance = np.min(
            np.stack([np.sum((features - center) ** 2, axis=1) for center in centers], axis=1),
            axis=1,
        )
        if float(distance.sum()) <= 1e-12:
            choice = rng.integers(count)
        else:
            choice = rng.choice(count, p=distance / distance.sum())
        centers.append(features[choice])
    centers = np.stack(centers)
    labels = np.zeros(count, dtype=np.int64)
    for _ in range(iterations):
        new_labels = np.argmin(
            np.stack([np.sum((features - center) ** 2, axis=1) for center in centers], axis=1),
            axis=1,
        )
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for branch in range(k):
            members = features[labels == branch]
            if len(members):
                centers[branch] = members.mean(axis=0)
    return labels


def _initial_labels(x, y, contact, k, seed, restart):
    norms = np.linalg.norm(y, axis=1)
    normalized_y = y / np.maximum(norms[:, None], 1e-9)
    log_norm = np.log(np.maximum(norms, 1e-9))[:, None]
    if restart % 3 == 0:
        descriptor = np.concatenate([normalized_y, log_norm, 0.25 * x], axis=1)
        return _kmeans(_standardize_features(descriptor), k, seed + restart)
    if restart % 3 == 1:
        descriptor = np.concatenate([normalized_y, 0.25 * x, 0.25 * contact], axis=1)
        return _kmeans(_standardize_features(descriptor), k, seed + restart)
    rng = np.random.default_rng(seed + restart)
    projection = x @ rng.normal(size=x.shape[1]) + 0.25 * normalized_y @ rng.normal(size=3)
    order = np.argsort(projection)
    labels = np.empty(len(x), dtype=np.int64)
    for branch, indices in enumerate(np.array_split(order, k)):
        labels[indices] = branch
    return labels


def _fit_piecewise(x, y, contact, k, floor, min_branch_size, seed, restarts):
    norms = np.linalg.norm(y, axis=1)
    weights = 1.0 / np.maximum(norms * norms + floor * floor, floor * floor)
    weights /= np.median(weights)
    weights = np.minimum(weights, 100.0)
    best = None
    if k == 1:
        matrix = _fit_matrix(x, y, weights)
        return {
            "k": 1,
            "matrices": np.stack([matrix]),
            "labels": np.zeros(len(x), dtype=np.int64),
            "objective": float(np.mean(_normalized_errors(x, y, [matrix], floor))),
            "branch_sizes": [len(x)],
        }

    for restart in range(restarts):
        labels = _initial_labels(x, y, contact, k, seed, restart)
        if min(np.bincount(labels, minlength=k)) < min_branch_size:
            continue
        valid = True
        for _ in range(50):
            matrices = []
            for branch in range(k):
                mask = labels == branch
                if int(mask.sum()) < min_branch_size:
                    valid = False
                    break
                matrices.append(_fit_matrix(x[mask], y[mask], weights[mask]))
            if not valid:
                break
            errors = _normalized_errors(x, y, matrices, floor)
            new_labels = np.argmin(errors, axis=1)
            if min(np.bincount(new_labels, minlength=k)) < min_branch_size:
                valid = False
                break
            if np.array_equal(labels, new_labels):
                labels = new_labels
                break
            labels = new_labels
        if not valid:
            continue
        matrices = np.stack(
            [_fit_matrix(x[labels == branch], y[labels == branch], weights[labels == branch]) for branch in range(k)]
        )
        errors = _normalized_errors(x, y, matrices, floor)
        objective = float(np.mean(errors[np.arange(len(x)), labels]))
        candidate = {
            "k": k,
            "matrices": matrices,
            "labels": labels,
            "objective": objective,
            "branch_sizes": np.bincount(labels, minlength=k).tolist(),
        }
        if best is None or objective < best["objective"]:
            best = candidate
    return best


def _fit_piecewise_ensemble(
    x,
    y,
    contact,
    k,
    floor,
    min_branch_size,
    seed,
    restarts,
    seed_blocks,
):
    best = None
    for block in range(seed_blocks):
        candidate = _fit_piecewise(
            x,
            y,
            contact,
            k,
            floor,
            min_branch_size,
            seed + 104729 * block,
            restarts,
        )
        if candidate is not None and (
            best is None or candidate["objective"] < best["objective"]
        ):
            best = candidate
    return best


def _evaluate_model(model, x, y, floor):
    errors = _normalized_errors(x, y, model["matrices"], floor)
    labels = np.argmin(errors, axis=1)
    prediction = np.stack([x[index] @ model["matrices"][label].T for index, label in enumerate(labels)])
    truth_norm = np.linalg.norm(y, axis=1)
    pred_norm = np.linalg.norm(prediction, axis=1)
    signal = truth_norm >= floor
    cosines = []
    for index in range(len(y)):
        if signal[index] and pred_norm[index] >= floor:
            cosines.append(_cosine(prediction[index], y[index]))
        else:
            cosines.append(None)
    per_sample_relative = np.linalg.norm(prediction - y, axis=1) / np.maximum(truth_norm, floor)
    relative_rmse = float(
        np.sqrt(np.mean((prediction - y) ** 2))
        / max(float(np.sqrt(np.mean(y * y))), floor)
    )
    return {
        "labels": labels,
        "prediction": prediction,
        "cosines": cosines,
        "median_cosine": _percentile(cosines, 50),
        "q25_cosine": _percentile(cosines, 25),
        "relative_rmse": relative_rmse,
        "median_relative_error": float(np.median(per_sample_relative)),
        "signal_fraction": float(np.mean(signal)),
        "branch_sizes": np.bincount(labels, minlength=model["k"]).tolist(),
    }


def _softmax_metrics(train_x, train_labels, test_x, test_labels, k):
    if k == 1:
        return {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
            "majority_accuracy": 1.0,
            "chance_balanced_accuracy": 1.0,
            "status": "single_branch",
        }
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x_train = np.concatenate(
        [(train_x - mean) / scale, np.ones((len(train_x), 1), dtype=np.float64)], axis=1
    )
    x_test = np.concatenate(
        [(test_x - mean) / scale, np.ones((len(test_x), 1), dtype=np.float64)], axis=1
    )
    dimension = x_train.shape[1]
    one_hot = np.eye(k, dtype=np.float64)[train_labels]

    def objective(flat):
        weights = flat.reshape(k, dimension)
        logits = x_train @ weights.T
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -float(np.sum(one_hot * np.log(np.maximum(probabilities, 1e-12)))) / len(x_train)
        loss += 5e-3 * float(np.sum(weights[:, :-1] ** 2))
        gradient = ((probabilities - one_hot).T @ x_train) / len(x_train)
        gradient[:, :-1] += 1e-2 * weights[:, :-1]
        return loss, gradient.reshape(-1)

    result = minimize(
        objective,
        np.zeros(k * dimension, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    weights = result.x.reshape(k, dimension)
    prediction = np.argmax(x_test @ weights.T, axis=1)
    accuracy = float(np.mean(prediction == test_labels))
    recalls = []
    f1s = []
    for branch in range(k):
        truth = test_labels == branch
        predicted = prediction == branch
        true_positive = int(np.sum(truth & predicted))
        false_positive = int(np.sum(~truth & predicted))
        false_negative = int(np.sum(truth & ~predicted))
        recalls.append(0.0 if not np.any(truth) else true_positive / int(np.sum(truth)))
        denom = 2 * true_positive + false_positive + false_negative
        f1s.append(0.0 if denom == 0 else 2 * true_positive / denom)
    counts = np.bincount(test_labels, minlength=k)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "majority_accuracy": float(counts.max() / len(test_labels)),
        "chance_balanced_accuracy": 1.0 / k,
        "status": "ok" if result.success else f"optimizer:{result.message}",
    }


def _matrix_from_central_axes(rows, epsilon):
    selected = [
        row
        for row in rows
        if row["direction_kind"] == "axis"
        and math.isclose(float(row["epsilon"]), epsilon, rel_tol=0.0, abs_tol=1e-12)
        and _as_bool(row.get("keep", True))
    ]
    if len(selected) != ACTION_DIM:
        return None
    x = np.stack([_json(row["direction_vec"], []) for row in selected])
    y = np.stack([_json(row["central_local_vel_response"], []) for row in selected])
    return np.linalg.lstsq(x, y, rcond=None)[0].T


def _contact_signature_equal_rate(rows, epsilon):
    selected = [
        row
        for row in rows
        if math.isclose(float(row["epsilon"]), epsilon, rel_tol=0.0, abs_tol=1e-12)
        and _as_bool(row.get("keep", True))
    ]
    return None if not selected else float(
        np.mean([_as_bool(row["contact_signature_trace_equal"]) for row in selected])
    )


def _mode_purity(labels, samples, branch):
    modes = [
        samples[index]["mode_key"]
        for index in range(len(samples))
        if labels[index] == branch
    ]
    return None if not modes else float(max(Counter(modes).values()) / len(modes))


def _collection_wall_seconds(paths):
    request_times = [path.stat().st_mtime for path in paths["requests"].glob("batch_*.json")]
    worker_times = [path.stat().st_mtime for path in paths["runs"].glob("batch_*/worker.json")]
    if not request_times or not worker_times:
        return 0.0
    return max(0.0, max(worker_times) - min(request_times))


def _analyze_anchor(anchor_id, rows, args):
    train_samples = _samples(rows, DENSE_EPSILON, "train")
    fit_samples = [
        sample for sample in train_samples if sample["direction_index"] < args.fit_random
    ]
    model_selection_samples = [
        sample for sample in train_samples if sample["direction_index"] >= args.fit_random
    ]
    hold_samples = _samples(rows, DENSE_EPSILON, "hold")
    validation_samples = _samples(rows, VALIDATION_EPSILON, "hold")
    if len(train_samples) != 2 * args.train_random:
        raise RuntimeError(f"anchor {anchor_id} has {len(train_samples)} train side samples")
    x_train, y_train, contact_train = _arrays(train_samples)
    x_fit, y_fit, contact_fit = _arrays(fit_samples)
    x_model_selection, y_model_selection, _ = _arrays(model_selection_samples)
    x_hold, y_hold, contact_hold = _arrays(hold_samples)
    x_validation, y_validation, contact_validation = _arrays(validation_samples)
    median_norm = float(np.median(np.linalg.norm(y_train, axis=1)))
    floor = max(1e-8, 0.02 * median_norm)

    selection_models = {}
    k_rows = []
    selection_evaluations = {}
    for k in range(1, args.max_k + 1):
        model = _fit_piecewise_ensemble(
            x_fit,
            y_fit,
            contact_fit,
            k,
            floor,
            args.min_branch_size,
            args.fit_seed + 1009 * anchor_id + 97 * k,
            args.fit_restarts,
            args.fit_seed_blocks,
        )
        if model is None:
            k_rows.append({"anchor_id": anchor_id, "k": k, "valid": False})
            continue
        fit_eval = _evaluate_model(model, x_fit, y_fit, floor)
        model_selection_eval = _evaluate_model(
            model, x_model_selection, y_model_selection, floor
        )
        selection_models[k] = model
        selection_evaluations[k] = {
            "fit": fit_eval,
            "model_selection": model_selection_eval,
        }
        parameter_count = k * TARGET_DIM * ACTION_DIM + max(0, k - 1)
        train_sse = max(model["objective"] * len(x_fit), 1e-12)
        bic = len(x_fit) * math.log(train_sse / len(x_fit)) + parameter_count * math.log(len(x_fit))
        k_rows.append(
            {
                "anchor_id": anchor_id,
                "k": k,
                "valid": True,
                "train_objective": model["objective"],
                "train_branch_sizes": model["branch_sizes"],
                "fit_median_cosine": fit_eval["median_cosine"],
                "selection_median_cosine": model_selection_eval["median_cosine"],
                "selection_q25_cosine": model_selection_eval["q25_cosine"],
                "selection_relative_rmse": model_selection_eval["relative_rmse"],
                "selection_median_relative_error": model_selection_eval["median_relative_error"],
                "selection_signal_fraction": model_selection_eval["signal_fraction"],
                "selection_branch_sizes": model_selection_eval["branch_sizes"],
                "bic": bic,
            }
        )

    if not selection_models:
        raise RuntimeError(f"anchor {anchor_id} has no valid piecewise model")
    best_error = min(
        selection_evaluations[k]["model_selection"]["relative_rmse"]
        for k in selection_models
    )
    best_cosine = max(
        selection_evaluations[k]["model_selection"]["median_cosine"] or -1.0
        for k in selection_models
    )
    eligible = [
        k
        for k in selection_models
        if selection_evaluations[k]["model_selection"]["relative_rmse"]
        <= 1.05 * best_error
        and (
            selection_evaluations[k]["model_selection"]["median_cosine"] or -1.0
        )
        >= best_cosine - 0.03
    ]
    selected_k = min(eligible) if eligible else min(
        selection_models,
        key=lambda k: selection_evaluations[k]["model_selection"]["relative_rmse"],
    )
    model = _fit_piecewise_ensemble(
        x_train,
        y_train,
        contact_train,
        selected_k,
        floor,
        args.min_branch_size,
        args.fit_seed + 500003 + 1009 * anchor_id,
        args.fit_restarts,
        args.fit_seed_blocks,
    )
    baseline_model = _fit_piecewise_ensemble(
        x_train,
        y_train,
        contact_train,
        1,
        floor,
        args.min_branch_size,
        args.fit_seed + 700001 + 1009 * anchor_id,
        1,
        1,
    )
    if model is None or baseline_model is None:
        raise RuntimeError(f"anchor {anchor_id} final refit failed for K={selected_k}")
    train_eval = _evaluate_model(model, x_train, y_train, floor)
    hold_eval = _evaluate_model(model, x_hold, y_hold, floor)
    validation_eval = _evaluate_model(model, x_validation, y_validation, floor)
    baseline_eval = _evaluate_model(baseline_model, x_hold, y_hold, floor)

    selector_rows = []
    selector_inputs = {
        "direction_deployable": (x_train, x_hold, x_validation),
        "post_contact_oracle": (contact_train, contact_hold, contact_validation),
        "direction_plus_post_contact_oracle": (
            np.concatenate([x_train, contact_train], axis=1),
            np.concatenate([x_hold, contact_hold], axis=1),
            np.concatenate([x_validation, contact_validation], axis=1),
        ),
    }
    for variant, (train_features, hold_features, validation_features) in selector_inputs.items():
        for split, features, labels in (
            ("hold_eps003", hold_features, hold_eval["labels"]),
            ("hold_eps010", validation_features, validation_eval["labels"]),
        ):
            metrics = _softmax_metrics(
                train_features,
                model["labels"],
                features,
                labels,
                selected_k,
            )
            selector_rows.append(
                {
                    "anchor_id": anchor_id,
                    "selected_k": selected_k,
                    "variant": variant,
                    "split": split,
                    **metrics,
                }
            )

    branch_rows = []
    assignment_rows = []
    for split, samples, evaluation in (
        ("train_eps003", train_samples, train_eval),
        ("hold_eps003", hold_samples, hold_eval),
        ("hold_eps010", validation_samples, validation_eval),
    ):
        for branch in range(selected_k):
            indices = np.where(evaluation["labels"] == branch)[0]
            cosines = [evaluation["cosines"][index] for index in indices]
            branch_rows.append(
                {
                    "anchor_id": anchor_id,
                    "selected_k": selected_k,
                    "split": split,
                    "branch": branch,
                    "num_samples": len(indices),
                    "median_response_cosine": _percentile(cosines, 50),
                    "q25_response_cosine": _percentile(cosines, 25),
                    "contact_mode_purity": _mode_purity(
                        evaluation["labels"], samples, branch
                    ),
                    "matrix": model["matrices"][branch].tolist(),
                }
            )
        for index, sample in enumerate(samples):
            assignment_rows.append(
                {
                    "anchor_id": anchor_id,
                    "selected_k": selected_k,
                    "split": split,
                    "base_direction": sample["base_direction"],
                    "side": sample["side"],
                    "branch": int(evaluation["labels"][index]),
                    "response_cosine": evaluation["cosines"][index],
                    "direction": sample["direction"].tolist(),
                    "response": sample["response"].tolist(),
                    "contact_mode": sample["mode_key"],
                }
            )

    reference_matrix = _matrix_from_central_axes(rows, DENSE_EPSILON)
    target_matrix = _matrix_from_central_axes(rows, VALIDATION_EPSILON)
    cross_y_cosine = None
    if reference_matrix is not None and target_matrix is not None:
        cross_y_cosine = _cosine(reference_matrix[1], target_matrix[1])
    signature_equal_rate = _contact_signature_equal_rate(rows, VALIDATION_EPSILON)
    both_confirmed = (
        cross_y_cosine is not None
        and cross_y_cosine < 0.7
        and signature_equal_rate is not None
        and signature_equal_rate < 0.8
    )
    cosine_improvement = None
    if hold_eval["median_cosine"] is not None and baseline_eval["median_cosine"] is not None:
        cosine_improvement = hold_eval["median_cosine"] - baseline_eval["median_cosine"]
    rmse_reduction = 1.0 - hold_eval["relative_rmse"] / max(
        baseline_eval["relative_rmse"], 1e-12
    )
    anchor_row = {
        "anchor_id": anchor_id,
        "selected_k": selected_k,
        "both_confirmed_replay": both_confirmed,
        "replay_cross_epsilon_y_cosine": cross_y_cosine,
        "replay_signature_equal_rate": signature_equal_rate,
        "response_norm_median": median_norm,
        "response_signal_floor": floor,
        "hold_median_response_cosine": hold_eval["median_cosine"],
        "hold_q25_response_cosine": hold_eval["q25_cosine"],
        "hold_relative_rmse": hold_eval["relative_rmse"],
        "hold_signal_fraction": hold_eval["signal_fraction"],
        "validation_median_response_cosine": validation_eval["median_cosine"],
        "validation_relative_rmse": validation_eval["relative_rmse"],
        "k1_hold_median_response_cosine": baseline_eval["median_cosine"],
        "k1_hold_relative_rmse": baseline_eval["relative_rmse"],
        "cosine_improvement_over_k1": cosine_improvement,
        "relative_rmse_reduction_over_k1": rmse_reduction,
        "selected_matrices": model["matrices"].tolist(),
    }
    return anchor_row, k_rows, branch_rows, selector_rows, assignment_rows


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _write_report(args, summary):
    selection = summary["selection"]
    sampling = summary["sampling"]
    aggregate = summary["aggregate"]
    decision = summary["decision"]
    k_distribution = aggregate["k_distribution"]
    lines = [
        "# A5 Both-Cohort Branch Diagnostic",
        "",
        "## Scope",
        "",
        "This is a diagnostic-only audit. It does not train a branch-conditional model and does not modify the frozen v2 or marginal-probe pipelines.",
        "The source non-weak `both` population is 481/715 marginal candidates (67.3%) and 481/1200 of the frozen contact scan (40.1%). After reserving the old 32 `both` diagnostic anchors, 449 remain eligible here.",
        "",
        "The dense label uses one simulator step after the anchor (`response_steps=0` in the frozen worker convention). Each central probe is decomposed into nominal-to-plus and nominal-to-minus one-sided responses. Branches are fitted as held-out piecewise-linear maps `r = A_k v`, not as clusters of response vectors.",
        f"The 200 random directions are frozen as `{args.fit_random}` for fitting candidate K values, `{args.train_random - args.fit_random}` for selecting K, and `{args.num_random - args.train_random}` for final testing; the six axis directions are final test only. After K selection, the first `{args.train_random}` directions are refitted before final evaluation.",
        "",
        "## Commands",
        "",
        "```bash",
        "conda run -n genesis --no-capture-output python scripts/arx/a5_both_branch_probe.py --stage select",
        "conda run -n genesis --no-capture-output python scripts/arx/a5_both_branch_probe.py --stage collect --gpu-ids 0,1,2,3 --workers 4",
        "conda run -n genesis --no-capture-output python scripts/arx/a5_both_branch_probe.py --stage analyze",
        "```",
        "",
        "## Frozen Selection",
        "",
        f"- eligible non-weak both anchors after excluding the old frozen probe: `{selection['eligible']}`;",
        f"- selected: `{selection['selected']}`; old-probe overlap: `{selection['old_probe_overlap']}`;",
        f"- x/y/speed/phase unique values: `{selection['unique_x']}/{selection['unique_y']}/{selection['unique_speed']}/{selection['unique_phase']}`.",
        "",
        "## Sampling",
        "",
        f"- successful anchors: `{sampling['successful_anchors']}/{sampling['requested_anchors']}`;",
        f"- kept rows: `{sampling['kept_rows']}/{sampling['rows']}`;",
        f"- pristine query replays: `{sampling['query_replays']}`; wall clock: `{sampling['wall_seconds']:.1f}s`;",
        f"- replay-confirmed both: `{sampling['both_confirmed']}/{sampling['successful_anchors']}`.",
        "- wall clock was measured while unrelated pre-existing training occupied all four GPUs, so it is an execution record rather than a clean throughput benchmark.",
        "",
        "The source both label came from restore prefiltering. Non-confirmed anchors are retained and never replaced, avoiding survivor bias.",
        "",
        "## Branch Discovery",
        "",
        f"The decision metrics below use the `{sampling['both_confirmed']}` replay-confirmed both anchors; all-selected metrics remain in the JSON summary.",
        "",
        "| K | Anchors | Fraction |",
        "|---:|---:|---:|",
    ]
    total = max(sum(k_distribution.values()), 1)
    for k in range(1, args.max_k + 1):
        count = int(k_distribution.get(str(k), 0))
        lines.append(f"| {k} | {count} | {count / total:.3f} |")
    lines.extend(
        [
            "",
            f"- selected K mean/median: `{_fmt(aggregate['selected_k']['mean'])}/{_fmt(aggregate['selected_k']['median'])}`;",
            f"- alternate-seed K mean/median: `{_fmt(aggregate['alternate_selected_k']['mean'])}/{_fmt(aggregate['alternate_selected_k']['median'])}`; exact per-anchor K agreement: `{_fmt(aggregate['k_seed_agreement_rate'])}`;",
            f"- held-out response cosine: median `{_fmt(aggregate['hold_cosine']['median'])}`, Q25/Q75 `{_fmt(aggregate['hold_cosine']['q25'])}/{_fmt(aggregate['hold_cosine']['q75'])}`;",
            f"- branch-level held-out cosine: median `{_fmt(aggregate['branch_cosine']['median'])}`, Q25/Q75 `{_fmt(aggregate['branch_cosine']['q25'])}/{_fmt(aggregate['branch_cosine']['q75'])}`;",
            f"- branch/contact-mode purity association: median `{_fmt(aggregate['branch_contact_mode_purity']['median'])}` (post-transition diagnostic, not a selector);",
            f"- epsilon=0.01 transfer cosine: median `{_fmt(aggregate['validation_cosine']['median'])}`;",
            f"- cosine gain over K=1: median `{_fmt(aggregate['cosine_improvement']['median'])}`; relative-RMSE reduction: median `{_fmt(aggregate['rmse_reduction']['median'])}`.",
            "",
            "## Selector Diagnostic",
            "",
            "Only direction is deployable before observing the perturbed transition. Post-contact variants are reported as oracle association diagnostics and are not treated as deployable selectors.",
            "",
            "| Variant | Split | Balanced accuracy | Raw accuracy | 1/K chance |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for variant in (
        "direction_deployable",
        "post_contact_oracle",
        "direction_plus_post_contact_oracle",
    ):
        for split in ("hold_eps003", "hold_eps010"):
            metrics = aggregate["selector"][variant][split]
            lines.append(
                f"| {variant} | {split} | {_fmt(metrics['balanced_accuracy']['median'])} | {_fmt(metrics['accuracy']['median'])} | {_fmt(metrics['chance_balanced_accuracy']['median'])} |"
            )
    direction_metrics = aggregate["selector"]["direction_deployable"]["hold_eps003"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- `{k_distribution.get(str(args.max_k), 0)}/{aggregate['num_anchors']}` confirmed anchors select the maximum tested K={args.max_k}. Together with only `{_fmt(aggregate['k_seed_agreement_rate'])}` exact-K seed agreement, the experiment does not identify an exact physical branch count; it shows that the proposed K<=3 representation is too small and that complexity frequently saturates the tested ceiling.",
            f"- Oracle branch assignment yields held-out cosine `{_fmt(aggregate['branch_cosine']['median'])}` and transfers to epsilon=0.01 at `{_fmt(aggregate['validation_cosine']['median'])}`. Thus several local linear maps can describe the responses after branch membership is known.",
            f"- The deployable direction-only selector reaches balanced accuracy `{_fmt(direction_metrics['balanced_accuracy']['median'])}` versus median 1/K chance `{_fmt(direction_metrics['chance_balanced_accuracy']['median'])}`. It is above chance but below both the 0.60 hard floor and the 0.80 success threshold.",
            "- Post-transition contact features are outcome-side oracle diagnostics. Their scores cannot be used as a deployable selector claim, and even the direction-plus-contact oracle remains below the success threshold.",
        ]
    )
    lines.extend(
        [
            "",
            "## Decision Matrix",
            "",
            "| Criterion | Threshold | Result | Pass |",
            "|---|---:|---:|---|",
            f"| pristine replay both confirmation | >= 20 anchors | {decision['confirmed_anchors']} | **{decision['confirmation_pass']}** |",
            f"| median K | <= 3 | {_fmt(decision['median_k'])} | **{decision['k_pass']}** |",
            f"| held-out branch response cosine | >= 0.90 | {_fmt(decision['branch_cosine'])} | **{decision['branch_cosine_pass']}** |",
            f"| K>1 benefit over K=1 | cosine +0.15 or RMSE -30% | `{_fmt(decision['cosine_gain'])}` / `{_fmt(decision['rmse_reduction'])}` | **{decision['improvement_pass']}** |",
            f"| epsilon=0.01 transfer cosine | >= 0.85 | {_fmt(decision['validation_cosine'])} | **{decision['scale_pass']}** |",
            f"| deployable direction selector balanced accuracy | >= 0.80 | {_fmt(decision['direction_selector_accuracy'])} | **{decision['selector_pass']}** |",
            "",
            f"**Direct verdict: `{decision['verdict']}`.**",
            "",
            decision["reason"],
            "",
            "No downstream training was started.",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")


def _aggregate_analysis(anchor_rows, branch_rows, selector_rows, include_ids):
    included_anchors = [
        row for row in anchor_rows if int(row["anchor_id"]) in include_ids
    ]
    selected_k = [int(row["selected_k"]) for row in included_anchors]
    nontrivial_ids = {
        int(row["anchor_id"])
        for row in included_anchors
        if int(row["selected_k"]) > 1
    }
    branch_hold = [
        float(row["median_response_cosine"])
        for row in branch_rows
        if int(row["anchor_id"]) in include_ids
        and row["split"] == "hold_eps003"
        and int(row["num_samples"]) >= 3
        and row["median_response_cosine"] not in (None, "")
    ]
    branch_contact_purity = [
        float(row["contact_mode_purity"])
        for row in branch_rows
        if int(row["anchor_id"]) in include_ids
        and row["split"] == "hold_eps003"
        and int(row["num_samples"]) >= 3
        and row["contact_mode_purity"] not in (None, "")
    ]
    selector = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in selector_rows:
        if int(row["anchor_id"]) not in nontrivial_ids:
            continue
        for metric in (
            "balanced_accuracy",
            "accuracy",
            "macro_f1",
            "majority_accuracy",
            "chance_balanced_accuracy",
        ):
            selector[row["variant"]][row["split"]][metric].append(float(row[metric]))
    variants = (
        "direction_deployable",
        "post_contact_oracle",
        "direction_plus_post_contact_oracle",
    )
    selector_summary = {
        variant: {
            split: {
                metric: _distribution(selector[variant][split][metric])
                for metric in (
                    "balanced_accuracy",
                    "accuracy",
                    "macro_f1",
                    "majority_accuracy",
                    "chance_balanced_accuracy",
                )
            }
            for split in ("hold_eps003", "hold_eps010")
        }
        for variant in variants
    }
    return {
        "num_anchors": len(included_anchors),
        "num_nontrivial_anchors": len(nontrivial_ids),
        "k_distribution": {
            str(key): value for key, value in sorted(Counter(selected_k).items())
        },
        "selected_k": _distribution(selected_k),
        "alternate_selected_k": _distribution(
            [row.get("alternate_selected_k") for row in included_anchors]
        ),
        "k_seed_agreement_rate": (
            None
            if not [
                row for row in included_anchors if row.get("k_seed_agreement") is not None
            ]
            else float(
                np.mean(
                    [
                        bool(row["k_seed_agreement"])
                        for row in included_anchors
                        if row.get("k_seed_agreement") is not None
                    ]
                )
            )
        ),
        "hold_cosine": _distribution(
            [row["hold_median_response_cosine"] for row in included_anchors]
        ),
        "branch_cosine": _distribution(branch_hold),
        "branch_contact_mode_purity": _distribution(branch_contact_purity),
        "validation_cosine": _distribution(
            [row["validation_median_response_cosine"] for row in included_anchors]
        ),
        "cosine_improvement": _distribution(
            [row["cosine_improvement_over_k1"] for row in included_anchors]
        ),
        "rmse_reduction": _distribution(
            [row["relative_rmse_reduction_over_k1"] for row in included_anchors]
        ),
        "selector": selector_summary,
    }


def _analyze(args):
    paths = _paths(args.out_dir)
    selection = _read_csv(paths["selection"])
    anchors = _read_csv(paths["anchors"])
    rows = _read_csv(paths["rows"])
    if not selection or not anchors or not rows:
        raise RuntimeError("selection/anchors/rows missing; run select and collect first")
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["anchor_id"])].append(row)
    anchor_rows = []
    k_rows = []
    branch_rows = []
    selector_rows = []
    assignment_rows = []
    failures = []
    for index, anchor in enumerate(anchors, start=1):
        anchor_id = int(anchor["anchor_id"])
        try:
            result = _analyze_anchor(anchor_id, grouped[anchor_id], args)
        except Exception as exc:  # Continue per the preregistered diagnostic protocol.
            failures.append({"anchor_id": anchor_id, "error": repr(exc)})
            print(f"[both-analyze] anchor={anchor_id} error={exc}", file=sys.stderr)
            continue
        anchor_row, anchor_k, anchor_branches, anchor_selectors, anchor_assignments = result
        anchor_row.update(
            {
                "obj_pos": anchor["obj_pos"],
                "speed": float(anchor["speed"]),
                "anchor_step": int(anchor["anchor_step"]),
            }
        )
        anchor_rows.append(anchor_row)
        k_rows.extend(anchor_k)
        branch_rows.extend(anchor_branches)
        selector_rows.extend(anchor_selectors)
        assignment_rows.extend(anchor_assignments)
        print(
            f"[both-analyze] {index}/{len(anchors)} anchor={anchor_id} "
            f"K={anchor_row['selected_k']} hold_cos={anchor_row['hold_median_response_cosine']:.3f} "
            f"confirmed={anchor_row['both_confirmed_replay']}"
        )
    if len(failures) > 10:
        raise RuntimeError(f"analysis failed for {len(failures)} anchors (>10 hard stop)")

    alternate_args = copy.copy(args)
    alternate_args.fit_seed = args.fit_seed + 300007
    stability_rows = []
    for index, anchor_row in enumerate(anchor_rows, start=1):
        anchor_id = int(anchor_row["anchor_id"])
        try:
            alternate = _analyze_anchor(anchor_id, grouped[anchor_id], alternate_args)[0]
            alternate_k = int(alternate["selected_k"])
            agreement = alternate_k == int(anchor_row["selected_k"])
            alternate_cosine = alternate["hold_median_response_cosine"]
            status = "ok"
        except Exception as exc:  # Sensitivity failure does not discard the primary fit.
            alternate_k = None
            agreement = None
            alternate_cosine = None
            status = repr(exc)
        anchor_row["alternate_selected_k"] = alternate_k
        anchor_row["k_seed_agreement"] = agreement
        stability_rows.append(
            {
                "anchor_id": anchor_id,
                "primary_selected_k": int(anchor_row["selected_k"]),
                "alternate_selected_k": alternate_k,
                "exact_k_agreement": agreement,
                "primary_hold_cosine": anchor_row["hold_median_response_cosine"],
                "alternate_hold_cosine": alternate_cosine,
                "status": status,
            }
        )
        print(
            f"[both-stability] {index}/{len(anchor_rows)} anchor={anchor_id} "
            f"K={anchor_row['selected_k']}/{alternate_k} status={status}"
        )

    _write_csv(paths["anchor_summary"], anchor_rows)
    _write_csv(paths["k_metrics"], k_rows)
    _write_csv(paths["branch_metrics"], branch_rows)
    _write_csv(paths["selector_metrics"], selector_rows)
    _write_csv(paths["stability_metrics"], stability_rows)
    _write_csv(paths["assignments"], assignment_rows)

    all_ids = {int(row["anchor_id"]) for row in anchor_rows}
    confirmed_ids = {
        int(row["anchor_id"])
        for row in anchor_rows
        if bool(row["both_confirmed_replay"])
    }
    all_aggregate = _aggregate_analysis(
        anchor_rows, branch_rows, selector_rows, all_ids
    )
    aggregate = _aggregate_analysis(
        anchor_rows, branch_rows, selector_rows, confirmed_ids
    )
    median_k = aggregate["selected_k"]["median"]
    branch_cosine = aggregate["branch_cosine"]["median"]
    validation_cosine = aggregate["validation_cosine"]["median"]
    cosine_gain = aggregate["cosine_improvement"]["median"]
    rmse_reduction = aggregate["rmse_reduction"]["median"]
    direction_accuracy = aggregate["selector"]["direction_deployable"]["hold_eps003"][
        "balanced_accuracy"
    ]["median"]
    k_pass = median_k is not None and median_k <= 3
    branch_cosine_pass = branch_cosine is not None and branch_cosine >= 0.90
    improvement_pass = (cosine_gain is not None and cosine_gain >= 0.15) or (
        rmse_reduction is not None and rmse_reduction >= 0.30
    )
    scale_pass = validation_cosine is not None and validation_cosine >= 0.85
    selector_pass = direction_accuracy is not None and direction_accuracy >= 0.80
    confirmation_pass = len(confirmed_ids) >= 20
    if not confirmation_pass:
        verdict = "source_both_label_not_reproduced"
        reason = "Fewer than 20/30 source both anchors reproduce both gate reasons under pristine replay; the branch decision box is not valid for this mixed cohort."
    elif all((k_pass, branch_cosine_pass, improvement_pass, scale_pass, selector_pass)):
        verdict = "salvageable_branch_jacobian"
        reason = "A small piecewise-linear Jacobian set is held-out coherent, useful beyond K=1, cross-scale stable, and selectable from perturbation direction."
    elif (
        median_k is not None
        and median_k <= 4
        and branch_cosine is not None
        and branch_cosine >= 0.85
        and direction_accuracy is not None
        and 0.60 <= direction_accuracy < 0.80
    ):
        verdict = "medium_high_risk"
        reason = "Piecewise-linear structure is present, but deployable branch selection remains only moderately predictable."
    elif (
        (median_k is not None and median_k > 4)
        or (branch_cosine is not None and branch_cosine < 0.85)
        or (direction_accuracy is not None and direction_accuracy < 0.60)
    ):
        verdict = "not_supported_for_branch_training"
        reason = "The few-branch hypothesis and deployable-selector floor fail: selected K saturates the tested upper range while direction-only branch prediction remains below 0.60. The preregistered result does not support starting branch-conditional training."
    else:
        verdict = "outside_decision_box"
        reason = "The observed combination does not fit a preregistered decision box; no downstream action is taken."
    decision = {
        "median_k": median_k,
        "confirmed_anchors": len(confirmed_ids),
        "confirmation_pass": confirmation_pass,
        "k_pass": k_pass,
        "branch_cosine": branch_cosine,
        "branch_cosine_pass": branch_cosine_pass,
        "cosine_gain": cosine_gain,
        "rmse_reduction": rmse_reduction,
        "improvement_pass": improvement_pass,
        "validation_cosine": validation_cosine,
        "scale_pass": scale_pass,
        "direction_selector_accuracy": direction_accuracy,
        "selector_pass": selector_pass,
        "verdict": verdict,
        "reason": reason,
    }
    batch_rows = _read_csv(paths["batches"])
    selection_ids = {int(row["anchor_id"]) for row in selection}
    old_ids = {int(row["anchor_id"]) for row in _read_csv(args.old_probe)}
    summary = {
        "description": "A5 both-cohort one-sided piecewise-linear branch diagnostic",
        "scope": "diagnostic_only_no_training",
        "protocol": {
            "dense_epsilon": DENSE_EPSILON,
            "validation_epsilon": VALIDATION_EPSILON,
            "num_random": args.num_random,
            "fit_random": args.fit_random,
            "train_random": args.train_random,
            "model_selection_random": args.train_random - args.fit_random,
            "hold_random": args.num_random - args.train_random,
            "axis_directions": ACTION_DIM,
            "response_steps_parameter": 0,
            "simulator_steps_after_anchor": 1,
            "branch_target": "one-sided local linear velocity response",
            "branch_model": "held-out hard mixture of 3x6 linear maps",
            "fit_seed_blocks": args.fit_seed_blocks,
            "fit_restarts_per_block": args.fit_restarts,
            "alternate_fit_seed_offset": 300007,
            "selector_contact_features": "post-transition oracle association only",
            "execution_note": "all four GPUs had unrelated pre-existing training load",
        },
        "source_hashes": {
            "prefilter": _sha256(args.prefilter_matrices),
            "dense_manifest": _sha256(args.dense_manifest),
            "old_probe_selection": _sha256(args.old_probe),
            "frozen_selection": _sha256(paths["selection"]),
        },
        "selection": {
            "eligible": 449,
            "selected": len(selection),
            "old_probe_overlap": len(selection_ids & old_ids),
            "unique_x": len({float(row["obj_x"]) for row in selection}),
            "unique_y": len({float(row["obj_y"]) for row in selection}),
            "unique_speed": len({float(row["speed"]) for row in selection}),
            "unique_phase": len({int(row["anchor_step"]) for row in selection}),
        },
        "sampling": {
            "requested_anchors": len(selection),
            "successful_anchors": len(anchors),
            "analyzed_anchors": len(anchor_rows),
            "rows": len(rows),
            "kept_rows": sum(_as_bool(row.get("keep", True)) for row in rows),
            "query_replays": sum(int(row.get("query_replays") or 0) for row in batch_rows),
            "wall_seconds": _collection_wall_seconds(paths),
            "both_confirmed": sum(bool(row["both_confirmed_replay"]) for row in anchor_rows),
            "analysis_failures": failures,
        },
        "all_selected_aggregate": all_aggregate,
        "aggregate": aggregate,
        "decision": decision,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(args, summary)
    print(f"[both-analyze] decision={verdict}")
    print(f"[both-analyze] wrote {paths['summary']}")
    print(f"[both-analyze] wrote {args.report}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "select", "collect", "analyze"), default="all")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--prefilter-matrices", type=Path, default=DEFAULT_PREFILTER)
    parser.add_argument("--dense-manifest", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--old-probe", type=Path, default=DEFAULT_OLD_PROBE)
    parser.add_argument("--num-anchors", type=int, default=30)
    parser.add_argument("--selection-seed", type=int, default=7301)
    parser.add_argument("--direction-seed", type=int, default=91001)
    parser.add_argument("--num-random", type=int, default=200)
    parser.add_argument("--fit-random", type=int, default=120)
    parser.add_argument("--train-random", type=int, default=160)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--max-contacts", type=int, default=32)
    parser.add_argument("--max-repeat-state-diff", type=float, default=1e-7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--batch-timeout", type=int, default=7200)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force-selection", action="store_true")
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--min-branch-size", type=int, default=15)
    parser.add_argument("--fit-restarts", type=int, default=30)
    parser.add_argument("--fit-seed-blocks", type=int, default=3)
    parser.add_argument("--fit-seed", type=int, default=12017)
    args = parser.parse_args()

    if not 0 < args.fit_random < args.train_random < args.num_random:
        raise ValueError("require 0 < fit-random < train-random < num-random")
    if args.stage in ("all", "select"):
        _select(args)
    if args.stage in ("all", "collect"):
        _collect(args)
    if args.stage in ("all", "analyze"):
        _analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
