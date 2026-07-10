"""Pristine-replay worker for the A5 both-cohort branch diagnostic."""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import genesis as gs
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from a5_action_vjp_v2_collect_worker import (  # noqa: E402
    _max_abs,
    _prepare_anchor,
    _query_replay,
)
from a5_pusher_forward_sanity import _make_scene  # noqa: E402


ACTION_DIM = 6
TARGET_DIM = 3


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


def _directions(seed, num_random, train_random):
    directions = []
    for index in range(ACTION_DIM):
        vector = np.zeros(ACTION_DIM, dtype=np.float64)
        vector[index] = 1.0
        directions.append(
            {
                "name": f"axis_{index + 1}",
                "kind": "axis",
                "index": index,
                "role": "hold",
                "vec": vector.tolist(),
            }
        )

    rng = np.random.default_rng(seed)
    for index in range(num_random):
        vector = rng.normal(size=ACTION_DIM)
        vector /= np.linalg.norm(vector)
        directions.append(
            {
                "name": f"random_{index + 1:03d}",
                "kind": "random",
                "index": index,
                "role": "train" if index < train_random else "hold",
                "vec": vector.tolist(),
            }
        )
    return directions


def _response(state, nominal, epsilon):
    state = np.asarray(state, dtype=np.float64)
    nominal = np.asarray(nominal, dtype=np.float64)
    return ((state - nominal) / epsilon).tolist()


def _central_response(plus, minus, epsilon):
    plus = np.asarray(plus, dtype=np.float64)
    minus = np.asarray(minus, dtype=np.float64)
    return ((plus - minus) / (2.0 * epsilon)).tolist()


def _signature_trace(record):
    return [item["signature"] for item in record["contact_geometry_trace"]]


def _finite(*vectors):
    return all(math.isfinite(float(value)) for vector in vectors for value in vector)


def _collect_anchor(scene, arm, obj, base_state, job, config):
    start = time.perf_counter()
    anchor = _prepare_anchor(scene, arm, obj, base_state, job, config)
    label_steps = 1

    def replay(action):
        return _query_replay(
            scene,
            arm,
            obj,
            base_state,
            job,
            action,
            label_steps,
            config,
            geometry_mode="full",
        )

    nominal = replay(job["qvel"])
    repeat = replay(job["qvel"])
    nominal_state = nominal["object"]["pos"] + nominal["object"]["qvel"]
    repeat_state = repeat["object"]["pos"] + repeat["object"]["qvel"]
    repeat_state_max_abs = _max_abs(nominal_state, repeat_state)
    nominal_contact_events = sum(
        int(item["count"]) for item in nominal["contact_geometry_trace"]
    )
    arm_contact_events = int(anchor["contact"]["count"]) + nominal_contact_events

    rows = []
    nominal_qvel = np.asarray(nominal["object"]["qvel"], dtype=np.float64)
    nominal_action = np.asarray(job["qvel"], dtype=np.float64)
    for direction in _directions(job["branch_seed"], config["num_random"], config["train_random"]):
        vector = np.asarray(direction["vec"], dtype=np.float64)
        epsilons = [config["dense_epsilon"]]
        if direction["role"] == "hold":
            epsilons.append(config["validation_epsilon"])
        for epsilon in epsilons:
            plus = replay((nominal_action + epsilon * vector).tolist())
            minus = replay((nominal_action - epsilon * vector).tolist())
            plus_qvel = np.asarray(plus["object"]["qvel"], dtype=np.float64)
            minus_qvel = np.asarray(minus["object"]["qvel"], dtype=np.float64)
            plus_linear = _response(plus_qvel[:TARGET_DIM], nominal_qvel[:TARGET_DIM], epsilon)
            minus_linear = _response(minus_qvel[:TARGET_DIM], nominal_qvel[:TARGET_DIM], epsilon)
            central_linear = _central_response(
                plus_qvel[:TARGET_DIM], minus_qvel[:TARGET_DIM], epsilon
            )
            plus_angular = _response(plus_qvel[TARGET_DIM:6], nominal_qvel[TARGET_DIM:6], epsilon)
            minus_angular = _response(minus_qvel[TARGET_DIM:6], nominal_qvel[TARGET_DIM:6], epsilon)
            central_angular = _central_response(
                plus_qvel[TARGET_DIM:6], minus_qvel[TARGET_DIM:6], epsilon
            )
            finite = _finite(
                plus_linear,
                minus_linear,
                central_linear,
                plus_angular,
                minus_angular,
                central_angular,
            )
            rows.append(
                {
                    "anchor_id": job["anchor_id"],
                    "direction": direction["name"],
                    "direction_kind": direction["kind"],
                    "direction_index": direction["index"],
                    "direction_role": direction["role"],
                    "direction_vec": direction["vec"],
                    "epsilon": epsilon,
                    "response_steps": 0,
                    "simulator_steps_after_anchor": 1,
                    "branch_mode": "pristine_replay",
                    "plus_local_vel_response": plus_linear,
                    "minus_local_vel_response": minus_linear,
                    "central_local_vel_response": central_linear,
                    "plus_local_ang_response": plus_angular,
                    "minus_local_ang_response": minus_angular,
                    "central_local_ang_response": central_angular,
                    "plus_object_qvel": plus["object"]["qvel"],
                    "minus_object_qvel": minus["object"]["qvel"],
                    "plus_contact_trace": plus["contact_trace"],
                    "minus_contact_trace": minus["contact_trace"],
                    "plus_contact_geometry_trace": plus["contact_geometry_trace"],
                    "minus_contact_geometry_trace": minus["contact_geometry_trace"],
                    "plus_contact_signature_trace": _signature_trace(plus),
                    "minus_contact_signature_trace": _signature_trace(minus),
                    "contact_signature_trace_equal": _signature_trace(plus)
                    == _signature_trace(minus),
                    "finite": finite,
                    "keep": finite and repeat_state_max_abs <= config["max_repeat_state_diff"],
                }
            )

    return {
        "anchor": {
            "anchor_id": job["anchor_id"],
            "split": job["split"],
            "obj_pos": job["obj_pos"],
            "speed": job["speed"],
            "anchor_step": job["anchor_step"],
            "branch_seed": job["branch_seed"],
            "status": "ok" if arm_contact_events > 0 else "no_arm_object_contact",
            "branch_mode": "pristine_replay",
            "anchor_object_state": anchor["object"],
            "anchor_arm_state": anchor["arm"],
            "anchor_contact": anchor["contact"],
            "nominal_object_state": nominal["object"],
            "nominal_contact_trace": nominal["contact_trace"],
            "nominal_contact_geometry_trace": nominal["contact_geometry_trace"],
            "arm_contact_events": arm_contact_events,
            "repeat_state_max_abs": repeat_state_max_abs,
            "num_rows": len(rows),
            "num_kept_rows": sum(bool(row["keep"]) for row in rows),
            "query_replays": 2 + 2 * len(rows),
            "elapsed_seconds": time.perf_counter() - start,
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text())
    jobs = request["jobs"]
    config = request["config"]
    if not jobs:
        raise ValueError("request has no jobs")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene, arm, obj = _make_scene(tuple(jobs[0]["obj_pos"]), requires_grad=True)
    base_state = scene.get_state()
    anchors = []
    rows = []
    start = time.perf_counter()
    for index, job in enumerate(jobs, start=1):
        record = _collect_anchor(scene, arm, obj, base_state, job, config)
        anchors.append(record["anchor"])
        rows.extend(record["rows"])
        print(
            f"[both-branch-worker] {index}/{len(jobs)} anchor={job['anchor_id']} "
            f"rows={record['anchor']['num_kept_rows']}/{record['anchor']['num_rows']} "
            f"replays={record['anchor']['query_replays']} "
            f"elapsed={record['anchor']['elapsed_seconds']:.1f}s",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "rows.csv"
    anchors_path = args.out_dir / "anchors.csv"
    worker_path = args.out_dir / "worker.json"
    _write_csv(rows_path, rows)
    _write_csv(anchors_path, anchors)
    payload = {
        "description": "A5 both-cohort one-sided pristine replay worker",
        "request": str(args.request),
        "config": config,
        "num_jobs": len(jobs),
        "num_rows": len(rows),
        "num_kept_rows": sum(bool(row["keep"]) for row in rows),
        "query_replays": sum(int(row["query_replays"]) for row in anchors),
        "elapsed_seconds": time.perf_counter() - start,
    }
    worker_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[both-branch-worker] wrote {worker_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
