"""Long-lived Genesis worker for A5 action-side VJP v2 data collection."""

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import genesis as gs


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from a5_fd_response_dataset import _directions  # noqa: E402
from a5_pusher_forward_sanity import _contact_count, _make_scene  # noqa: E402


def _tensor_list(value):
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _entity_state(entity):
    state = entity.get_state()
    return {
        "pos": _tensor_list(state.pos)[:3],
        "quat": _tensor_list(state.quat)[:4],
        "qvel": _tensor_list(entity.get_dofs_velocity()),
    }


def _max_abs(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _world_to_local(quat, vector):
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    vx, vy, vz = np.asarray(vector, dtype=np.float64)
    return np.asarray(
        [
            (1.0 - 2.0 * (y * y + z * z)) * vx
            + 2.0 * (x * y + z * w) * vy
            + 2.0 * (x * z - y * w) * vz,
            2.0 * (x * y - z * w) * vx
            + (1.0 - 2.0 * (x * x + z * z)) * vy
            + 2.0 * (y * z + x * w) * vz,
            2.0 * (x * z + y * w) * vx
            + 2.0 * (y * z - x * w) * vy
            + (1.0 - 2.0 * (x * x + y * y)) * vz,
        ],
        dtype=np.float64,
    )


def _contact_geometry(obj, arm, max_contacts):
    data = obj.get_contacts(with_entity=arm)
    if not data or "geom_a" not in data:
        return {"count": 0, "signature": [], "contacts": [], "pooled": _empty_contact_pool()}

    arrays = {}
    for key, value in data.items():
        if key == "valid_mask":
            continue
        arrays[key] = value.detach().cpu().numpy()
    count = int(arrays["geom_a"].reshape(-1).shape[0])
    contacts = []
    obj_state = _entity_state(obj)
    obj_pos = np.asarray(obj_state["pos"], dtype=np.float64)
    obj_quat = np.asarray(obj_state["quat"], dtype=np.float64)
    for idx in range(count):
        geom_a = int(arrays["geom_a"].reshape(-1)[idx])
        geom_b = int(arrays["geom_b"].reshape(-1)[idx])
        link_a = int(arrays["link_a"].reshape(-1)[idx])
        link_b = int(arrays["link_b"].reshape(-1)[idx])
        position = np.asarray(arrays["position"].reshape(-1, 3)[idx], dtype=np.float64)
        normal = np.asarray(arrays["normal"].reshape(-1, 3)[idx], dtype=np.float64)
        penetration = float(arrays["penetration"].reshape(-1)[idx])
        object_is_a = obj.geom_start <= geom_a < obj.geom_end
        object_is_b = obj.geom_start <= geom_b < obj.geom_end
        if object_is_a:
            force = np.asarray(arrays["force_a"].reshape(-1, 3)[idx], dtype=np.float64)
            oriented_normal = normal
            object_side = "a"
        elif object_is_b:
            force = np.asarray(arrays["force_b"].reshape(-1, 3)[idx], dtype=np.float64)
            oriented_normal = -normal
            object_side = "b"
        else:
            force = np.zeros(3, dtype=np.float64)
            oriented_normal = normal
            object_side = "none"
        relative_world = position - obj_pos
        contacts.append(
            {
                "geom_a": geom_a,
                "geom_b": geom_b,
                "link_a": link_a,
                "link_b": link_b,
                "object_side": object_side,
                "position_world": position.tolist(),
                "position_object_world": relative_world.tolist(),
                "position_object": _world_to_local(obj_quat, relative_world).tolist(),
                "normal_raw": normal.tolist(),
                "normal_object_world": oriented_normal.tolist(),
                "normal_object": _world_to_local(obj_quat, oriented_normal).tolist(),
                "penetration": penetration,
                "force_object_world": force.tolist(),
                "force_object": _world_to_local(obj_quat, force).tolist(),
            }
        )
    contacts.sort(
        key=lambda item: (
            item["geom_a"],
            item["geom_b"],
            -abs(item["penetration"]),
            *item["position_world"],
        )
    )
    signature = [
        [item["geom_a"], item["geom_b"], item["link_a"], item["link_b"]]
        for item in contacts
    ]
    return {
        "count": count,
        "signature": signature,
        "contacts": contacts[:max_contacts],
        "pooled": _pool_contacts(contacts),
    }


def _contact_signature(obj, arm):
    data = obj.get_contacts(with_entity=arm)
    if not data or "geom_a" not in data:
        return {"count": 0, "signature": []}
    geom_a = data["geom_a"].detach().cpu().reshape(-1).tolist()
    geom_b = data["geom_b"].detach().cpu().reshape(-1).tolist()
    link_a = data["link_a"].detach().cpu().reshape(-1).tolist()
    link_b = data["link_b"].detach().cpu().reshape(-1).tolist()
    signature = sorted(
        [int(a), int(b), int(la), int(lb)]
        for a, b, la, lb in zip(geom_a, geom_b, link_a, link_b)
    )
    return {"count": len(signature), "signature": signature}


def _empty_contact_pool():
    return {
        "penetration_sum": 0.0,
        "penetration_mean": 0.0,
        "penetration_max": 0.0,
        "position_object_mean": [0.0, 0.0, 0.0],
        "normal_object_mean": [0.0, 0.0, 0.0],
        "force_object_sum": [0.0, 0.0, 0.0],
    }


def _pool_contacts(contacts):
    if not contacts:
        return _empty_contact_pool()
    penetration = np.asarray([abs(item["penetration"]) for item in contacts], dtype=np.float64)
    weights = penetration + 1e-12
    weights /= weights.sum()
    positions = np.asarray([item["position_object"] for item in contacts], dtype=np.float64)
    normals = np.asarray([item["normal_object"] for item in contacts], dtype=np.float64)
    forces = np.asarray([item["force_object"] for item in contacts], dtype=np.float64)
    return {
        "penetration_sum": float(penetration.sum()),
        "penetration_mean": float(penetration.mean()),
        "penetration_max": float(penetration.max()),
        "position_object_mean": (weights[:, None] * positions).sum(axis=0).tolist(),
        "normal_object_mean": (weights[:, None] * normals).sum(axis=0).tolist(),
        "force_object_sum": forces.sum(axis=0).tolist(),
    }


def _arm_context(arm):
    links_pos = arm.get_links_pos().detach().cpu().reshape(-1, 3)
    links_quat = arm.get_links_quat().detach().cpu().reshape(-1, 4)
    links_vel = arm.get_links_vel().detach().cpu().reshape(-1, 3)
    links_ang = arm.get_links_ang().detach().cpu().reshape(-1, 3)
    return {
        "qpos": _tensor_list(arm.get_qpos()),
        "qvel": _tensor_list(arm.get_dofs_velocity()),
        "tip_pos": _tensor_list(links_pos[-1]),
        "tip_quat": _tensor_list(links_quat[-1]),
        "tip_vel": _tensor_list(links_vel[-1]),
        "tip_ang": _tensor_list(links_ang[-1]),
    }


def _set_velocity(arm, values):
    arm.set_dofs_velocity(gs.tensor([float(value) for value in values]))


def _prepare_anchor(scene, arm, obj, base_state, job, config):
    # reset(state=...) replaces Genesis' registered initial state. Always seed a
    # new job from the pristine post-build state, not from the previous query.
    scene.reset(state=base_state)
    obj.set_pos(gs.tensor(job["obj_pos"]), zero_velocity=True)
    arm.set_dofs_position(torch.tensor(job["qpos"], dtype=torch.float32), zero_velocity=True)
    zero = [0.0] * len(job["qvel"])
    settle_contacts = []
    for _ in range(config["settle_steps"]):
        _set_velocity(arm, zero)
        scene.step()
        settle_contacts.append(_contact_count(scene))
    for _ in range(job["anchor_step"]):
        _set_velocity(arm, job["qvel"])
        scene.step()
    return {
        "state": scene.get_state(),
        "object": _entity_state(obj),
        "arm": _arm_context(arm),
        "contact": _contact_geometry(obj, arm, config["max_contacts"]),
        "settle_contacts": settle_contacts,
    }


def _query(
    scene,
    arm,
    obj,
    anchor_state,
    first_action,
    nominal_action,
    label_steps,
    max_contacts,
    geometry_mode="full",
):
    scene.reset(state=anchor_state)
    contacts = []
    geometries = []
    for step in range(label_steps):
        _set_velocity(arm, first_action if step == 0 else nominal_action)
        scene.step()
        contacts.append(_contact_count(scene))
        geometries.append(
            _contact_geometry(obj, arm, max_contacts)
            if geometry_mode == "full"
            else _contact_signature(obj, arm)
        )
    return {
        "object": _entity_state(obj),
        "contact_trace": contacts,
        "contact_geometry_trace": geometries,
    }


def _query_replay(
    scene, arm, obj, base_state, job, first_action, label_steps, config, geometry_mode="full"
):
    scene.reset(state=base_state)
    obj.set_pos(gs.tensor(job["obj_pos"]), zero_velocity=True)
    arm.set_dofs_position(torch.tensor(job["qpos"], dtype=torch.float32), zero_velocity=True)
    zero = [0.0] * len(job["qvel"])
    for _ in range(config["settle_steps"]):
        _set_velocity(arm, zero)
        scene.step()
    for _ in range(job["anchor_step"]):
        _set_velocity(arm, job["qvel"])
        scene.step()
    contacts = []
    geometries = []
    for step in range(label_steps):
        _set_velocity(arm, first_action if step == 0 else job["qvel"])
        scene.step()
        contacts.append(_contact_count(scene))
        geometries.append(
            _contact_geometry(obj, arm, config["max_contacts"])
            if geometry_mode == "full"
            else _contact_signature(obj, arm)
        )
    return {
        "object": _entity_state(obj),
        "contact_trace": contacts,
        "contact_geometry_trace": geometries,
    }


def _response(plus, minus, epsilon):
    return ((np.asarray(plus, dtype=np.float64) - np.asarray(minus, dtype=np.float64)) / (2.0 * epsilon)).tolist()


def _signature_trace(record):
    return [item["signature"] for item in record["contact_geometry_trace"]]


def _collect_job(scene, arm, obj, base_state, job, config):
    start = time.perf_counter()
    anchor = _prepare_anchor(scene, arm, obj, base_state, job, config)
    label_steps = config["response_steps"] + 1
    branch_mode = config.get("branch_mode", "replay")

    def run_query(first_action, geometry_mode="full"):
        if branch_mode == "replay":
            return _query_replay(
                scene,
                arm,
                obj,
                base_state,
                job,
                first_action,
                label_steps,
                config,
                geometry_mode,
            )
        if branch_mode == "restore":
            return _query(
                scene,
                arm,
                obj,
                anchor["state"],
                first_action,
                job["qvel"],
                label_steps,
                config["max_contacts"],
                geometry_mode,
            )
        raise ValueError(f"unknown branch_mode: {branch_mode}")

    nominal = run_query(job["qvel"])
    repeat = run_query(job["qvel"], "signature")
    repeat_state_diff = _max_abs(
        nominal["object"]["pos"] + nominal["object"]["qvel"],
        repeat["object"]["pos"] + repeat["object"]["qvel"],
    )

    nominal_arm_contact_events = sum(
        int(item["count"]) for item in nominal["contact_geometry_trace"]
    )
    arm_contact_events = int(anchor["contact"]["count"]) + nominal_arm_contact_events
    if config.get("require_arm_object_contact", True) and arm_contact_events < config.get(
        "min_arm_object_contact_events", 1
    ):
        return {
            "anchor_id": job["anchor_id"],
            "split": job["split"],
            "obj_pos": job["obj_pos"],
            "speed": job["speed"],
            "anchor_step": job["anchor_step"],
            "status": "no_arm_object_contact",
            "branch_mode": branch_mode,
            "anchor_object_state": anchor["object"],
            "anchor_arm_state": anchor["arm"],
            "anchor_contact": anchor["contact"],
            "nominal_object_state": nominal["object"],
            "nominal_contact_trace": nominal["contact_trace"],
            "nominal_contact_geometry_trace": nominal["contact_geometry_trace"],
            "nominal_arm_contact_events": nominal_arm_contact_events,
            "arm_contact_events": arm_contact_events,
            "repeat_state_max_abs": repeat_state_diff,
            "num_rows": 0,
            "num_kept_rows": 0,
            "elapsed_seconds": time.perf_counter() - start,
            "rows": [],
        }

    if not config.get("collect_probes", True):
        return {
            "anchor_id": job["anchor_id"],
            "split": job["split"],
            "obj_pos": job["obj_pos"],
            "speed": job["speed"],
            "anchor_step": job["anchor_step"],
            "status": "contact_candidate",
            "branch_mode": "scan_continuous",
            "anchor_object_state": anchor["object"],
            "anchor_arm_state": anchor["arm"],
            "anchor_contact": anchor["contact"],
            "nominal_object_state": nominal["object"],
            "nominal_contact_trace": nominal["contact_trace"],
            "nominal_contact_geometry_trace": nominal["contact_geometry_trace"],
            "nominal_arm_contact_events": nominal_arm_contact_events,
            "arm_contact_events": arm_contact_events,
            "repeat_state_max_abs": repeat_state_diff,
            "num_rows": 0,
            "num_kept_rows": 0,
            "elapsed_seconds": time.perf_counter() - start,
            "rows": [],
        }

    directions = _directions(job["qvel"], config["num_random"], job["seed"])
    rows = []
    for epsilon in config["epsilons"]:
        for direction in directions:
            vec = np.asarray(direction["vec"], dtype=np.float64)
            nominal_action = np.asarray(job["qvel"], dtype=np.float64)
            plus = run_query((nominal_action + epsilon * vec).tolist(), "signature")
            minus = run_query((nominal_action - epsilon * vec).tolist(), "signature")
            linear_velocity_response = _response(
                plus["object"]["qvel"][:3], minus["object"]["qvel"][:3], epsilon
            )
            angular_velocity_response = _response(
                plus["object"]["qvel"][3:6], minus["object"]["qvel"][3:6], epsilon
            )
            position_response = _response(plus["object"]["pos"], minus["object"]["pos"], epsilon)
            plus_signature = _signature_trace(plus)
            minus_signature = _signature_trace(minus)
            finite = all(
                math.isfinite(value)
                for value in linear_velocity_response + angular_velocity_response + position_response
            )
            rows.append(
                {
                    "anchor_id": job["anchor_id"],
                    "split": job["split"],
                    "obj_x": job["obj_pos"][0],
                    "obj_y": job["obj_pos"][1],
                    "obj_z": job["obj_pos"][2],
                    "speed": job["speed"],
                    "anchor_step": job["anchor_step"],
                    "response_steps": config["response_steps"],
                    "branch_mode": branch_mode,
                    "direction": direction["name"],
                    "direction_vec": direction["vec"],
                    "epsilon": epsilon,
                    "linear_velocity_response": linear_velocity_response,
                    "angular_velocity_response": angular_velocity_response,
                    "position_response": position_response,
                    "linear_velocity_response_norm": float(np.linalg.norm(linear_velocity_response)),
                    "position_response_norm": float(np.linalg.norm(position_response)),
                    "plus_object_pos": plus["object"]["pos"],
                    "minus_object_pos": minus["object"]["pos"],
                    "plus_object_qvel": plus["object"]["qvel"],
                    "minus_object_qvel": minus["object"]["qvel"],
                    "plus_contact_trace": plus["contact_trace"],
                    "minus_contact_trace": minus["contact_trace"],
                    "plus_contact_signature_trace": plus_signature,
                    "minus_contact_signature_trace": minus_signature,
                    "contact_count_trace_equal": plus["contact_trace"] == minus["contact_trace"],
                    "contact_signature_trace_equal": plus_signature == minus_signature,
                    "anchor_object_state": anchor["object"],
                    "anchor_arm_state": anchor["arm"],
                    "anchor_contact": anchor["contact"],
                    "nominal_object_state": nominal["object"],
                    "nominal_contact_trace": nominal["contact_trace"],
                    "nominal_contact_geometry_trace": nominal["contact_geometry_trace"],
                    "nominal_arm_contact_events": nominal_arm_contact_events,
                    "arm_contact_events": arm_contact_events,
                    "repeat_state_max_abs": repeat_state_diff,
                    "finite": finite,
                    "keep": finite and repeat_state_diff <= config["max_repeat_state_diff"],
                }
            )
    return {
        "anchor_id": job["anchor_id"],
        "split": job["split"],
        "obj_pos": job["obj_pos"],
        "speed": job["speed"],
        "anchor_step": job["anchor_step"],
        "status": "ok",
        "branch_mode": branch_mode,
        "anchor_object_state": anchor["object"],
        "anchor_arm_state": anchor["arm"],
        "anchor_contact": anchor["contact"],
        "nominal_object_state": nominal["object"],
        "nominal_contact_trace": nominal["contact_trace"],
        "nominal_contact_geometry_trace": nominal["contact_geometry_trace"],
        "nominal_arm_contact_events": nominal_arm_contact_events,
        "arm_contact_events": arm_contact_events,
        "repeat_state_max_abs": repeat_state_diff,
        "num_rows": len(rows),
        "num_kept_rows": sum(bool(row["keep"]) for row in rows),
        "elapsed_seconds": time.perf_counter() - start,
        "rows": rows,
    }


def _scan_group(scene, arm, obj, base_state, jobs, config):
    jobs = sorted(jobs, key=lambda job: job["anchor_step"])
    scene.reset(state=base_state)
    obj.set_pos(gs.tensor(jobs[0]["obj_pos"]), zero_velocity=True)
    arm.set_dofs_position(torch.tensor(jobs[0]["qpos"], dtype=torch.float32), zero_velocity=True)
    zero = [0.0] * len(jobs[0]["qvel"])
    for _ in range(config["settle_steps"]):
        _set_velocity(arm, zero)
        scene.step()

    records = []
    current_step = 0
    if config["response_steps"] != 0:
        raise ValueError("continuous grouped scan requires response_steps=0")
    for job in jobs:
        start = time.perf_counter()
        while current_step < job["anchor_step"]:
            _set_velocity(arm, job["qvel"])
            scene.step()
            current_step += 1
        anchor_object = _entity_state(obj)
        anchor_arm = _arm_context(arm)
        anchor_contact = _contact_geometry(obj, arm, config["max_contacts"])
        _set_velocity(arm, job["qvel"])
        scene.step()
        current_step += 1
        nominal_geometry = _contact_geometry(obj, arm, config["max_contacts"])
        nominal = {
            "object": _entity_state(obj),
            "contact_trace": [_contact_count(scene)],
            "contact_geometry_trace": [nominal_geometry],
        }
        nominal_arm_contact_events = sum(
            int(item["count"]) for item in nominal["contact_geometry_trace"]
        )
        arm_contact_events = int(anchor_contact["count"]) + nominal_arm_contact_events
        status = (
            "contact_candidate"
            if arm_contact_events >= config.get("min_arm_object_contact_events", 1)
            else "no_arm_object_contact"
        )
        records.append(
            {
                "anchor_id": job["anchor_id"],
                "split": job["split"],
                "obj_pos": job["obj_pos"],
                "speed": job["speed"],
                "anchor_step": job["anchor_step"],
                "status": status,
                "branch_mode": "scan_continuous",
                "anchor_object_state": anchor_object,
                "anchor_arm_state": anchor_arm,
                "anchor_contact": anchor_contact,
                "nominal_object_state": nominal["object"],
                "nominal_contact_trace": nominal["contact_trace"],
                "nominal_contact_geometry_trace": nominal["contact_geometry_trace"],
                "nominal_arm_contact_events": nominal_arm_contact_events,
                "arm_contact_events": arm_contact_events,
                "repeat_state_max_abs": None,
                "num_rows": 0,
                "num_kept_rows": 0,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
    return records


def _collect_scan_jobs(scene, arm, obj, base_state, jobs, config):
    grouped = defaultdict(list)
    for job in jobs:
        key = (
            tuple(job["obj_pos"]),
            tuple(job["qpos"]),
            tuple(job["qvel"]),
        )
        grouped[key].append(job)
    records = []
    for group_jobs in grouped.values():
        records.extend(_scan_group(scene, arm, obj, base_state, group_jobs, config))
    order = {job["anchor_id"]: index for index, job in enumerate(jobs)}
    records.sort(key=lambda record: order[record["anchor_id"]])
    return records


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return value


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


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
    records = []
    all_rows = []
    start = time.perf_counter()
    if config.get("collect_probes", True):
        for job in jobs:
            record = _collect_job(scene, arm, obj, base_state, job, config)
            all_rows.extend(record.pop("rows"))
            records.append(record)
    else:
        records = _collect_scan_jobs(scene, arm, obj, base_state, jobs, config)
    for index, (job, record) in enumerate(zip(jobs, records), start=1):
        print(
            f"[a5-vjp-v2-worker] {index}/{len(jobs)} anchor={job['anchor_id']} "
            f"status={record['status']} "
            f"rows={record['num_kept_rows']}/{record['num_rows']} "
            f"contacts={record['anchor_contact']['count']} elapsed={record['elapsed_seconds']:.2f}s",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "rows.csv"
    anchors_path = args.out_dir / "anchors.csv"
    json_path = args.out_dir / "worker.json"
    _write_csv(rows_path, all_rows)
    _write_csv(anchors_path, records)
    payload = {
        "description": "A5 action-side VJP v2 long-lived worker",
        "request": str(args.request),
        "config": config,
        "num_jobs": len(jobs),
        "num_rows": len(all_rows),
        "num_kept_rows": sum(bool(row["keep"]) for row in all_rows),
        "elapsed_seconds": time.perf_counter() - start,
        "rows_csv": str(rows_path),
        "anchors_csv": str(anchors_path),
        "anchors": records,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[a5-vjp-v2-worker] wrote {json_path}")
    print(f"[a5-vjp-v2-worker] wrote {rows_path}")


if __name__ == "__main__":
    raise SystemExit(main())
