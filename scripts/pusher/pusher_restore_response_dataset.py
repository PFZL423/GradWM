"""Restored-anchor FD response labels for the Genesis-native pusher arm.

This is the simple articulated-pusher counterpart to
``scripts/arx/a5_restore_response_dataset.py``. It keeps the CSV schema close to
the A5 dataset so the same matrix-head trainer can be reused with
``--robot-dof 9``.
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import genesis as gs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pusher.pusher_like_arm_sanity import (  # noqa: E402
    ARM_LPOSE_QPOS,
    INITIAL_FINGER_OPEN,
    OBJECT_POS,
    TARGET_POS,
    TOTAL_DOFS,
    _candidate_push_qvels,
    _close_qvel,
    _contact_count,
    _make_scene,
    _set_initial_arm_pose,
    _zeros,
)


DEFAULT_JSON = Path("analysis/2026-07-09_arx_pusher/simple_pusher_restore_response_dataset.json")
DEFAULT_CSV = Path("analysis/2026-07-09_arx_pusher/simple_pusher_restore_response_dataset.csv")


def _norm(values):
    return math.sqrt(sum(float(x) * float(x) for x in values))


def _finite(values):
    return all(math.isfinite(float(x)) for x in values)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _sub(a, b):
    return [float(a[i]) - float(b[i]) for i in range(len(a))]


def _scale(a, value):
    return [float(x) * value for x in a]


def _normalize(vec):
    n = _norm(vec)
    if n < 1e-12:
        return list(vec)
    return [float(x) / n for x in vec]


def _jsonable_vec(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    return [float(x) for x in values]


def _object_pos(obj):
    return _jsonable_vec(obj.get_state().pos.reshape(-1, 3)[0])


def _object_qvel(obj):
    return _jsonable_vec(obj.get_dofs_velocity())


def _object_state(obj):
    return _object_pos(obj) + _object_qvel(obj)


def _arm_qpos_list():
    qpos = [0.0 for _ in range(TOTAL_DOFS)]
    for i, value in enumerate(ARM_LPOSE_QPOS):
        qpos[i] = float(value)
    qpos[7] = float(INITIAL_FINGER_OPEN)
    qpos[8] = float(INITIAL_FINGER_OPEN)
    return qpos


def _set_arm_qvel(arm, values):
    arm.set_dofs_velocity(gs.tensor(values))


def _program_qvel(program, speed):
    choices = dict(_candidate_push_qvels(speed))
    if program not in choices:
        raise ValueError(f"unknown program={program}; options={sorted(choices)}")
    return _jsonable_vec(choices[program])


def _directions(qvel, num_random, seed):
    dirs = []
    for i in range(len(qvel)):
        vec = [0.0 for _ in qvel]
        vec[i] = 1.0
        dirs.append({"name": f"dof{i + 1}+", "vec": vec})

    nominal = _normalize(qvel)
    if any(abs(x) > 1e-12 for x in nominal):
        dirs.append({"name": "nominal_qvel", "vec": nominal})

    rng = random.Random(seed)
    for i in range(num_random):
        vec = [rng.gauss(0.0, 1.0) for _ in qvel]
        dirs.append({"name": f"random{i + 1:02d}", "vec": _normalize(vec)})
    return dirs


def _parse_scales(text):
    return [float(x) for x in text.split(",") if x.strip()]


def _contact_stats(values):
    return {
        "mean": float(_mean(values)),
        "max": int(max(values)) if values else 0,
        "min": int(min(values)) if values else 0,
        "trace": [int(x) for x in values],
    }


def _trace_vec(trace, idx, length):
    if not trace:
        return [0.0 for _ in range(length)]
    idx = max(0, min(idx, len(trace) - 1))
    values = list(trace[idx])
    if len(values) < length:
        values = values + [0.0] * (length - len(values))
    return values[:length]


def _contact_window_stats(contacts, center, radius):
    lo = max(0, center - radius)
    hi = min(len(contacts), center + radius + 1)
    return _contact_stats(contacts[lo:hi])


def _choose_anchor_step(trace, threshold):
    if len(trace) < 2:
        return 0
    prev = trace[0]
    best_step = 0
    best_delta = 0.0
    for i in range(1, len(trace)):
        delta = _norm(_sub(trace[i], prev))
        if delta > best_delta:
            best_delta = delta
            best_step = i
        if delta >= threshold:
            return i
        prev = trace[i]
    return best_step


def _rollout_nominal(args, scene, arm, obj, qvel):
    scene.reset()
    _set_initial_arm_pose(arm)

    contacts = []
    zero = _jsonable_vec(_zeros())
    close = _jsonable_vec(_close_qvel(args.close_speed))

    for _ in range(args.settle_steps):
        _set_arm_qvel(arm, zero)
        scene.step()
        contacts.append(_contact_count(scene))

    for _ in range(args.close_steps):
        _set_arm_qvel(arm, close)
        scene.step()
        contacts.append(_contact_count(scene))

    initial_pos = _object_pos(obj)
    initial_qvel = _object_qvel(obj)
    trace = []
    qvel_trace = []
    for _ in range(args.push_steps):
        _set_arm_qvel(arm, qvel)
        scene.step()
        contacts.append(_contact_count(scene))
        trace.append(_object_pos(obj))
        qvel_trace.append(_object_qvel(obj))

    return {
        "initial_pos": initial_pos,
        "initial_qvel": initial_qvel,
        "final_pos": _object_pos(obj),
        "final_qvel": _object_qvel(obj),
        "trace": trace,
        "qvel_trace": qvel_trace,
        "contacts": contacts,
    }


def _continue_query(scene, arm, obj, qvel, first_action, local_steps):
    contacts = []
    local_pos = None
    local_qvel = None
    local_state = None
    for step in range(local_steps):
        _set_arm_qvel(arm, first_action if step == 0 else qvel)
        scene.step()
        contacts.append(_contact_count(scene))
        if step == local_steps - 1:
            local_pos = _object_pos(obj)
            local_qvel = _object_qvel(obj)
            local_state = local_pos + local_qvel

    return {
        "local_pos": local_pos,
        "local_qvel": local_qvel,
        "local_state": local_state,
        "final_pos": local_pos,
        "final_qvel": local_qvel,
        "final_state": local_state,
        "contacts": contacts,
    }


def _query_from_anchor(scene, arm, obj, anchor_state, qvel, first_action, local_steps):
    scene.reset(state=anchor_state)
    return _continue_query(scene, arm, obj, qvel, first_action, local_steps)


def _capture_anchor(args, scene, arm, obj, qvel, anchor_step, local_steps):
    scene.reset()
    _set_initial_arm_pose(arm)

    zero = _jsonable_vec(_zeros())
    close = _jsonable_vec(_close_qvel(args.close_speed))
    for _ in range(args.settle_steps):
        _set_arm_qvel(arm, zero)
        scene.step()
    for _ in range(args.close_steps):
        _set_arm_qvel(arm, close)
        scene.step()
    for _ in range(anchor_step):
        _set_arm_qvel(arm, qvel)
        scene.step()

    anchor_state = scene.get_state()
    anchor_snapshot = {
        "state": _object_state(obj),
        "contact_count": _contact_count(scene),
    }
    online_nominal = _continue_query(scene, arm, obj, qvel, qvel, local_steps)

    scene.reset(state=anchor_state)
    restored_anchor_snapshot = {
        "state": _object_state(obj),
        "contact_count": _contact_count(scene),
    }
    restored_nominal = _continue_query(scene, arm, obj, qvel, qvel, local_steps)
    return {
        "anchor_state": anchor_state,
        "anchor_snapshot": anchor_snapshot,
        "restored_anchor_snapshot": restored_anchor_snapshot,
        "online_nominal": online_nominal,
        "restored_nominal": restored_nominal,
    }


def _max_abs(a, b):
    if len(a) != len(b):
        return float("inf")
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def _nominal_context(args, qvel, nominal, anchor_step):
    contact_center = args.settle_steps + args.close_steps + anchor_step
    anchor_pos = _trace_vec(nominal["trace"], anchor_step, 3)
    pre_pos = _trace_vec(nominal["trace"], anchor_step - args.response_steps, 3)
    post_pos = _trace_vec(nominal["trace"], anchor_step + args.response_steps, 3)
    anchor_qvel = _trace_vec(nominal["qvel_trace"], anchor_step, 6)
    pre_qvel = _trace_vec(nominal["qvel_trace"], anchor_step - args.response_steps, 6)
    post_qvel = _trace_vec(nominal["qvel_trace"], anchor_step + args.response_steps, 6)
    contact = _contact_window_stats(nominal["contacts"], contact_center, args.contact_window)
    return {
        "nominal_anchor_pos": anchor_pos,
        "nominal_pre_pos": pre_pos,
        "nominal_post_pos": post_pos,
        "nominal_initial_pos": nominal["initial_pos"],
        "nominal_final_pos": nominal["final_pos"],
        "nominal_anchor_qvel": anchor_qvel,
        "nominal_pre_qvel": pre_qvel,
        "nominal_post_qvel": post_qvel,
        "nominal_initial_qvel": nominal["initial_qvel"],
        "nominal_final_qvel": nominal["final_qvel"],
        "nominal_pre_disp": _sub(anchor_pos, pre_pos),
        "nominal_post_disp": _sub(post_pos, anchor_pos),
        "nominal_total_disp": _sub(nominal["final_pos"], nominal["initial_pos"]),
        "nominal_pre_qvel_delta": _sub(anchor_qvel, pre_qvel),
        "nominal_post_qvel_delta": _sub(post_qvel, anchor_qvel),
        "nominal_total_qvel_delta": _sub(nominal["final_qvel"], nominal["initial_qvel"]),
        "nominal_contact_mean": contact["mean"],
        "nominal_contact_max": contact["max"],
        "nominal_contact_min": contact["min"],
        "nominal_contact_trace": contact["trace"],
    }


def _row_filter(args, plus_stats, minus_stats, local_state_response):
    contact_mean_delta = abs(plus_stats["mean"] - minus_stats["mean"])
    contact_max_delta = abs(plus_stats["max"] - minus_stats["max"])
    state_norm = _norm(local_state_response)
    reasons = []
    if contact_mean_delta > args.max_contact_mean_delta:
        reasons.append("contact_mean_delta")
    if contact_max_delta > args.max_contact_max_delta:
        reasons.append("contact_max_delta")
    if state_norm < args.min_response_norm:
        reasons.append("response_too_small")
    if state_norm > args.max_response_norm:
        reasons.append("response_too_large")
    if not _finite(local_state_response):
        reasons.append("nonfinite_response")
    return {
        "keep": not reasons,
        "filter_reason": "ok" if not reasons else "|".join(reasons),
        "contact_mean_delta": float(contact_mean_delta),
        "contact_max_delta": float(contact_max_delta),
    }


def _anchor_filter(args, metrics):
    reasons = []
    if metrics["restored_nominal_local_state_max_abs"] > args.max_restore_local_state_diff:
        reasons.append("restore_local_state_diff")
    if metrics["repeat_local_state_max_abs"] > args.max_repeat_local_state_diff:
        reasons.append("repeat_local_state_diff")
    if metrics["horizontal_disp"] < args.min_horizontal_disp:
        reasons.append("horizontal_disp_too_small")
    if abs(metrics["vertical_disp"]) > args.max_abs_vertical_disp:
        reasons.append("vertical_disp_too_large")
    return {
        "anchor_keep": not reasons,
        "anchor_filter_reason": "ok" if not reasons else "|".join(reasons),
    }


def _build_row(args, direction, eps, plus_rec, minus_rec, nominal_context):
    local_response = _scale(_sub(plus_rec["local_pos"], minus_rec["local_pos"]), 1.0 / (2.0 * eps))
    local_vel_response = _scale(_sub(plus_rec["local_qvel"], minus_rec["local_qvel"]), 1.0 / (2.0 * eps))
    local_state_response = local_response + local_vel_response
    plus_stats = _contact_stats(plus_rec["contacts"])
    minus_stats = _contact_stats(minus_rec["contacts"])
    filter_status = _row_filter(args, plus_stats, minus_stats, local_state_response)
    return {
        "direction": direction["name"],
        "direction_vec": direction["vec"],
        "epsilon": eps,
        "local_response": local_response,
        "final_response": local_response,
        "local_vel_response": local_vel_response,
        "final_vel_response": local_vel_response,
        "local_state_response": local_state_response,
        "final_state_response": local_state_response,
        "local_response_norm": _norm(local_response),
        "final_response_norm": _norm(local_response),
        "local_vel_response_norm": _norm(local_vel_response),
        "final_vel_response_norm": _norm(local_vel_response),
        "local_state_response_norm": _norm(local_state_response),
        "final_state_response_norm": _norm(local_state_response),
        "local_qvel_plus": plus_rec["local_qvel"],
        "local_qvel_minus": minus_rec["local_qvel"],
        "final_qvel_plus": plus_rec["final_qvel"],
        "final_qvel_minus": minus_rec["final_qvel"],
        "plus_contact_mean": plus_stats["mean"],
        "minus_contact_mean": minus_stats["mean"],
        "plus_contact_max": plus_stats["max"],
        "minus_contact_max": minus_stats["max"],
        "plus_contact_trace": plus_stats["trace"],
        "minus_contact_trace": minus_stats["trace"],
        "max_total_contact_count_plus": plus_stats["max"],
        "max_total_contact_count_minus": minus_stats["max"],
        **nominal_context,
        **filter_status,
    }


def _repeat_check(scene, arm, obj, anchor_state, qvel, direction, eps, local_steps):
    first_action = [qvel[i] + eps * direction["vec"][i] for i in range(len(qvel))]
    a = _query_from_anchor(scene, arm, obj, anchor_state, qvel, first_action, local_steps)
    b = _query_from_anchor(scene, arm, obj, anchor_state, qvel, first_action, local_steps)
    return {
        "direction": direction["name"],
        "epsilon": eps,
        "local_state_max_abs": _max_abs(a["local_state"], b["local_state"]),
        "contact_trace_a": a["contacts"],
        "contact_trace_b": b["contacts"],
    }


def _collect_rows(args, scene, arm, obj, anchor_state, qvel, directions, scales, local_steps, nominal_context):
    rows = []
    start = time.perf_counter()
    for direction in directions:
        vec = direction["vec"]
        for eps in scales:
            plus_action = [qvel[i] + eps * vec[i] for i in range(len(qvel))]
            minus_action = [qvel[i] - eps * vec[i] for i in range(len(qvel))]
            plus_rec = _query_from_anchor(scene, arm, obj, anchor_state, qvel, plus_action, local_steps)
            minus_rec = _query_from_anchor(scene, arm, obj, anchor_state, qvel, minus_action, local_steps)
            row = _build_row(args, direction, eps, plus_rec, minus_rec, nominal_context)
            if row["keep"] or not args.drop_filtered:
                rows.append(row)
    return rows, time.perf_counter() - start


def _write_csv(path, rows, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "anchor_id",
        "row_idx",
        "obj_x",
        "obj_y",
        "obj_z",
        "speed",
        "program",
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
    vector_fields = {
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
    }
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row_idx, row in enumerate(rows):
            flat = {**meta, **row, "row_idx": row_idx}
            for key in vector_fields:
                flat[key] = json.dumps(flat[key])
            writer.writerow(flat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", default="j6-")
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--close-speed", type=float, default=1.5)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--push-steps", type=int, default=80)
    parser.add_argument("--response-steps", type=int, default=5)
    parser.add_argument("--anchor-step", type=int, default=-1)
    parser.add_argument("--motion-threshold", type=float, default=2e-5)
    parser.add_argument("--contact-window", type=int, default=5)
    parser.add_argument("--max-contact-mean-delta", type=float, default=8.0)
    parser.add_argument("--max-contact-max-delta", type=float, default=12.0)
    parser.add_argument("--min-response-norm", type=float, default=1e-10)
    parser.add_argument("--max-response-norm", type=float, default=20.0)
    parser.add_argument("--drop-filtered", action="store_true")
    parser.add_argument("--drop-bad-anchor", action="store_true")
    parser.add_argument("--max-restore-local-state-diff", type=float, default=float("inf"))
    parser.add_argument("--max-repeat-local-state-diff", type=float, default=1e-8)
    parser.add_argument("--min-horizontal-disp", type=float, default=0.005)
    parser.add_argument("--max-abs-vertical-disp", type=float, default=0.005)
    parser.add_argument("--scales", default="1e-3")
    parser.add_argument("--num-random", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    scene, arm, obj = _make_scene()
    if arm.n_dofs != TOTAL_DOFS:
        raise RuntimeError(f"expected arm n_dofs={TOTAL_DOFS}, got {arm.n_dofs}")

    qvel = _program_qvel(args.program, args.speed)
    nominal_start = time.perf_counter()
    nominal = _rollout_nominal(args, scene, arm, obj, qvel)
    anchor_step = args.anchor_step
    if anchor_step < 0:
        anchor_step = _choose_anchor_step(nominal["trace"], args.motion_threshold)
    local_steps = max(1, args.response_steps)
    anchor = _capture_anchor(args, scene, arm, obj, qvel, anchor_step, local_steps)
    nominal_seconds = time.perf_counter() - nominal_start

    directions = _directions(qvel, args.num_random, args.seed)
    scales = _parse_scales(args.scales)
    nominal_context = _nominal_context(args, qvel, nominal, anchor_step)
    repeat_check = _repeat_check(scene, arm, obj, anchor["anchor_state"], qvel, directions[0], scales[0], local_steps)

    displacement = _sub(nominal["final_pos"], nominal["initial_pos"])
    horizontal_disp = math.sqrt(displacement[0] ** 2 + displacement[1] ** 2)
    vertical_disp = displacement[2]
    anchor_metrics = {
        "horizontal_disp": horizontal_disp,
        "vertical_disp": vertical_disp,
        "restored_anchor_state_max_abs": _max_abs(
            anchor["anchor_snapshot"]["state"], anchor["restored_anchor_snapshot"]["state"]
        ),
        "restored_nominal_local_state_max_abs": _max_abs(
            anchor["online_nominal"]["local_state"], anchor["restored_nominal"]["local_state"]
        ),
        "repeat_local_state_max_abs": repeat_check["local_state_max_abs"],
    }
    anchor_status = _anchor_filter(args, anchor_metrics)
    if anchor_status["anchor_keep"] or not args.drop_bad_anchor:
        rows, query_seconds = _collect_rows(
            args,
            scene,
            arm,
            obj,
            anchor["anchor_state"],
            qvel,
            directions,
            scales,
            local_steps,
            nominal_context,
        )
    else:
        rows = []
        query_seconds = 0.0

    meta = {
        "anchor_id": 1,
        "obj_x": OBJECT_POS[0],
        "obj_y": OBJECT_POS[1],
        "obj_z": OBJECT_POS[2],
        "speed": args.speed,
        "program": args.program,
        "qpos": _arm_qpos_list(),
        "qvel": qvel,
        "anchor_step": anchor_step,
        "response_steps": args.response_steps,
        "horizontal_disp": horizontal_disp,
        "vertical_disp": vertical_disp,
        **anchor_status,
        "restored_nominal_local_state_max_abs": anchor_metrics["restored_nominal_local_state_max_abs"],
        "repeat_local_state_max_abs": anchor_metrics["repeat_local_state_max_abs"],
    }

    num_possible_rows = len(directions) * len(scales)
    payload = {
        "description": "Genesis-native pusher restored-anchor FD response dataset",
        "sampler": "same_process_restore",
        "program": args.program,
        "speed": args.speed,
        "qpos": meta["qpos"],
        "qvel": qvel,
        "object_pos": list(OBJECT_POS),
        "target_pos": list(TARGET_POS),
        "settle_steps": args.settle_steps,
        "close_steps": args.close_steps,
        "push_steps": args.push_steps,
        "response_steps": args.response_steps,
        "query_steps": local_steps,
        "anchor_step": anchor_step,
        "scales": scales,
        "num_random": args.num_random,
        "seed": args.seed,
        "drop_filtered": args.drop_filtered,
        "drop_bad_anchor": args.drop_bad_anchor,
        **anchor_status,
        "initial_pos": nominal["initial_pos"],
        "final_pos": nominal["final_pos"],
        "displacement": displacement,
        "horizontal_disp": horizontal_disp,
        "vertical_disp": vertical_disp,
        "num_rows": len(rows),
        "num_kept_rows": sum(1 for row in rows if row["keep"]),
        "num_possible_rows": num_possible_rows,
        "nominal_seconds": nominal_seconds,
        "query_seconds": query_seconds,
        "rows_per_second": len(rows) / query_seconds if query_seconds > 0 else 0.0,
        "anchor_contact_count": anchor["anchor_snapshot"]["contact_count"],
        "restored_anchor_contact_count": anchor["restored_anchor_snapshot"]["contact_count"],
        **anchor_metrics,
        "online_nominal_contact_trace": anchor["online_nominal"]["contacts"],
        "restored_nominal_contact_trace": anchor["restored_nominal"]["contacts"],
        "repeat_check": repeat_check,
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(args.csv, rows, meta)
    print(
        f"[pusher-restore] program={args.program} anchor_step={anchor_step} "
        f"rows={len(rows)}/{num_possible_rows} keep={anchor_status['anchor_keep']} "
        f"hdisp={horizontal_disp:.5f} vdisp={vertical_disp:.5f} "
        f"query_s={query_seconds:.2f}"
    )
    print(
        "[pusher-restore] restore_diff="
        f"{anchor_metrics['restored_nominal_local_state_max_abs']:.3e} "
        f"repeat_diff={repeat_check['local_state_max_abs']:.3e}"
    )
    print(f"[pusher-restore] wrote {args.json} and {args.csv}")


if __name__ == "__main__":
    raise SystemExit(main())
