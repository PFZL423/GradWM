"""Collect A5 local response labels with same-process state restore.

This is the fast path for action-side VJP data:

    nominal rollout -> save contact anchor SimState -> reset(state=anchor)
    -> query a_t +/- eps v_k

It keeps the row schema close to ``a5_fd_response_dataset.py`` so the existing
multi-anchor aggregator and matrix-head trainer can consume the output.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import genesis as gs

from a5_fd_response_dataset import (
    DEFAULT_CSV,
    DEFAULT_JSON,
    DEFAULT_SCALES,
    _choose_anchor_step,
    _directions,
    _flat_qvel,
    _nominal_context,
    _parse_scales,
    _parse_vec,
    _rollout,
    _scale,
    _sub,
    _tensor_vec,
    _vec3,
    _write_csv,
)
from a5_pusher_forward_sanity import (
    DEFAULT_OBJ_POS,
    DEFAULT_PUSH_STEPS,
    DEFAULT_QPOS,
    DEFAULT_QVEL,
    _contact_count,
    _flat_pos,
    _make_scene,
)


DEFAULT_RESTORE_JSON = Path("analysis/2026-07-09_arx_pusher/a5_restore_response_dataset.json")
DEFAULT_RESTORE_CSV = Path("analysis/2026-07-09_arx_pusher/a5_restore_response_dataset.csv")


def _norm(values):
    return math.sqrt(sum(float(x) * float(x) for x in values))


def _finite(values):
    return all(math.isfinite(float(x)) for x in values)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _contact_stats(values):
    return {
        "mean": float(_mean(values)),
        "max": int(max(values)) if values else 0,
        "min": int(min(values)) if values else 0,
        "trace": [int(x) for x in values],
    }


def _set_arm_qvel(arm, values):
    arm.set_dofs_velocity(gs.tensor(values))


def _object_state(obj):
    return _vec3(_flat_pos(obj)) + _tensor_vec(_flat_qvel(obj))


def _max_abs(a, b):
    if len(a) != len(b):
        return float("inf")
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def _continue_query(args, scene, arm, obj, first_action, local_steps, total_steps):
    contacts = []
    nominal = list(args.qvel)
    local_pos = None
    local_qvel = None
    local_state = None
    for step in range(total_steps):
        _set_arm_qvel(arm, first_action if step == 0 else nominal)
        scene.step()
        contacts.append(_contact_count(scene))
        if step == local_steps - 1:
            local_pos = _flat_pos(obj)
            local_qvel = _flat_qvel(obj)
            local_state = _object_state(obj)

    final_pos = _flat_pos(obj)
    final_qvel = _flat_qvel(obj)
    final_state = _object_state(obj)
    if local_pos is None:
        local_pos = final_pos
        local_qvel = final_qvel
        local_state = final_state
    return {
        "local_pos": local_pos,
        "local_qvel": local_qvel,
        "local_state": local_state,
        "final_pos": final_pos,
        "final_qvel": final_qvel,
        "final_state": final_state,
        "contacts": contacts,
    }


def _query_from_anchor(args, scene, arm, obj, anchor_state, first_action, local_steps, total_steps):
    scene.reset(state=anchor_state)
    return _continue_query(args, scene, arm, obj, first_action, local_steps, total_steps)


def _capture_anchor(args, scene, arm, obj, anchor_step, local_steps, total_steps):
    scene.reset()
    arm.set_dofs_position(torch.tensor(args.qpos, dtype=torch.float32))

    zero = [0.0 for _ in args.qvel]
    settle_contacts = []
    for _ in range(args.settle_steps):
        _set_arm_qvel(arm, zero)
        scene.step()
        settle_contacts.append(_contact_count(scene))

    for _ in range(anchor_step):
        _set_arm_qvel(arm, args.qvel)
        scene.step()

    anchor_state = scene.get_state()
    anchor_snapshot = {
        "state": _object_state(obj),
        "contact_count": _contact_count(scene),
    }

    online_nominal = _continue_query(args, scene, arm, obj, list(args.qvel), local_steps, total_steps)
    scene.reset(state=anchor_state)
    restored_anchor_snapshot = {
        "state": _object_state(obj),
        "contact_count": _contact_count(scene),
    }
    restored_nominal = _continue_query(args, scene, arm, obj, list(args.qvel), local_steps, total_steps)

    return {
        "anchor_state": anchor_state,
        "anchor_snapshot": anchor_snapshot,
        "restored_anchor_snapshot": restored_anchor_snapshot,
        "online_nominal": online_nominal,
        "restored_nominal": restored_nominal,
        "settle_contacts": settle_contacts,
    }


def _response(a, b, eps):
    return _scale(_sub(a, b), 1.0 / (2.0 * eps))


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
    local_plus = _vec3(plus_rec["local_pos"])
    local_minus = _vec3(minus_rec["local_pos"])
    final_plus = _vec3(plus_rec["final_pos"])
    final_minus = _vec3(minus_rec["final_pos"])
    local_qvel_plus = _tensor_vec(plus_rec["local_qvel"])
    local_qvel_minus = _tensor_vec(minus_rec["local_qvel"])
    final_qvel_plus = _tensor_vec(plus_rec["final_qvel"])
    final_qvel_minus = _tensor_vec(minus_rec["final_qvel"])

    local_response = _response(local_plus, local_minus, eps)
    final_response = _response(final_plus, final_minus, eps)
    local_vel_response = _response(local_qvel_plus, local_qvel_minus, eps)
    final_vel_response = _response(final_qvel_plus, final_qvel_minus, eps)
    local_state_response = local_response + local_vel_response
    final_state_response = final_response + final_vel_response

    plus_stats = _contact_stats(plus_rec["contacts"])
    minus_stats = _contact_stats(minus_rec["contacts"])
    filter_status = _row_filter(args, plus_stats, minus_stats, local_state_response)

    return {
        "direction": direction["name"],
        "direction_vec": direction["vec"],
        "epsilon": eps,
        "local_plus": local_plus,
        "local_minus": local_minus,
        "final_plus": final_plus,
        "final_minus": final_minus,
        "local_qvel_plus": local_qvel_plus,
        "local_qvel_minus": local_qvel_minus,
        "final_qvel_plus": final_qvel_plus,
        "final_qvel_minus": final_qvel_minus,
        "local_response": local_response,
        "final_response": final_response,
        "local_vel_response": local_vel_response,
        "final_vel_response": final_vel_response,
        "local_state_response": local_state_response,
        "final_state_response": final_state_response,
        "local_response_norm": _norm(local_response),
        "final_response_norm": _norm(final_response),
        "local_vel_response_norm": _norm(local_vel_response),
        "final_vel_response_norm": _norm(final_vel_response),
        "local_state_response_norm": _norm(local_state_response),
        "final_state_response_norm": _norm(final_state_response),
        "plus_contact_mean": plus_stats["mean"],
        "minus_contact_mean": minus_stats["mean"],
        "plus_contact_max": plus_stats["max"],
        "minus_contact_max": minus_stats["max"],
        "plus_contact_trace": plus_stats["trace"],
        "minus_contact_trace": minus_stats["trace"],
        "max_total_contact_count_plus": max(plus_rec["contacts"]) if plus_rec["contacts"] else 0,
        "max_total_contact_count_minus": max(minus_rec["contacts"]) if minus_rec["contacts"] else 0,
        **nominal_context,
        **filter_status,
    }


def _repeat_check(args, scene, arm, obj, anchor_state, direction, eps, local_steps, total_steps):
    first_action = [args.qvel[i] + eps * direction["vec"][i] for i in range(len(args.qvel))]
    a = _query_from_anchor(args, scene, arm, obj, anchor_state, first_action, local_steps, total_steps)
    b = _query_from_anchor(args, scene, arm, obj, anchor_state, first_action, local_steps, total_steps)
    return {
        "direction": direction["name"],
        "epsilon": eps,
        "local_state_max_abs": _max_abs(a["local_state"], b["local_state"]),
        "final_state_max_abs": _max_abs(a["final_state"], b["final_state"]),
        "contact_trace_a": a["contacts"],
        "contact_trace_b": b["contacts"],
    }


def _collect_rows(args, scene, arm, obj, anchor_state, directions, scales, local_steps, total_steps, nominal_context):
    rows = []
    query_start = time.perf_counter()
    for direction in directions:
        vec = direction["vec"]
        for eps in scales:
            plus_action = [args.qvel[i] + eps * vec[i] for i in range(len(args.qvel))]
            minus_action = [args.qvel[i] - eps * vec[i] for i in range(len(args.qvel))]
            plus_rec = _query_from_anchor(args, scene, arm, obj, anchor_state, plus_action, local_steps, total_steps)
            minus_rec = _query_from_anchor(args, scene, arm, obj, anchor_state, minus_action, local_steps, total_steps)
            row = _build_row(args, direction, eps, plus_rec, minus_rec, nominal_context)
            if row["keep"] or not args.drop_filtered:
                rows.append(row)
    query_seconds = time.perf_counter() - query_start
    return rows, query_seconds


def _write_manifest_csv(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sampler",
        "requires_grad",
        "anchor_step",
        "query_steps",
        "num_rows",
        "num_kept_rows",
        "drop_bad_anchor",
        "anchor_keep",
        "anchor_filter_reason",
        "nominal_seconds",
        "query_seconds",
        "rows_per_second",
        "estimated_step_reuse_speedup",
        "restored_anchor_state_max_abs",
        "restored_nominal_local_state_max_abs",
        "repeat_local_state_max_abs",
    ]
    row = {key: payload.get(key) for key in fields}
    row["repeat_local_state_max_abs"] = payload["repeat_check"]["local_state_max_abs"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-x", type=float, default=DEFAULT_OBJ_POS[0])
    parser.add_argument("--obj-y", type=float, default=DEFAULT_OBJ_POS[1])
    parser.add_argument("--obj-z", type=float, default=DEFAULT_OBJ_POS[2])
    parser.add_argument("--qpos", type=lambda x: _parse_vec(x, 6, "qpos"), default=DEFAULT_QPOS)
    parser.add_argument("--qvel", type=lambda x: _parse_vec(x, 6, "qvel"), default=DEFAULT_QVEL)
    parser.add_argument("--requires-grad", dest="requires_grad", action="store_true")
    parser.add_argument("--no-requires-grad", dest="requires_grad", action="store_false")
    parser.set_defaults(requires_grad=True)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--push-steps", type=int, default=DEFAULT_PUSH_STEPS)
    parser.add_argument("--response-steps", type=int, default=10)
    parser.add_argument("--anchor-step", type=int, default=-1)
    parser.add_argument("--motion-threshold", type=float, default=2e-5)
    parser.add_argument("--contact-window", type=int, default=5)
    parser.add_argument("--max-contact-mean-delta", type=float, default=8.0)
    parser.add_argument("--max-contact-max-delta", type=float, default=12.0)
    parser.add_argument("--min-response-norm", type=float, default=1e-8)
    parser.add_argument("--max-response-norm", type=float, default=1e3)
    parser.add_argument("--drop-filtered", action="store_true")
    parser.add_argument("--drop-bad-anchor", action="store_true")
    parser.add_argument("--max-restore-local-state-diff", type=float, default=float("inf"))
    parser.add_argument("--max-repeat-local-state-diff", type=float, default=1e-8)
    parser.add_argument("--min-horizontal-disp", type=float, default=0.0)
    parser.add_argument("--max-abs-vertical-disp", type=float, default=float("inf"))
    parser.add_argument("--collect-final", action="store_true")
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--num-random", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path, default=DEFAULT_RESTORE_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_RESTORE_CSV)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    obj_pos = (args.obj_x, args.obj_y, args.obj_z)
    scene, arm, obj = _make_scene(obj_pos, requires_grad=args.requires_grad)

    nominal_start = time.perf_counter()
    nominal = _rollout(args, scene, arm, obj)
    anchor_step = args.anchor_step
    if anchor_step < 0:
        anchor_step = _choose_anchor_step(nominal["trace"], args.motion_threshold)
    local_steps = args.response_steps + 1
    remaining_steps = max(local_steps, args.push_steps - anchor_step)
    total_steps = remaining_steps if args.collect_final else local_steps
    anchor = _capture_anchor(args, scene, arm, obj, anchor_step, local_steps, total_steps)
    nominal_seconds = time.perf_counter() - nominal_start

    directions = _directions(args.qvel, args.num_random, args.seed)
    scales = _parse_scales(args.scales)
    nominal_context = _nominal_context(args, nominal, anchor_step)
    repeat_check = _repeat_check(
        args,
        scene,
        arm,
        obj,
        anchor["anchor_state"],
        directions[0],
        scales[0],
        local_steps,
        total_steps,
    )

    initial_pos = _vec3(nominal["initial_pos"])
    final_pos = _vec3(nominal["final_pos"])
    displacement = _sub(final_pos, initial_pos)
    horizontal_disp = math.sqrt(displacement[0] ** 2 + displacement[1] ** 2)
    vertical_disp = displacement[2]
    anchor_metrics = {
        "horizontal_disp": horizontal_disp,
        "vertical_disp": vertical_disp,
        "restored_anchor_state_max_abs": _max_abs(
            anchor["anchor_snapshot"]["state"],
            anchor["restored_anchor_snapshot"]["state"],
        ),
        "restored_nominal_local_state_max_abs": _max_abs(
            anchor["online_nominal"]["local_state"],
            anchor["restored_nominal"]["local_state"],
        ),
        "restored_nominal_final_state_max_abs": _max_abs(
            anchor["online_nominal"]["final_state"],
            anchor["restored_nominal"]["final_state"],
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
            directions,
            scales,
            local_steps,
            total_steps,
            nominal_context,
        )
    else:
        rows = []
        query_seconds = 0.0

    num_possible_rows = len(directions) * len(scales)
    query_rollouts = 2 * num_possible_rows
    restored_steps = args.settle_steps + anchor_step + query_rollouts * total_steps
    slow_steps = query_rollouts * (args.settle_steps + args.push_steps)
    estimated_step_reuse_speedup = slow_steps / max(restored_steps, 1)
    rows_per_second = len(rows) / query_seconds if query_seconds > 0 else 0.0

    payload = {
        "description": "A5 restored-anchor finite-difference response dataset",
        "sampler": "same_process_restore",
        "requires_grad": args.requires_grad,
        "qpos": args.qpos,
        "qvel": args.qvel,
        "object_pos": obj_pos,
        "settle_steps": args.settle_steps,
        "push_steps": args.push_steps,
        "response_steps": args.response_steps,
        "query_steps": local_steps,
        "collect_final": args.collect_final,
        "total_query_steps": total_steps,
        "anchor_step": anchor_step,
        "scales": scales,
        "num_random": args.num_random,
        "seed": args.seed,
        "contact_window": args.contact_window,
        "max_contact_mean_delta": args.max_contact_mean_delta,
        "max_contact_max_delta": args.max_contact_max_delta,
        "drop_filtered": args.drop_filtered,
        "drop_bad_anchor": args.drop_bad_anchor,
        "max_restore_local_state_diff": args.max_restore_local_state_diff,
        "max_repeat_local_state_diff": args.max_repeat_local_state_diff,
        "min_horizontal_disp": args.min_horizontal_disp,
        "max_abs_vertical_disp": args.max_abs_vertical_disp,
        **anchor_status,
        "initial_pos": initial_pos,
        "final_pos": final_pos,
        "displacement": displacement,
        "horizontal_disp": horizontal_disp,
        "vertical_disp": vertical_disp,
        "num_rows": len(rows),
        "num_kept_rows": sum(1 for row in rows if row["keep"]),
        "num_possible_rows": num_possible_rows,
        "nominal_seconds": nominal_seconds,
        "query_seconds": query_seconds,
        "rows_per_second": rows_per_second,
        "query_rollouts": query_rollouts,
        "estimated_slow_steps": slow_steps,
        "estimated_restored_steps": restored_steps,
        "estimated_step_reuse_speedup": estimated_step_reuse_speedup,
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
    _write_csv(args.csv, rows)
    if args.manifest_csv is not None:
        _write_manifest_csv(args.manifest_csv, payload)
    print(
        f"[a5-restore-dataset] anchor_step={anchor_step} rows={len(rows)}/{num_possible_rows} "
        f"anchor_keep={anchor_status['anchor_keep']} "
        f"query_steps={local_steps} query_s={query_seconds:.2f} "
        f"rows/s={rows_per_second:.3f} est_step_speedup={estimated_step_reuse_speedup:.2f}x"
    )
    print(
        "[a5-restore-dataset] restore_diff="
        f"{payload['restored_nominal_local_state_max_abs']:.3e} "
        f"repeat_diff={repeat_check['local_state_max_abs']:.3e}"
    )
    print(f"[a5-restore-dataset] wrote {args.json} and {args.csv}")


if __name__ == "__main__":
    raise SystemExit(main())
