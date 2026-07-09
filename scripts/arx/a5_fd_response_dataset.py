"""Collect finite-difference object-response labels for the A5 clean anchor.

Rows from this script are the first supervised labels for action-side VJP:

    A(z_anchor) v_k ~= d_k

where d_k is measured from paired Genesis forward rollouts with
qvel(anchor_step) +/- eps * v_k.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import genesis as gs

from a5_pusher_forward_sanity import (
    DEFAULT_OBJ_POS,
    DEFAULT_PUSH_STEPS,
    DEFAULT_QPOS,
    DEFAULT_QVEL,
    _contact_count,
    _flat_pos,
    _make_scene,
)


DEFAULT_JSON = Path("analysis/2026-07-09_arx_pusher/a5_fd_response_dataset_clean_anchor.json")
DEFAULT_CSV = Path("analysis/2026-07-09_arx_pusher/a5_fd_response_dataset_clean_anchor.csv")
DEFAULT_SCALES = "3e-4,1e-3"


def _parse_vec(text, expected, name):
    values = [float(x) for x in text.split(",") if x.strip()]
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"{name} expects {expected} comma-separated values, got {len(values)}")
    return values


def _parse_scales(text):
    return [float(x) for x in text.split(",") if x.strip()]


def _normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec
    return [x / norm for x in vec]


def _directions(qvel, num_random, seed):
    dirs = []
    for i in range(len(qvel)):
        v = [0.0 for _ in qvel]
        v[i] = 1.0
        dirs.append({"name": f"joint{i + 1}+", "vec": v})

    nominal = _normalize(list(qvel))
    if any(abs(x) > 1e-12 for x in nominal):
        dirs.append({"name": "nominal_qvel", "vec": nominal})

    rng = random.Random(seed)
    for i in range(num_random):
        v = [rng.gauss(0.0, 1.0) for _ in qvel]
        dirs.append({"name": f"random{i + 1:02d}", "vec": _normalize(v)})
    return dirs


def _rollout(args, scene, arm, obj, anchor_step=None, perturb=None):
    scene.reset()
    arm.set_dofs_position(torch.tensor(args.qpos, dtype=torch.float32))

    zero = gs.tensor([0.0 for _ in args.qvel])
    contacts = []
    for _ in range(args.settle_steps):
        arm.set_dofs_velocity(zero)
        scene.step()
        contacts.append(_contact_count(scene))

    initial_pos = _flat_pos(obj)
    initial_qvel = _flat_qvel(obj)
    trace = []
    qvel_trace = []
    local_pos = None
    local_qvel = None
    capture_full_qvel_trace = perturb is None
    for step in range(args.push_steps):
        values = list(args.qvel)
        if perturb is not None and step == anchor_step:
            values = [values[i] + perturb[i] for i in range(len(values))]
        arm.set_dofs_velocity(gs.tensor(values))
        scene.step()
        contacts.append(_contact_count(scene))
        pos = _flat_pos(obj)
        trace.append([float(x) for x in pos.tolist()])
        qvel = None
        if capture_full_qvel_trace:
            qvel = _flat_qvel(obj)
            qvel_trace.append([float(x) for x in qvel.tolist()])
        if anchor_step is not None and step == anchor_step + args.response_steps:
            if qvel is None:
                qvel = _flat_qvel(obj)
            local_pos = pos
            local_qvel = qvel

    final_pos = _flat_pos(obj)
    final_qvel = _flat_qvel(obj)
    if local_pos is None:
        local_pos = final_pos
    if local_qvel is None:
        local_qvel = final_qvel
    return {
        "initial_pos": initial_pos,
        "initial_qvel": initial_qvel,
        "local_pos": local_pos,
        "local_qvel": local_qvel,
        "final_pos": final_pos,
        "final_qvel": final_qvel,
        "trace": trace,
        "qvel_trace": qvel_trace,
        "contacts": contacts,
    }


def _trace_pos(trace, idx):
    if not trace:
        return [0.0, 0.0, 0.0]
    idx = max(0, min(idx, len(trace) - 1))
    return list(trace[idx])


def _trace_vec(trace, idx, length):
    if not trace:
        return [0.0 for _ in range(length)]
    idx = max(0, min(idx, len(trace) - 1))
    values = list(trace[idx])
    if len(values) < length:
        values = values + [0.0] * (length - len(values))
    return values[:length]


def _contact_slice(contacts, center, radius):
    if not contacts:
        return []
    lo = max(0, center - radius)
    hi = min(len(contacts), center + radius + 1)
    return contacts[lo:hi]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _contact_stats(contacts, center, radius):
    values = _contact_slice(contacts, center, radius)
    return {
        "mean": float(_mean(values)),
        "max": int(max(values)) if values else 0,
        "min": int(min(values)) if values else 0,
        "trace": [int(x) for x in values],
    }


def _nominal_context(args, nominal, anchor_step):
    contact_center = args.settle_steps + anchor_step
    anchor_pos = _trace_pos(nominal["trace"], anchor_step)
    pre_pos = _trace_pos(nominal["trace"], anchor_step - args.response_steps)
    post_pos = _trace_pos(nominal["trace"], anchor_step + args.response_steps)
    anchor_qvel = _trace_vec(nominal["qvel_trace"], anchor_step, 6)
    pre_qvel = _trace_vec(nominal["qvel_trace"], anchor_step - args.response_steps, 6)
    post_qvel = _trace_vec(nominal["qvel_trace"], anchor_step + args.response_steps, 6)
    final_pos = _vec3(nominal["final_pos"])
    initial_pos = _vec3(nominal["initial_pos"])
    final_qvel = _tensor_vec(nominal["final_qvel"])
    initial_qvel = _tensor_vec(nominal["initial_qvel"])
    contact = _contact_stats(nominal["contacts"], contact_center, args.contact_window)
    return {
        "nominal_anchor_pos": anchor_pos,
        "nominal_pre_pos": pre_pos,
        "nominal_post_pos": post_pos,
        "nominal_initial_pos": initial_pos,
        "nominal_final_pos": final_pos,
        "nominal_anchor_qvel": anchor_qvel,
        "nominal_pre_qvel": pre_qvel,
        "nominal_post_qvel": post_qvel,
        "nominal_initial_qvel": initial_qvel,
        "nominal_final_qvel": final_qvel,
        "nominal_pre_disp": _sub(anchor_pos, pre_pos),
        "nominal_post_disp": _sub(post_pos, anchor_pos),
        "nominal_total_disp": _sub(final_pos, initial_pos),
        "nominal_pre_qvel_delta": _sub(anchor_qvel, pre_qvel),
        "nominal_post_qvel_delta": _sub(post_qvel, anchor_qvel),
        "nominal_total_qvel_delta": _sub(final_qvel, initial_qvel),
        "nominal_contact_mean": contact["mean"],
        "nominal_contact_max": contact["max"],
        "nominal_contact_min": contact["min"],
        "nominal_contact_trace": contact["trace"],
    }


def _choose_anchor_step(trace, threshold):
    if len(trace) < 2:
        return 0
    prev = trace[0]
    best_step = 0
    best_delta = 0.0
    for i in range(1, len(trace)):
        dx = trace[i][0] - prev[0]
        dy = trace[i][1] - prev[1]
        dz = trace[i][2] - prev[2]
        delta = math.sqrt(dx * dx + dy * dy + dz * dz)
        if delta > best_delta:
            best_delta = delta
            best_step = i
        if delta >= threshold:
            return i
        prev = trace[i]
    return best_step


def _vec3(tensor):
    return [float(x) for x in tensor.detach().cpu().tolist()]


def _flat_qvel(ent):
    return ent.get_dofs_velocity().detach().cpu().reshape(-1)


def _tensor_vec(tensor):
    return [float(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def _sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def _scale(a, s):
    return [x * s for x in a]


def _filter_status(args, rec_plus, rec_minus, anchor_step, local_response, final_response):
    center = args.settle_steps + anchor_step
    plus_stats = _contact_stats(rec_plus["contacts"], center, args.contact_window)
    minus_stats = _contact_stats(rec_minus["contacts"], center, args.contact_window)
    contact_mean_delta = abs(plus_stats["mean"] - minus_stats["mean"])
    contact_max_delta = abs(plus_stats["max"] - minus_stats["max"])
    local_norm = math.sqrt(sum(x * x for x in local_response))
    final_norm = math.sqrt(sum(x * x for x in final_response))
    reasons = []
    if contact_mean_delta > args.max_contact_mean_delta:
        reasons.append("contact_mean_delta")
    if contact_max_delta > args.max_contact_max_delta:
        reasons.append("contact_max_delta")
    if final_norm < args.min_response_norm:
        reasons.append("response_too_small")
    if final_norm > args.max_response_norm:
        reasons.append("response_too_large")
    if not all(math.isfinite(x) for x in local_response + final_response):
        reasons.append("nonfinite_response")
    return {
        "keep": not reasons,
        "filter_reason": "ok" if not reasons else "|".join(reasons),
        "contact_mean_delta": float(contact_mean_delta),
        "contact_max_delta": float(contact_max_delta),
        "plus_contact_mean": plus_stats["mean"],
        "minus_contact_mean": minus_stats["mean"],
        "plus_contact_max": plus_stats["max"],
        "minus_contact_max": minus_stats["max"],
        "plus_contact_trace": plus_stats["trace"],
        "minus_contact_trace": minus_stats["trace"],
        "local_response_norm": local_norm,
        "final_response_norm": final_norm,
    }


def _collect_rows(args, scene, arm, obj, anchor_step, directions, scales, nominal_context):
    rows = []
    for direction in directions:
        vec = direction["vec"]
        for eps in scales:
            plus = [eps * x for x in vec]
            minus = [-eps * x for x in vec]
            rec_plus = _rollout(args, scene, arm, obj, anchor_step=anchor_step, perturb=plus)
            rec_minus = _rollout(args, scene, arm, obj, anchor_step=anchor_step, perturb=minus)

            local_plus = _vec3(rec_plus["local_pos"])
            local_minus = _vec3(rec_minus["local_pos"])
            final_plus = _vec3(rec_plus["final_pos"])
            final_minus = _vec3(rec_minus["final_pos"])
            local_qvel_plus = _tensor_vec(rec_plus["local_qvel"])
            local_qvel_minus = _tensor_vec(rec_minus["local_qvel"])
            final_qvel_plus = _tensor_vec(rec_plus["final_qvel"])
            final_qvel_minus = _tensor_vec(rec_minus["final_qvel"])
            local_response = _scale(_sub(local_plus, local_minus), 1.0 / (2.0 * eps))
            final_response = _scale(_sub(final_plus, final_minus), 1.0 / (2.0 * eps))
            local_vel_response = _scale(_sub(local_qvel_plus, local_qvel_minus), 1.0 / (2.0 * eps))
            final_vel_response = _scale(_sub(final_qvel_plus, final_qvel_minus), 1.0 / (2.0 * eps))
            local_state_response = local_response + local_vel_response
            final_state_response = final_response + final_vel_response
            filter_status = _filter_status(args, rec_plus, rec_minus, anchor_step, local_response, final_response)
            row = {
                "direction": direction["name"],
                "direction_vec": vec,
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
                "local_vel_response_norm": math.sqrt(sum(x * x for x in local_vel_response)),
                "final_vel_response_norm": math.sqrt(sum(x * x for x in final_vel_response)),
                "local_state_response_norm": math.sqrt(sum(x * x for x in local_state_response)),
                "final_state_response_norm": math.sqrt(sum(x * x for x in final_state_response)),
                "max_total_contact_count_plus": max(rec_plus["contacts"]) if rec_plus["contacts"] else 0,
                "max_total_contact_count_minus": max(rec_minus["contacts"]) if rec_minus["contacts"] else 0,
                **nominal_context,
                **filter_status,
            }
            if row["keep"] or not args.drop_filtered:
                rows.append(row)
    return rows


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "direction",
        "epsilon",
        "local_response_norm",
        "final_response_norm",
        "direction_vec",
        "local_response",
        "final_response",
        "local_plus",
        "local_minus",
        "final_plus",
        "final_minus",
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
        "local_vel_response_norm",
        "final_vel_response_norm",
        "local_state_response_norm",
        "final_state_response_norm",
        "local_vel_response",
        "final_vel_response",
        "local_state_response",
        "final_state_response",
        "max_total_contact_count_plus",
        "max_total_contact_count_minus",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key in (
                "direction_vec",
                "local_response",
                "final_response",
                "local_plus",
                "local_minus",
                "final_plus",
                "final_minus",
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
                "local_vel_response",
                "final_vel_response",
                "local_state_response",
                "final_state_response",
            ):
                flat[key] = json.dumps(flat[key])
            writer.writerow(flat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-x", type=float, default=DEFAULT_OBJ_POS[0])
    parser.add_argument("--obj-y", type=float, default=DEFAULT_OBJ_POS[1])
    parser.add_argument("--obj-z", type=float, default=DEFAULT_OBJ_POS[2])
    parser.add_argument("--qpos", type=lambda x: _parse_vec(x, 6, "qpos"), default=DEFAULT_QPOS)
    parser.add_argument("--qvel", type=lambda x: _parse_vec(x, 6, "qvel"), default=DEFAULT_QVEL)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--push-steps", type=int, default=DEFAULT_PUSH_STEPS)
    parser.add_argument("--response-steps", type=int, default=1)
    parser.add_argument("--anchor-step", type=int, default=-1)
    parser.add_argument("--motion-threshold", type=float, default=2e-5)
    parser.add_argument("--contact-window", type=int, default=5)
    parser.add_argument("--max-contact-mean-delta", type=float, default=8.0)
    parser.add_argument("--max-contact-max-delta", type=float, default=12.0)
    parser.add_argument("--min-response-norm", type=float, default=1e-8)
    parser.add_argument("--max-response-norm", type=float, default=1e3)
    parser.add_argument("--drop-filtered", action="store_true")
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--num-random", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    obj_pos = (args.obj_x, args.obj_y, args.obj_z)
    scene, arm, obj = _make_scene(obj_pos, requires_grad=True)

    nominal = _rollout(args, scene, arm, obj)
    anchor_step = args.anchor_step
    if anchor_step < 0:
        anchor_step = _choose_anchor_step(nominal["trace"], args.motion_threshold)

    directions = _directions(args.qvel, args.num_random, args.seed)
    scales = _parse_scales(args.scales)
    nominal_context = _nominal_context(args, nominal, anchor_step)
    rows = _collect_rows(args, scene, arm, obj, anchor_step, directions, scales, nominal_context)

    initial_pos = _vec3(nominal["initial_pos"])
    final_pos = _vec3(nominal["final_pos"])
    displacement = _sub(final_pos, initial_pos)
    payload = {
        "description": "A5 clean-anchor finite-difference response dataset",
        "qpos": args.qpos,
        "qvel": args.qvel,
        "object_pos": obj_pos,
        "settle_steps": args.settle_steps,
        "push_steps": args.push_steps,
        "response_steps": args.response_steps,
        "anchor_step": anchor_step,
        "scales": scales,
        "num_random": args.num_random,
        "seed": args.seed,
        "contact_window": args.contact_window,
        "max_contact_mean_delta": args.max_contact_mean_delta,
        "max_contact_max_delta": args.max_contact_max_delta,
        "drop_filtered": args.drop_filtered,
        "initial_pos": initial_pos,
        "final_pos": final_pos,
        "displacement": displacement,
        "horizontal_disp": math.sqrt(displacement[0] ** 2 + displacement[1] ** 2),
        "vertical_disp": displacement[2],
        "num_rows": len(rows),
        "num_kept_rows": sum(1 for row in rows if row["keep"]),
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(args.csv, rows)
    print(
        f"[a5-fd-dataset] anchor_step={anchor_step} rows={len(rows)} "
        f"hdisp={payload['horizontal_disp']:.6f} vdisp={payload['vertical_disp']:.6f}"
    )
    print(f"[a5-fd-dataset] wrote {args.json} and {args.csv}")


if __name__ == "__main__":
    raise SystemExit(main())
