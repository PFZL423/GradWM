"""FD-vs-autograd diagnostic at the first usable A5 Pusher-like anchor.

This probes whether a one-step joint-velocity perturbation changes the final
object position, and whether Genesis autograd reports a matching gradient.
It is a diagnostic only; it does not train the learned VJP module.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import torch
import genesis as gs

from a5_pusher_forward_sanity import (
    DEFAULT_OBJ_POS,
    DEFAULT_OUT as FORWARD_OUT,
    DEFAULT_PUSH_STEPS,
    DEFAULT_QPOS,
    DEFAULT_QVEL,
    _contact_count,
    _flat_pos,
    _make_scene,
)


DEFAULT_JSON = Path("analysis/2026-07-09_arx_pusher/a5_contact_fd_diag.json")
DEFAULT_CSV = Path("analysis/2026-07-09_arx_pusher/a5_contact_fd_diag.csv")
DEFAULT_SCALES = "1e-4,3e-4,1e-3"


def _parse_vec(text, expected, name):
    values = [float(x) for x in text.split(",") if x.strip()]
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"{name} expects {expected} comma-separated values, got {len(values)}")
    return values


def _parse_scales(text):
    return [float(x) for x in text.split(",") if x.strip()]


def _axis_directions(qvel):
    n_dofs = len(qvel)
    directions = []
    for i in range(n_dofs):
        v = [0.0 for _ in range(n_dofs)]
        v[i] = 1.0
        directions.append({"name": f"joint{i + 1}+", "vec": v})
    nominal = list(qvel)
    norm = math.sqrt(sum(x * x for x in nominal))
    if norm > 0.0:
        directions.append({"name": "nominal_qvel", "vec": [x / norm for x in nominal]})
    return directions


def _probe_scalar(pos, kind):
    if kind == "obj_y":
        return pos[1]
    if kind == "obj_x":
        return pos[0]
    if kind == "obj_xy_goal":
        target = gs.tensor([0.306, 0.105])
        return ((pos[:2] - target) ** 2).sum()
    raise ValueError(kind)


def _rollout(scene, arm, obj, args, anchor_step=None, perturb=None, requires_grad=False):
    scene.reset()
    arm.set_dofs_position(torch.tensor(args.qpos, dtype=torch.float32))

    zero = torch.zeros(6, dtype=torch.float32)
    contacts = []
    for _ in range(args.settle_steps):
        arm.set_dofs_velocity(zero)
        scene.step()
        contacts.append(_contact_count(scene))

    initial_pos = _flat_pos(obj)
    trace = []
    anchor_tensor = None

    for step in range(args.push_steps):
        values = list(args.qvel)
        if perturb is not None and step == anchor_step:
            values = [values[i] + perturb[i] for i in range(len(values))]
        if requires_grad and step == anchor_step:
            qvel = gs.tensor(values, requires_grad=True)
            anchor_tensor = qvel
        else:
            qvel = gs.tensor(values)
        arm.set_dofs_velocity(qvel)
        scene.step()
        contacts.append(_contact_count(scene))
        pos = _flat_pos(obj)
        trace.append([float(x) for x in pos.detach().cpu().tolist()])

    final_pos = _flat_pos(obj)
    probe = _probe_scalar(final_pos, args.probe)
    return {
        "initial_pos": initial_pos,
        "final_pos": final_pos,
        "probe": probe,
        "trace": trace,
        "contacts": contacts,
        "anchor_tensor": anchor_tensor,
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
        delta = math.sqrt(dx * dx + dy * dy)
        if delta > best_delta:
            best_delta = delta
            best_step = i
        if delta >= threshold:
            return i
        prev = trace[i]
    return best_step


def _analytic(scene, arm, obj, args, anchor_step):
    record = _rollout(scene, arm, obj, args, anchor_step=anchor_step, requires_grad=True)
    probe = record["probe"]
    anchor_tensor = record["anchor_tensor"]
    status = "ok"
    grad = None
    try:
        probe.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if anchor_tensor is None or anchor_tensor.grad is None:
            status = "grad_none"
            grad = [0.0 for _ in args.qvel]
        elif torch.isnan(anchor_tensor.grad).any():
            status = "grad_nan"
        else:
            grad = [float(x) for x in anchor_tensor.grad.detach().cpu().tolist()]
    except RuntimeError as exc:
        if "does not require grad" in str(exc):
            status = "loss_has_no_grad_path"
            grad = [0.0 for _ in args.qvel]
        else:
            status = f"backward_error:{type(exc).__name__}:{str(exc)[:160]}"
    except Exception as exc:
        status = f"backward_error:{type(exc).__name__}:{str(exc)[:160]}"
    return record, status, grad


def _fd_rows(scene, arm, obj, args, anchor_step, scales, directions, analytic_grad):
    rows = []
    for direction in directions:
        vec = direction["vec"]
        analytic_directional = None
        if analytic_grad is not None:
            analytic_directional = sum(analytic_grad[i] * vec[i] for i in range(len(vec)))
        for scale in scales:
            plus = [scale * x for x in vec]
            minus = [-scale * x for x in vec]
            rec_plus = _rollout(scene, arm, obj, args, anchor_step=anchor_step, perturb=plus)
            rec_minus = _rollout(scene, arm, obj, args, anchor_step=anchor_step, perturb=minus)
            val_plus = float(rec_plus["probe"].detach().cpu().item())
            val_minus = float(rec_minus["probe"].detach().cpu().item())
            fd = (val_plus - val_minus) / (2.0 * scale)
            abs_error = None if analytic_directional is None else abs(fd - analytic_directional)
            rel_error = None if abs_error is None else abs_error / (abs(fd) + 1e-12)
            rows.append({
                "direction": direction["name"],
                "scale": scale,
                "fd_directional": fd,
                "analytic_directional": analytic_directional,
                "abs_error": abs_error,
                "rel_error": rel_error,
                "val_plus": val_plus,
                "val_minus": val_minus,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=("obj_y", "obj_x", "obj_xy_goal"), default="obj_y")
    parser.add_argument("--obj-x", type=float, default=DEFAULT_OBJ_POS[0])
    parser.add_argument("--obj-y", type=float, default=DEFAULT_OBJ_POS[1])
    parser.add_argument("--obj-z", type=float, default=DEFAULT_OBJ_POS[2])
    parser.add_argument("--qpos", type=lambda x: _parse_vec(x, 6, "qpos"), default=DEFAULT_QPOS)
    parser.add_argument("--qvel", type=lambda x: _parse_vec(x, 6, "qvel"), default=DEFAULT_QVEL)
    parser.add_argument("--anchor-step", type=int, default=-1, help="-1 means choose from nominal object-motion trace")
    parser.add_argument("--motion-threshold", type=float, default=2e-5)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--push-steps", type=int, default=DEFAULT_PUSH_STEPS)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    obj_pos = (args.obj_x, args.obj_y, args.obj_z)
    scene, arm, obj = _make_scene(obj_pos, requires_grad=True)

    nominal = _rollout(scene, arm, obj, args)
    anchor_step = args.anchor_step
    if anchor_step < 0:
        anchor_step = _choose_anchor_step(nominal["trace"], args.motion_threshold)

    analytic_record, analytic_status, analytic_grad = _analytic(scene, arm, obj, args, anchor_step)
    directions = _axis_directions(args.qvel)
    scales = _parse_scales(args.scales)
    rows = _fd_rows(scene, arm, obj, args, anchor_step, scales, directions, analytic_grad)

    final_pos = analytic_record["final_pos"].detach().cpu()
    initial_pos = analytic_record["initial_pos"].detach().cpu()
    disp = final_pos - initial_pos
    analytic_norm = None
    if analytic_grad is not None:
        analytic_norm = math.sqrt(sum(x * x for x in analytic_grad))

    payload = {
        "description": "A5 Pusher-like contact FD-vs-autograd diagnostic",
        "source_forward_sanity": str(FORWARD_OUT),
        "qpos": args.qpos,
        "qvel": args.qvel,
        "object_pos": obj_pos,
        "probe": args.probe,
        "anchor_step": anchor_step,
        "settle_steps": args.settle_steps,
        "push_steps": args.push_steps,
        "analytic_status": analytic_status,
        "analytic_grad": analytic_grad,
        "analytic_grad_norm": analytic_norm,
        "initial_pos": [float(x) for x in initial_pos.tolist()],
        "final_pos": [float(x) for x in final_pos.tolist()],
        "displacement": [float(x) for x in disp.tolist()],
        "horizontal_disp": float(disp[:2].norm().item()),
        "vertical_disp": float(disp[2].item()),
        "max_total_contact_count": max(analytic_record["contacts"]) if analytic_record["contacts"] else 0,
        "rows": rows,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "direction",
                "scale",
                "fd_directional",
                "analytic_directional",
                "abs_error",
                "rel_error",
                "val_plus",
                "val_minus",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    fd_nonzero = [r for r in rows if abs(r["fd_directional"]) > 1e-5]
    print(
        f"[a5-contact-fd] status={analytic_status} anchor_step={anchor_step} "
        f"analytic_norm={analytic_norm} fd_nonzero={len(fd_nonzero)}/{len(rows)}"
    )
    print(f"[a5-contact-fd] wrote {args.json} and {args.csv}")


if __name__ == "__main__":
    raise SystemExit(main())
