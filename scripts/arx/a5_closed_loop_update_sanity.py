"""Per-anchor closed-loop update sanity for A5 restored-anchor labels."""
import argparse
import csv
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import genesis as gs

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from a5_pusher_forward_sanity import _make_scene  # noqa: E402
from a5_restore_response_dataset import _capture_anchor, _query_from_anchor  # noqa: E402
from a5_fd_response_dataset import _tensor_vec, _vec3  # noqa: E402


DEFAULT_DATA = Path(
    "analysis/2026-07-09_arx_pusher/stage2_short1_eps3e3_shard1/"
    "a5_stage2_short1_eps3e3_shard1.csv"
)
DEFAULT_OUT = Path("analysis/2026-07-09_arx_pusher/a5_closed_loop_update_sanity.json")


def _loads_vec(text):
    return np.asarray(json.loads(text), dtype=np.float64)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "none", "")


def _parse_scales(text):
    return [float(x) for x in text.split(",") if x.strip()]


def _read_anchor_rows(path, anchor_id, require_keep):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["anchor_id"]) != anchor_id:
                continue
            if require_keep and not _as_bool(row.get("keep", True)):
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no rows for anchor_id={anchor_id} in {path}")
    return rows


def _fit_response_matrix(rows, target_name):
    x = np.stack([_loads_vec(row["direction_vec"]) for row in rows], axis=0)
    y = np.stack([_loads_vec(row[target_name]) for row in rows], axis=0)
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    target_rms = float(np.sqrt(np.mean(y ** 2)))
    return {
        "coef_action_by_target": coef,
        "num_rows": len(rows),
        "rank": int(rank),
        "fit_rmse": rmse,
        "fit_target_rms": target_rms,
        "fit_relative_rmse": rmse / (target_rms + 1e-12),
        "action_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
    }


def _make_target(pos, target_dy):
    pos = np.asarray(pos, dtype=np.float64)
    target = pos.copy()
    target[1] += target_dy
    return target


def _loss_and_lambda(pos, target, target_dim, loss_dims):
    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dims = {"x": [0], "y": [1], "xy": [0, 1]}[loss_dims]
    diff = pos - target
    loss = float(sum(diff[i] * diff[i] for i in dims))
    lam = np.zeros(target_dim, dtype=np.float64)
    for i in dims:
        if i < target_dim:
            lam[i] = 2.0 * diff[i]
    return loss, lam


def _add(a, b):
    return [float(x) + float(y) for x, y in zip(a, b)]


def _query_loss(args_ns, scene, arm, obj, anchor_state, first_action, local_steps, total_steps, target_pos, target_dim, loss_dims):
    rec = _query_from_anchor(args_ns, scene, arm, obj, anchor_state, first_action, local_steps, total_steps)
    pos = _vec3(rec["local_pos"])
    loss, _ = _loss_and_lambda(pos, target_pos, target_dim, loss_dims)
    return {
        "loss": loss,
        "local_pos": pos,
        "local_qvel": _tensor_vec(rec["local_qvel"]),
        "target_pos": list(target_pos),
        "contact_trace": rec["contacts"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--anchor-id", type=int, default=5)
    parser.add_argument(
        "--target",
        choices=("local_response", "local_vel_response", "local_state_response"),
        default="local_state_response",
    )
    parser.add_argument("--require-keep", action="store_true")
    parser.add_argument("--target-dy", type=float, default=0.05)
    parser.add_argument("--loss-dims", choices=("x", "y", "xy"), default="y")
    parser.add_argument("--step-scales", default="1e-2,3e-2,1e-1,3e-1,1.0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = _read_anchor_rows(args.data, args.anchor_id, args.require_keep)
    fit = _fit_response_matrix(rows, args.target)
    coef = fit.pop("coef_action_by_target")
    first = rows[0]

    qpos = json.loads(first["qpos"])
    qvel = json.loads(first["qvel"])
    obj_pos = (float(first["obj_x"]), float(first["obj_y"]), float(first["obj_z"]))
    anchor_step = int(first["anchor_step"])
    response_steps = int(first["response_steps"])
    settle_steps = 20
    push_steps = max(anchor_step + response_steps + 2, 170)
    local_steps = response_steps + 1
    total_steps = local_steps
    args_ns = Namespace(
        qpos=qpos,
        qvel=qvel,
        settle_steps=settle_steps,
        push_steps=push_steps,
        response_steps=response_steps,
    )

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene, arm, obj = _make_scene(obj_pos, requires_grad=True)
    anchor = _capture_anchor(args_ns, scene, arm, obj, anchor_step, local_steps, total_steps)
    nominal_rec = _query_from_anchor(
        args_ns,
        scene,
        arm,
        obj,
        anchor["anchor_state"],
        qvel,
        local_steps,
        total_steps,
    )
    nominal_pos = _vec3(nominal_rec["local_pos"])
    target_pos = _make_target(nominal_pos, args.target_dy)
    nominal_loss, lam = _loss_and_lambda(
        nominal_pos, target_pos, coef.shape[1], args.loss_dims
    )
    nominal = {
        "loss": nominal_loss,
        "local_pos": nominal_pos,
        "local_qvel": _tensor_vec(nominal_rec["local_qvel"]),
        "target_pos": target_pos.tolist(),
        "contact_trace": nominal_rec["contacts"],
    }
    grad = coef @ lam
    grad_norm = float(np.linalg.norm(grad))
    if grad_norm < 1e-12:
        raise RuntimeError("zero fitted VJP gradient")
    descent_dir = -grad / grad_norm

    trials = []
    start = time.perf_counter()
    for scale in _parse_scales(args.step_scales):
        for kind, direction in (("descent", descent_dir), ("ascent", -descent_dir)):
            action = _add(qvel, scale * direction)
            rec = _query_loss(
                args_ns,
                scene,
                arm,
                obj,
                anchor["anchor_state"],
                action,
                local_steps,
                total_steps,
                target_pos,
                coef.shape[1],
                args.loss_dims,
            )
            trials.append(
                {
                    "kind": kind,
                    "scale": scale,
                    "loss": rec["loss"],
                    "loss_delta_vs_nominal": rec["loss"] - nominal["loss"],
                    "local_pos": rec["local_pos"],
                    "local_qvel": rec["local_qvel"],
                    "contact_trace": rec["contact_trace"],
                }
            )

    best = min(trials, key=lambda row: row["loss"])
    payload = {
        "description": "A5 per-anchor closed-loop update sanity",
        "data": str(args.data),
        "anchor_id": args.anchor_id,
        "target": args.target,
        "fit": fit,
        "qpos": qpos,
        "qvel": qvel,
        "obj_pos": list(obj_pos),
        "anchor_step": anchor_step,
        "response_steps": response_steps,
        "target_dy": args.target_dy,
        "loss_dims": args.loss_dims,
        "nominal": nominal,
        "target_pos": target_pos.tolist(),
        "lambda": lam.tolist(),
        "vjp_grad": grad.tolist(),
        "vjp_grad_norm": grad_norm,
        "trials": trials,
        "best": best,
        "query_seconds": time.perf_counter() - start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[a5-update] anchor={args.anchor_id} target={args.target} "
        f"fit_rel={fit['fit_relative_rmse']:.4f} nominal_loss={nominal['loss']:.8f} "
        f"best={best['kind']}@{best['scale']:.1e} delta={best['loss_delta_vs_nominal']:+.8e}"
    )
    for row in trials:
        print(
            f"[a5-update] {row['kind']:7s} scale={row['scale']:.1e} "
            f"loss={row['loss']:.8f} delta={row['loss_delta_vs_nominal']:+.8e}"
        )
    print(f"[a5-update] wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
