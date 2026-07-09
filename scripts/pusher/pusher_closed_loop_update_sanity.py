"""Single-anchor closed-loop action-update sanity for simple pusher.

The script fits a local response matrix from a restored-anchor FD CSV:

    d_hat = A v

Then it uses the VJP direction ``A^T lambda`` to perturb the anchor action and
checks whether the true Genesis restored-anchor rollout reduces object loss.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pusher.pusher_like_arm_sanity import OBJECT_POS, TARGET_POS, TOTAL_DOFS, _make_scene  # noqa: E402
from pusher.pusher_restore_response_dataset import (  # noqa: E402
    _capture_anchor,
    _choose_anchor_step,
    _program_qvel,
    _query_from_anchor,
    _rollout_nominal,
)


DEFAULT_DATA = Path("analysis/2026-07-09_arx_pusher/simple_pusher_restore_j246_clean64_resp1.csv")
DEFAULT_OUT = Path("analysis/2026-07-09_arx_pusher/simple_pusher_closed_loop_update_sanity.json")


def _loads_vec(text):
    return np.asarray(json.loads(text), dtype=np.float64)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "none", "")


def _parse_scales(text):
    return [float(x) for x in text.split(",") if x.strip()]


def _fit_response_matrix(path, target_name, require_keep):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if require_keep and not _as_bool(row.get("keep", True)):
                continue
            rows.append(
                {
                    "v": _loads_vec(row["direction_vec"]),
                    "target": _loads_vec(row[target_name]),
                }
            )
    if len(rows) < 2:
        raise RuntimeError(f"not enough rows in {path}: {len(rows)}")
    x = np.stack([row["v"] for row in rows], axis=0)
    y = np.stack([row["target"] for row in rows], axis=0)
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


def _loss_and_lambda(pos, target, target_dim, loss_dims):
    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dim_mask = {
        "x": [0],
        "xy": [0, 1],
        "xyz": [0, 1, 2],
    }[loss_dims]
    diff = pos - target
    loss = float(sum(diff[i] * diff[i] for i in dim_mask))
    lam = np.zeros(target_dim, dtype=np.float64)
    for i in dim_mask:
        if i < target_dim:
            lam[i] = 2.0 * diff[i]
    return loss, lam


def _add(a, b):
    return [float(x) + float(y) for x, y in zip(a, b)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--target", choices=("local_response", "local_state_response"), default="local_response")
    parser.add_argument("--program", default="j2j4j6+")
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--close-speed", type=float, default=1.5)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--push-steps", type=int, default=80)
    parser.add_argument("--response-steps", type=int, default=1)
    parser.add_argument("--anchor-step", type=int, default=-1)
    parser.add_argument("--motion-threshold", type=float, default=2e-5)
    parser.add_argument("--loss-dims", choices=("x", "xy", "xyz"), default="xy")
    parser.add_argument("--step-scales", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    parser.add_argument("--require-keep", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    fit = _fit_response_matrix(args.data, args.target, args.require_keep)
    coef = fit.pop("coef_action_by_target")

    scene, arm, obj = _make_scene()
    if arm.n_dofs != TOTAL_DOFS:
        raise RuntimeError(f"expected arm n_dofs={TOTAL_DOFS}, got {arm.n_dofs}")

    qvel = _program_qvel(args.program, args.speed)
    nominal_rollout = _rollout_nominal(args, scene, arm, obj, qvel)
    anchor_step = args.anchor_step
    if anchor_step < 0:
        anchor_step = _choose_anchor_step(nominal_rollout["trace"], args.motion_threshold)
    local_steps = max(1, args.response_steps)
    anchor = _capture_anchor(args, scene, arm, obj, qvel, anchor_step, local_steps)
    nominal_query = _query_from_anchor(scene, arm, obj, anchor["anchor_state"], qvel, qvel, local_steps)

    nominal_loss, lam = _loss_and_lambda(
        nominal_query["local_pos"], TARGET_POS, coef.shape[1], args.loss_dims
    )
    grad = coef @ lam
    grad_norm = float(np.linalg.norm(grad))
    if grad_norm < 1e-12:
        raise RuntimeError("zero fitted VJP gradient")
    descent_dir = -grad / grad_norm

    rows = []
    start = time.perf_counter()
    for scale in _parse_scales(args.step_scales):
        for kind, direction in (("descent", descent_dir), ("ascent", -descent_dir)):
            action = _add(qvel, scale * direction)
            rec = _query_from_anchor(scene, arm, obj, anchor["anchor_state"], qvel, action, local_steps)
            loss, _ = _loss_and_lambda(rec["local_pos"], TARGET_POS, coef.shape[1], args.loss_dims)
            rows.append(
                {
                    "kind": kind,
                    "scale": scale,
                    "loss": loss,
                    "loss_delta_vs_nominal": loss - nominal_loss,
                    "local_pos": rec["local_pos"],
                    "contact_trace": rec["contacts"],
                }
            )

    best = min(rows, key=lambda row: row["loss"])
    payload = {
        "description": "Simple pusher single-anchor closed-loop update sanity",
        "data": str(args.data),
        "target": args.target,
        "fit": fit,
        "program": args.program,
        "speed": args.speed,
        "qvel": qvel,
        "object_pos": list(OBJECT_POS),
        "target_pos": list(TARGET_POS),
        "loss_dims": args.loss_dims,
        "anchor_step": anchor_step,
        "response_steps": args.response_steps,
        "nominal_local_pos": nominal_query["local_pos"],
        "nominal_loss": nominal_loss,
        "lambda": lam.tolist(),
        "vjp_grad": grad.tolist(),
        "vjp_grad_norm": grad_norm,
        "rows": rows,
        "best": best,
        "query_seconds": time.perf_counter() - start,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[pusher-update] target={args.target} fit_rel={fit['fit_relative_rmse']:.4f} "
        f"nominal_loss={nominal_loss:.8f} best={best['kind']}@{best['scale']:.1e} "
        f"delta={best['loss_delta_vs_nominal']:+.8e}"
    )
    for row in rows:
        print(
            f"[pusher-update] {row['kind']:7s} scale={row['scale']:.1e} "
            f"loss={row['loss']:.8f} delta={row['loss_delta_vs_nominal']:+.8e}"
        )
    print(f"[pusher-update] wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
