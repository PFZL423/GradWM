"""Minimal trajectory optimization for the Genesis rope-grasp scene.

Outer process owns Adam state. Each iteration spawns one fresh worker process
that builds a Genesis scene, runs one rollout/backward, and returns loss/grad.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RUNTIME_CACHE_ROOT = Path(tempfile.gettempdir()) / "genisis_runtime"


def _configure_runtime_dirs():
    defaults = {
        "NUMBA_CACHE_DIR": RUNTIME_CACHE_ROOT / "numba",
        "MPLCONFIGDIR": RUNTIME_CACHE_ROOT / "matplotlib",
        "XDG_CACHE_HOME": RUNTIME_CACHE_ROOT / "xdg",
        "GS_CACHE_FILE_PATH": RUNTIME_CACHE_ROOT / "genesis",
        "QD_OFFLINE_CACHE_FILE_PATH": RUNTIME_CACHE_ROOT / "qdcache",
    }
    for key, path in defaults.items():
        os.environ.setdefault(key, str(path))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


_configure_runtime_dirs()
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import genesis as gs

import grasp_scene as gs_mod
from grasp_scene import (
    _enable_arm_contact_geoms,
    _grayscale_arm_geoms,
    _write_temp_mjcf,
    APPROACH_QVEL,
    CLOSE_QVEL,
    LIFT_QVEL,
    TARGET_QVEL,
    ARM_LPOSE_QPOS,
)
from make_arm_mjcf import TOTAL_DOFS, make_arm_gripper_mjcf

OUT_CSV = Path("analysis/traj_opt_loss.csv")
OUT_PNG = Path("analysis/traj_opt_loss.png")
OUT_QVEL = Path("analysis/traj_opt_qvel.json")

N_SETTLE = 30
FINGER_OPEN = 0.008
CABLE_REST_Z_BUMP = 0.013
CABLE_PARAMS = {
    "N_CABLE_SEG": 80,
    "CABLE_SEG_RADIUS": 0.005,
    "CABLE_DAMPING": 9e-3,
    "CABLE_ARMATURE": 2e-5,
    "CABLE_SEG_MASS": 4e-4,
}
PATCH_KEYS = list(CABLE_PARAMS.keys())


def _make_initial_velocities(horizon, mode):
    if mode == "zero":
        return torch.zeros((horizon, TOTAL_DOFS), dtype=torch.float32)
    rows = []
    for i in range(horizon):
        if i < 15:
            rows.append(APPROACH_QVEL)
        elif i < 35:
            rows.append(CLOSE_QVEL)
        else:
            rows.append(LIFT_QVEL)
    return torch.tensor(rows, dtype=torch.float32)


def _make_target_qvel(name):
    targets = {
        "close": TARGET_QVEL,
        "approach": APPROACH_QVEL,
        "lift": LIFT_QVEL,
        "zero": [0.0] * TOTAL_DOFS,
    }
    try:
        return list(targets[name])
    except KeyError as e:
        raise ValueError(f"unknown target preset={name!r}") from e


def _make_tables_mjcf():
    table_top_thick = gs_mod.TABLE_TOP_THICK
    leg_half = gs_mod.TABLE_LEG_HALF
    leg_top_z = gs_mod.TABLE_TOP_Z - table_top_thick
    leg_half_z = leg_top_z * 0.5
    return f"""<mujoco model="table_scene">
    <worldbody>
        <geom name="table_L_top" type="box" pos="{gs_mod.TABLE_X_LEFT} 0 {gs_mod.TABLE_TOP_Z}"
              size="{gs_mod.TABLE_TOP_HALF} {gs_mod.TABLE_TOP_HALF_Y} {table_top_thick}"
              rgba="0.55 0.40 0.25 1" contype="1" conaffinity="1"/>
        <geom name="table_L_leg" type="box" pos="{gs_mod.TABLE_X_LEFT} 0 {leg_half_z}"
              size="{leg_half} {leg_half} {leg_half_z}"
              rgba="0.55 0.40 0.25 1" contype="0" conaffinity="0"/>
        <geom name="table_R_top" type="box" pos="{gs_mod.TABLE_X_RIGHT} 0 {gs_mod.TABLE_TOP_Z}"
              size="{gs_mod.TABLE_TOP_HALF} {gs_mod.TABLE_TOP_HALF_Y} {table_top_thick}"
              rgba="0.55 0.40 0.25 1" contype="1" conaffinity="1"/>
        <geom name="table_R_leg" type="box" pos="{gs_mod.TABLE_X_RIGHT} 0 {leg_half_z}"
              size="{leg_half} {leg_half} {leg_half_z}"
              rgba="0.55 0.40 0.25 1" contype="0" conaffinity="0"/>
    </worldbody>
</mujoco>
"""


def _build_scene(scene_kind):
    saved = {k: getattr(gs_mod, k) for k in PATCH_KEYS}
    saved_z = gs_mod.CABLE_REST_Z
    try:
        for k, v in CABLE_PARAMS.items():
            setattr(gs_mod, k, v)
        gs_mod.CABLE_REST_Z = saved_z + CABLE_REST_Z_BUMP
        env_xml = gs_mod._make_bridge_scene_mjcf() if scene_kind == "rope" else _make_tables_mjcf()
    finally:
        for k, v in saved.items():
            setattr(gs_mod, k, v)
        gs_mod.CABLE_REST_Z = saved_z

    arm_xml = _grayscale_arm_geoms(_enable_arm_contact_geoms(
        make_arm_gripper_mjcf(finger_open=FINGER_OPEN, finger_range=(0.0, FINGER_OPEN))
    ))
    arm_tmp = _write_temp_mjcf("traj_opt_arm_", arm_xml)
    env_tmp = _write_temp_mjcf(f"traj_opt_{scene_kind}_", env_xml)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=4, substeps_local=4, requires_grad=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))
    arm = scene.add_entity(gs.morphs.MJCF(file=arm_tmp))
    scene.add_entity(gs.morphs.MJCF(file=env_tmp))
    scene.build()
    return scene, arm, arm_tmp, env_tmp


def _reset_pose_and_settle(scene, arm):
    scene.reset()
    q0 = torch.zeros(arm.n_dofs, dtype=torch.float32)
    for i, q in enumerate(ARM_LPOSE_QPOS):
        q0[i] = q
    q0[7] = FINGER_OPEN * 0.3
    q0[8] = FINGER_OPEN * 0.3
    arm.set_dofs_position(q0)
    for _ in range(N_SETTLE):
        arm.set_dofs_velocity(torch.zeros(arm.n_dofs))
        scene.step()


def _maybe_clamp_fingers(arm, v):
    q = arm.get_dofs_position()
    vel = v.tolist() if hasattr(v, "tolist") else list(v.detach().cpu().numpy())
    if q[7].item() >= FINGER_OPEN:
        vel[7] = 0.0
    if q[8].item() >= FINGER_OPEN:
        vel[8] = 0.0
    return gs.tensor(vel, requires_grad=True)


def _worker_mode(request_path):
    payload = json.loads(Path(request_path).read_text())
    velocities = payload["velocities"]
    loss_mode = payload["loss_mode"]
    target_qvel = payload["target_qvel"]
    scene_kind = payload["scene"]
    sample_every = int(payload["sample_every"])
    clamp_fingers = bool(payload["clamp_fingers"])
    control_weight = float(payload["control_weight"])
    smooth_weight = float(payload["smooth_weight"])

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    arm_tmp = env_tmp = None
    try:
        scene, arm, arm_tmp, env_tmp = _build_scene(scene_kind)
        _reset_pose_and_settle(scene, arm)

        target = gs.tensor(target_qvel)
        base_tensors = [gs.tensor(v, requires_grad=True) for v in velocities]
        used_tensors = list(base_tensors)
        snapshots = []

        for i, v in enumerate(base_tensors):
            used_v = _maybe_clamp_fingers(arm, v) if clamp_fingers else v
            used_tensors[i] = used_v
            arm.set_dofs_velocity(used_v)
            scene.step()
            if loss_mode == "trajectory" and ((i + 1) % sample_every == 0):
                snapshots.append(arm.get_dofs_velocity())

        final_qvel = arm.get_dofs_velocity()
        if loss_mode == "final":
            loss = (final_qvel - target).pow(2).sum()
        elif loss_mode == "trajectory":
            loss = sum((q - target).pow(2).sum() for q in snapshots)
        else:
            raise ValueError(f"unknown loss_mode={loss_mode!r}")

        if control_weight:
            loss = loss + control_weight * sum(v.pow(2).sum() for v in used_tensors)
        if smooth_weight and len(used_tensors) > 1:
            loss = loss + smooth_weight * sum(
                (used_tensors[i] - used_tensors[i - 1]).pow(2).sum()
                for i in range(1, len(used_tensors))
            )

        status = "ok"
        try:
            loss.backward()
        except Exception as e:
            status = f"bwd_error:{repr(e)[:160]}"

        grads = []
        grad_nan = 0
        grad_norm_sq = 0.0
        for v in used_tensors:
            if v.grad is None or torch.isnan(v.grad).any():
                grad_nan += 1
                g = torch.zeros(TOTAL_DOFS, dtype=torch.float32)
            else:
                g = v.grad.detach().float().cpu()
                grad_norm_sq += float((g * g).sum().item())
            grads.append(g.tolist())

        print("__TRAJOPT__" + json.dumps({
            "status": status,
            "loss": float(loss.item()),
            "grad": grads,
            "grad_nan": grad_nan,
            "grad_norm": math.sqrt(grad_norm_sq),
        }))
    finally:
        if arm_tmp:
            Path(arm_tmp).unlink(missing_ok=True)
        if env_tmp:
            Path(env_tmp).unlink(missing_ok=True)


def _tail(text, n=12):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _run_worker(script_path, velocities, args):
    with tempfile.NamedTemporaryFile(prefix="traj_opt_req_", suffix=".json", delete=False, mode="w") as f:
        json.dump({
            "velocities": velocities.detach().cpu().tolist(),
            "loss_mode": args.loss,
            "target_qvel": _make_target_qvel(args.target),
            "scene": args.scene,
            "sample_every": args.sample_every,
            "clamp_fingers": args.clamp_fingers,
            "control_weight": args.control_weight,
            "smooth_weight": args.smooth_weight,
        }, f)
        request_path = f.name
    try:
        cmd = [
            "conda", "run", "-n", "genesis", "--no-capture-output",
            "python", script_path, "--worker", request_path,
        ]
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=os.environ.copy(),
        )
    finally:
        Path(request_path).unlink(missing_ok=True)

    result = next(
        (json.loads(line[len("__TRAJOPT__"):])
         for line in p.stdout.splitlines() if line.startswith("__TRAJOPT__")),
        None,
    )
    if result is None:
        print(f"[worker-error] returncode={p.returncode}")
        if _tail(p.stdout):
            print("[worker-stdout-tail]")
            print(_tail(p.stdout))
        if _tail(p.stderr):
            print("[worker-stderr-tail]")
            print(_tail(p.stderr))
        raise RuntimeError("trajectory optimization worker did not return a result")
    return result


def _write_outputs(rows, velocities, csv_path, png_path, qvel_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "loss", "grad_norm", "grad_nan", "status"])
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(6.5, 4.0))
    plt.plot([r["iter"] for r in rows], [r["loss"] for r in rows], marker="o")
    plt.xlabel("Adam iteration")
    plt.ylabel("loss")
    plt.title("Trajectory optimization loss")
    plt.grid(True, alpha=0.3)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close()

    qvel_path.write_text(json.dumps({
        "shape": list(velocities.shape),
        "qvel": velocities.detach().cpu().tolist(),
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=None)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init", choices=("scripted", "zero"), default="scripted")
    parser.add_argument("--target", choices=("close", "approach", "lift", "zero"), default="close")
    parser.add_argument("--scene", choices=("rope", "no-rope"), default="rope")
    parser.add_argument("--loss", choices=("final", "trajectory"), default="final")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-vel", type=float, default=3.0)
    parser.add_argument("--control-weight", type=float, default=0.0)
    parser.add_argument("--smooth-weight", type=float, default=0.0)
    parser.add_argument("--no-clamp-fingers", action="store_false", dest="clamp_fingers")
    parser.set_defaults(clamp_fingers=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--csv", type=Path, default=OUT_CSV)
    parser.add_argument("--plot", type=Path, default=OUT_PNG)
    parser.add_argument("--qvel-out", type=Path, default=OUT_QVEL)
    args = parser.parse_args()

    if args.worker:
        _worker_mode(args.worker)
        return

    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")

    script_path = str(Path(__file__).resolve())
    velocities = _make_initial_velocities(args.horizon, args.init)
    velocities.requires_grad_(True)
    opt = torch.optim.Adam([velocities], lr=args.lr)
    rows = []

    print(
        f"Trajectory opt: H={args.horizon} iters={args.iters} lr={args.lr} "
        f"init={args.init} target={args.target} scene={args.scene} loss={args.loss}"
    )
    for it in range(args.iters + 1):
        result = _run_worker(script_path, velocities.detach(), args)
        row = {
            "iter": it,
            "loss": result["loss"],
            "grad_norm": result["grad_norm"],
            "grad_nan": result["grad_nan"],
            "status": result["status"],
        }
        rows.append(row)
        print(
            f"  iter={it:03d} loss={row['loss']:.6f} "
            f"grad_norm={row['grad_norm']:.6f} grad_nan={row['grad_nan']}/{args.horizon} "
            f"status={row['status']}"
        )
        if it == args.iters:
            break
        if result["status"] != "ok" or result["grad_nan"]:
            print("  stopping: worker returned non-ok status or NaN gradients")
            break

        opt.zero_grad(set_to_none=True)
        velocities.grad = torch.tensor(result["grad"], dtype=torch.float32)
        opt.step()
        with torch.no_grad():
            velocities.clamp_(-args.max_vel, args.max_vel)

    _write_outputs(rows, velocities.detach(), args.csv, args.plot, args.qvel_out)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.plot}")
    print(f"Wrote {args.qvel_out}")


if __name__ == "__main__":
    raise SystemExit(main())
