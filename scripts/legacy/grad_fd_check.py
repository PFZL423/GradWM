"""Finite-difference gradient check for Genesis rope grasp.

For each checked step, compares analytic gradient (backward) vs numerical
gradient (finite difference). Reports cosine similarity.
cos≈+1 = agree, cos≈0 = orthogonal, cos<0 = wrong direction.

Each step runs in a subprocess (Genesis can't do multiple backward in one process).
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
    """Keep Genesis/Quadrants/numba caches out of read-only home paths."""
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

import torch
import genesis as gs

from grasp_scene import (
    _enable_arm_contact_geoms, _grayscale_arm_geoms, _write_temp_mjcf,
    APPROACH_QVEL, CLOSE_QVEL, TARGET_QVEL, ARM_LPOSE_QPOS,
)
import grasp_scene as gs_mod
from make_arm_mjcf import make_arm_gripper_mjcf

CABLE_PARAMS = {
    "N_CABLE_SEG": 80, "CABLE_SEG_RADIUS": 0.005,
    "CABLE_DAMPING": 9e-3, "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 4e-4,
}
CABLE_REST_Z_BUMP = 0.013
FINGER_OPEN = 0.008
PATCH_KEYS  = list(CABLE_PARAMS.keys())
N_SETTLE    = 30
N_APPROACH  = 15
EPS         = 1e-4
CHECK_STEPS = [7, 15, 18]  # approach stable, close onset, close settled
DEFAULT_CSV = Path("analysis/grad_fd_check.csv")


def build_scene():
    saved   = {k: getattr(gs_mod, k) for k in PATCH_KEYS}
    saved_z = gs_mod.CABLE_REST_Z
    try:
        for k, v in CABLE_PARAMS.items():
            setattr(gs_mod, k, v)
        gs_mod.CABLE_REST_Z = saved_z + CABLE_REST_Z_BUMP
        bridge_xml = gs_mod._make_bridge_scene_mjcf()
    finally:
        for k, v in saved.items():
            setattr(gs_mod, k, v)
        gs_mod.CABLE_REST_Z = saved_z

    arm_xml    = _grayscale_arm_geoms(_enable_arm_contact_geoms(
        make_arm_gripper_mjcf(finger_open=FINGER_OPEN, finger_range=(0.0, FINGER_OPEN))
    ))
    arm_tmp    = _write_temp_mjcf("fd_arm_",    arm_xml)
    bridge_tmp = _write_temp_mjcf("fd_bridge_", bridge_xml)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=4, substeps_local=4, requires_grad=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))
    arm = scene.add_entity(gs.morphs.MJCF(file=arm_tmp))
    scene.add_entity(gs.morphs.MJCF(file=bridge_tmp))
    scene.build()
    return scene, arm, arm_tmp, bridge_tmp


def _reset_arm(scene, arm):
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


def rollout(scene, arm, vel_list, requires_grad_idx=None):
    """Roll out vel_list. If requires_grad_idx set, that v gets requires_grad=True."""
    target = gs.tensor(TARGET_QVEL)
    _reset_arm(scene, arm)
    v_tensors = []
    for i, v in enumerate(vel_list):
        rg = (i == requires_grad_idx)
        v_tensors.append(gs.tensor(v, requires_grad=rg))
    snapshots = []
    for v in v_tensors:
        arm.set_dofs_velocity(v)
        scene.step()
        snapshots.append(arm.get_dofs_velocity())
    loss = sum((q - target).pow(2).sum() for q in snapshots)
    return loss, v_tensors


def fd_check(scene, arm, check_step, graph_cleanup, eps):
    vels_nom = []
    for i in range(check_step + 1):
        vels_nom.append(list(APPROACH_QVEL) if i < N_APPROACH else list(CLOSE_QVEL))

    # analytic
    loss, v_tensors = rollout(scene, arm, vels_nom, requires_grad_idx=check_step)
    loss.backward()
    analytic_grad = v_tensors[check_step].grad.clone()
    graph_cleanup()

    # numerical
    target = gs.tensor(TARGET_QVEL)
    numerical_grad = torch.zeros(arm.n_dofs)
    for dof in range(arm.n_dofs):
        vp = [list(v) for v in vels_nom]; vp[check_step][dof] += eps
        vm = [list(v) for v in vels_nom]; vm[check_step][dof] -= eps

        loss_p, _ = rollout(scene, arm, vp)
        lp = loss_p.item()
        loss_m, _ = rollout(scene, arm, vm)
        lm = loss_m.item()
        numerical_grad[dof] = (lp - lm) / (2 * eps)

    ag  = analytic_grad.float()
    ng  = numerical_grad.float()
    cos = torch.nn.functional.cosine_similarity(ag.unsqueeze(0), ng.unsqueeze(0)).item()
    return cos, ag.norm().item(), ng.norm().item()


def worker_mode(check_step, eps):
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = arm = None
    arm_tmp = bridge_tmp = None
    try:
        scene, arm, arm_tmp, bridge_tmp = build_scene()

        graph_cleanup = None
        for name in ("reset_grad", "reset_grad_state", "clear_grad", "zero_grad", "_reset_grad"):
            for obj in (getattr(scene, "sim", None), scene):
                if obj and callable(getattr(obj, name, None)):
                    graph_cleanup = getattr(obj, name)
                    break
            if graph_cleanup:
                break
        if graph_cleanup is None:
            graph_cleanup = lambda: None

        cos, ag, ng = fd_check(scene, arm, check_step, graph_cleanup, eps)
        print("__FDRESULT__" + json.dumps({
            "step": check_step,
            "eps": eps,
            "cos": cos,
            "ag_norm": ag,
            "ng_norm": ng,
        }))
    finally:
        if arm_tmp:
            Path(arm_tmp).unlink(missing_ok=True)
        if bridge_tmp:
            Path(bridge_tmp).unlink(missing_ok=True)


def _parse_steps(raw):
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def _tail(text, n=12):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _tag(cos):
    if cos > 0.9:
        return "GOOD"
    if cos > 0.5:
        return "OK"
    if cos > 0.1:
        return "WEAK"
    return "BAD"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--steps", default=",".join(str(s) for s in CHECK_STEPS))
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    if args.worker is not None:
        worker_mode(args.worker, args.eps)
        return

    script = str(Path(__file__).resolve())
    check_steps = _parse_steps(args.steps)
    print(f"Finite-difference gradient check  eps={args.eps}")
    print(f"cos≈+1 = agree | cos≈0 = orthogonal | cos<0 = wrong direction\n")

    results = []
    for step in check_steps:
        cmd = ["conda", "run", "-n", "genesis", "--no-capture-output",
               "python", script, "--worker", str(step), "--eps", str(args.eps)]
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            env=os.environ.copy(),
        )
        r = next((json.loads(l[len("__FDRESULT__"):])
                  for l in p.stdout.split("\n") if l.startswith("__FDRESULT__")),
                 None)
        if r is None:
            print(f"[worker-error] step={step} returncode={p.returncode}")
            if _tail(p.stdout):
                print("[worker-stdout-tail]")
                print(_tail(p.stdout))
            if _tail(p.stderr):
                print("[worker-stderr-tail]")
                print(_tail(p.stderr))
            r = {"step": step, "eps": args.eps, "cos": float("nan"),
                 "ag_norm": float("nan"), "ng_norm": float("nan")}
        cos   = r["cos"]
        phase = "approach" if step < N_APPROACH else "close-onset"
        tag   = _tag(cos)
        print(f"  step={step:2d} ({phase:12s})  cos={cos:+.4f}  "
              f"|analytic|={r['ag_norm']:.4f}  |numerical|={r['ng_norm']:.4f}  [{tag}]")
        results.append({
            "step": step,
            "phase": phase,
            "eps": args.eps,
            "cos": cos,
            "analytic_norm": r["ag_norm"],
            "numerical_norm": r["ng_norm"],
            "tag": tag,
        })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "phase", "eps", "cos", "analytic_norm", "numerical_norm", "tag"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSummary:")
    for r in results:
        print(f"  step={r['step']:2d} ({r['phase']:12s})  cos={r['cos']:+.4f}  [{r['tag']}]")
    print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    raise SystemExit(main())
