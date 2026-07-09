"""Gradient analysis for rope grasp scene.

Runs forward+backward with per-step grad logging, produces:
  analysis/grad_analysis.png  — 3-panel figure (approach/close/lift per-step grad norm)
  analysis/grad_analysis.csv  — raw data

Can compare two parameter sets (new vs old) side-by-side.
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import genesis as gs

from grasp_scene import (
    _enable_arm_contact_geoms, _grayscale_arm_geoms, _write_temp_mjcf,
    APPROACH_QVEL, CLOSE_QVEL, LIFT_QVEL,
    ARM_LPOSE_QPOS, INITIAL_FINGER_OPEN, TARGET_QVEL,
    TABLE_X_LEFT, TABLE_X_RIGHT, TABLE_TOP_Z, CABLE_REST_Z,
)
import grasp_scene as gs_mod
from make_arm_mjcf import make_arm_gripper_mjcf

OUT_PNG = Path("analysis/grad_analysis.png")
OUT_CSV = Path("analysis/grad_analysis.csv")

N_SETTLE   = 30
N_APPROACH = 15
N_CLOSE    = 20
N_LIFT     = 25   # same as grasp_scene.py HORIZON breakdown

CONFIGS = [
    {
        "label": "new (N=80 r=0.005)",
        "N_CABLE_SEG": 80,
        "CABLE_SEG_RADIUS": 0.005,
        "CABLE_DAMPING": 9e-3,
        "CABLE_ARMATURE": 2e-5,
        "CABLE_SEG_MASS": 4e-4,
        "cable_rest_z_bump": 0.013,
        "finger_open": 0.008,
    },
    {
        "label": "old (N=30 r=0.010)",
        "N_CABLE_SEG": 30,
        "CABLE_SEG_RADIUS": 0.010,
        "CABLE_DAMPING": 2e-4,
        "CABLE_ARMATURE": 2e-5,
        "CABLE_SEG_MASS": 4.67e-4,
        "cable_rest_z_bump": 0.0,
        "finger_open": 0.013,
    },
]

PATCH_KEYS = ["N_CABLE_SEG", "CABLE_SEG_RADIUS", "CABLE_DAMPING", "CABLE_ARMATURE", "CABLE_SEG_MASS"]


def run_config(cfg: dict) -> dict:
    saved = {k: getattr(gs_mod, k) for k in PATCH_KEYS}
    saved_z = gs_mod.CABLE_REST_Z
    try:
        for k in PATCH_KEYS:
            setattr(gs_mod, k, cfg[k])
        gs_mod.CABLE_REST_Z = saved_z + cfg["cable_rest_z_bump"]
        bridge_xml = gs_mod._make_bridge_scene_mjcf()
    finally:
        for k, v in saved.items():
            setattr(gs_mod, k, v)
        gs_mod.CABLE_REST_Z = saved_z

    finger_open = cfg["finger_open"]
    arm_xml = _grayscale_arm_geoms(_enable_arm_contact_geoms(
        make_arm_gripper_mjcf(finger_open=finger_open, finger_range=(0.0, finger_open))
    ))
    arm_tmp = _write_temp_mjcf("grad_arm_", arm_xml)
    bridge_tmp = _write_temp_mjcf("grad_bridge_", bridge_xml)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=4, substeps_local=4, requires_grad=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))
    arm = scene.add_entity(gs.morphs.MJCF(file=arm_tmp))
    scene.add_entity(gs.morphs.MJCF(file=bridge_tmp))
    scene.build()
    scene.reset()

    # find grad cleanup method (same probe as grasp_scene.py)
    graph_cleanup = None
    for name in ("reset_grad", "reset_grad_state", "clear_grad", "zero_grad", "_reset_grad"):
        for target in (getattr(scene, "sim", None), scene):
            if target is None:
                continue
            method = getattr(target, name, None)
            if callable(method):
                graph_cleanup = method
                break
        if graph_cleanup is not None:
            break
    if graph_cleanup is None:
        graph_cleanup = lambda: None

    q0 = torch.zeros(arm.n_dofs, dtype=torch.float32)
    for i, q in enumerate(ARM_LPOSE_QPOS):
        q0[i] = q
    q0[7] = finger_open * 0.3
    q0[8] = finger_open * 0.3
    arm.set_dofs_position(q0)

    # settle (no grad needed)
    for _ in range(N_SETTLE):
        arm.set_dofs_velocity(torch.zeros(arm.n_dofs))
        scene.step()

    target = gs.tensor(TARGET_QVEL)
    grad_norms = {"approach": [], "close": [], "lift": []}
    nan_counts = {"approach": 0, "close": 0, "lift": 0}

    # Build full velocity program (approach + close + lift), requires_grad=True
    approach_vels = [gs.tensor(APPROACH_QVEL, requires_grad=True) for _ in range(N_APPROACH)]
    close_vels    = [gs.tensor(CLOSE_QVEL,    requires_grad=True) for _ in range(N_CLOSE)]
    lift_vels     = [gs.tensor(LIFT_QVEL,     requires_grad=True) for _ in range(N_LIFT)]
    all_vels = approach_vels + close_vels + lift_vels

    qvel_snapshots = []
    status = "ok"
    try:
        for i, v in enumerate(all_vels):
            # close phase: clamp finger velocity if at limit
            if N_APPROACH <= i < N_APPROACH + N_CLOSE:
                q = arm.get_dofs_position()
                vel_list = v.tolist() if hasattr(v, 'tolist') else list(v.detach().cpu().numpy())
                if q[7].item() >= finger_open: vel_list[7] = 0.0
                if q[8].item() >= finger_open: vel_list[8] = 0.0
                v = gs.tensor(vel_list, requires_grad=True)
                close_vels[i - N_APPROACH] = v
                all_vels[i] = v
            arm.set_dofs_velocity(v)
            scene.step()
            qvel_snapshots.append((i, arm.get_dofs_velocity()))
    except Exception as e:
        status = f"error:{repr(e)[:100]}"

    if status == "ok":
        loss = sum((q - target).pow(2).sum() for _, q in qvel_snapshots)
        try:
            loss.backward()
        except Exception as e:
            status = f"bwd_error:{repr(e)[:100]}"

    # collect per-step grad norms
    phase_map = (
        [("approach", v) for v in approach_vels] +
        [("close",    v) for v in close_vels] +
        [("lift",     v) for v in lift_vels]
    )
    for phase_name, v in phase_map:
        if v.grad is not None and not torch.isnan(v.grad).any():
            grad_norms[phase_name].append(float(v.grad.norm().item()))
        else:
            grad_norms[phase_name].append(float("nan"))
            nan_counts[phase_name] += 1

    graph_cleanup()

    Path(arm_tmp).unlink(missing_ok=True)
    Path(bridge_tmp).unlink(missing_ok=True)

    return {"label": cfg["label"], "grad_norms": grad_norms, "nan_counts": nan_counts}


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    results = []
    for cfg in CONFIGS:
        print(f"\n[grad_analysis] running: {cfg['label']}")
        t0 = time.time()
        r = run_config(cfg)
        print(f"[grad_analysis] done in {time.time()-t0:.1f}s  "
              f"NaN: approach={r['nan_counts']['approach']} "
              f"close={r['nan_counts']['close']} lift={r['nan_counts']['lift']}")
        results.append(r)

    # --- plot ---
    phases = ["approach", "close", "lift"]
    colors = ["C0", "C1"]
    phase_lengths = {"approach": N_APPROACH, "close": N_CLOSE, "lift": N_LIFT}

    fig, ax = plt.subplots(figsize=(12, 4.5))

    for r, color in zip(results, colors):
        all_steps = []
        all_grads = []
        offset = 0
        for phase in phases:
            norms = r["grad_norms"][phase]
            for i, g in enumerate(norms):
                all_steps.append(offset + i)
                all_grads.append(math.log10(g) if math.isfinite(g) and g > 0 else float("nan"))
            offset += len(norms)
        ax.plot(all_steps, all_grads, marker="o", markersize=4, linewidth=1.4,
                color=color, label=r["label"])

    # phase boundary lines
    ax.axvline(N_APPROACH - 0.5,              color="0.35", linestyle="--", linewidth=1.0, label="approach→close")
    ax.axvline(N_APPROACH + N_CLOSE - 0.5,    color="0.55", linestyle="--", linewidth=1.0, label="close→lift")

    # phase labels
    ax.text(N_APPROACH * 0.5,                          ax.get_ylim()[0] if ax.get_ylim()[0] > -999 else -1, "approach", ha="center", fontsize=9, color="0.4")
    ax.text(N_APPROACH + N_CLOSE * 0.5,                ax.get_ylim()[0] if ax.get_ylim()[0] > -999 else -1, "close",    ha="center", fontsize=9, color="0.4")
    ax.text(N_APPROACH + N_CLOSE + N_LIFT * 0.5,       ax.get_ylim()[0] if ax.get_ylim()[0] > -999 else -1, "lift",     ha="center", fontsize=9, color="0.4")

    ax.set_xlabel("step (global)")
    ax.set_ylabel("log10(grad norm)")
    ax.set_title("Genesis grad norm — full grasp trajectory (new vs old rope params)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n[grad_analysis] plot -> {OUT_PNG}")

    # --- csv ---
    rows = ["config,phase,step,grad_norm"]
    for r in results:
        for phase in phases:
            for step, g in enumerate(r["grad_norms"][phase]):
                rows.append(f"{r['label']},{phase},{step},{g:.6g}")
    OUT_CSV.write_text("\n".join(rows) + "\n")
    print(f"[grad_analysis] csv  -> {OUT_CSV}")


if __name__ == "__main__":
    raise SystemExit(main())
