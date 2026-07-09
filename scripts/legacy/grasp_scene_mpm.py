"""Grasp scene variant: cable is an MPM Elastic body (E=200, soft enough that
gravity sag is visible) instead of a 16-segment ball-joint rigid chain.

Same arm + same scripted approach/close/lift policy, same camera. Only the
cable physics + table geometry change. Tables are pulled closer (gap 0.20m
vs the original 0.30m) so the soft MPM cable doesn't slump entirely off the
bridge.

Caveats from rope_solver_probe sweep (memory genesis_pbd_fem_grad_stubs):
  - MPM is the only Genesis 1.1.1 solver with working backward + thin-object
    contact path. PBD/FEM are stub-only.
  - enable_CPIC=True helps coupling on thin objects but is incompatible with
    requires_grad=True. So we run without CPIC and accept that arm-MPM
    contact may be weaker than visually expected.
  - grid_density=128 is the sweet spot — 64 gives ~99 particles (no
    deformation), 200 causes backward NaN.
  - E=200 makes the body slump like jelly under gravity; bring tables closer
    to keep the cable approximately on top.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import imageio.v2 as imageio
import numpy as np
import torch
import genesis as gs

from make_arm_mjcf import TOTAL_DOFS, make_arm_gripper_mjcf

SCRIPT_NAME = Path(__file__).name
LOG_PATH = Path("logs/grasp_grad_phase1_mpm.log")
PLOT_PATH = Path("analysis/grad_norm_phase1_mpm.png")
VIDEO_PATH = Path("analysis/grasp_phase1_mpm.mp4")

CONTACT_GEOMS = {
    "L4_capsule",
    "L5_capsule",
    "L6_capsule",
    "L7_capsule",
    "palm_box",
    "finger_left_box",
    "finger_right_box",
}

ARM_LPOSE_QPOS = (0.0, 0.5, 0.0, 0.94, 0.0, 1.60, 0.0)

# Tables — closer than original (gap 0.20m vs 0.30m) so soft MPM cable
# doesn't completely slump between them.
TABLE_X_LEFT = 0.23
TABLE_X_RIGHT = 0.43
TABLE_TOP_Z = 0.14
TABLE_TOP_HALF = 0.04
TABLE_TOP_HALF_Y = 0.05
TABLE_TOP_THICK = 0.01
TABLE_LEG_HALF = 0.025

# MPM cable — soft enough to deform under gravity, thin cross-section.
ROPE_LEN = 0.30
ROPE_CROSS = 0.012
ROPE_REST_Z = TABLE_TOP_Z + TABLE_TOP_THICK + ROPE_CROSS * 0.5 + 0.002
ROPE_CENTER_X = (TABLE_X_LEFT + TABLE_X_RIGHT) * 0.5  # 0.33

INITIAL_FINGER_OPEN = 0.005

# Drive program — same shape as the rigid version but a touch gentler since
# MPM contact response is softer.
APPROACH_QVEL = [0.0] * 9
CLOSE_QVEL    = [0.0]*7 + [1.5, 1.5]
LIFT_QVEL     = [0.0, -2.0, 0.0, -2.0, 0.0, -2.0, 0.0, 0.0, 0.0]
TARGET_QVEL   = [0.0]*7 + [1.5, 1.5]
N_APPROACH = 8    # short settle so cable doesn't fully slump before fingers close
N_CLOSE = 25
N_LIFT = 35
HORIZON = N_APPROACH + N_CLOSE + N_LIFT

FPS = 20


def gpu_mem_mb():
    try:
        pid = os.getpid()
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if int(parts[0]) == pid:
                return float(parts[1])
    except Exception:
        pass
    return None


def fmt_mb(v):
    return "None" if v is None else f"{v:.1f}"


def find_solver_state(scene_state, type_name: str):
    for s in scene_state.solvers_state:
        if s is None or type(s).__name__ != type_name:
            continue
        pos = getattr(s, "pos", None)
        if pos is None or (len(pos.shape) >= 2 and int(pos.shape[1]) > 0):
            return s
    return None


def _make_tables_only_mjcf() -> str:
    """Two static box tables with NO cable — cable is added as a separate
    MPM entity in main()."""
    leg_top_z = TABLE_TOP_Z - TABLE_TOP_THICK
    leg_half_z = leg_top_z * 0.5
    return f"""<mujoco model="tables_only">
    <worldbody>
        <geom name="table_L_top" type="box" pos="{TABLE_X_LEFT} 0 {TABLE_TOP_Z}"
              size="{TABLE_TOP_HALF} {TABLE_TOP_HALF_Y} {TABLE_TOP_THICK}"
              rgba="0.55 0.40 0.25 1" contype="1" conaffinity="1"/>
        <geom name="table_L_leg" type="box" pos="{TABLE_X_LEFT} 0 {leg_half_z}"
              size="{TABLE_LEG_HALF} {TABLE_LEG_HALF} {leg_half_z}"
              rgba="0.55 0.40 0.25 1" contype="0" conaffinity="0"/>
        <geom name="table_R_top" type="box" pos="{TABLE_X_RIGHT} 0 {TABLE_TOP_Z}"
              size="{TABLE_TOP_HALF} {TABLE_TOP_HALF_Y} {TABLE_TOP_THICK}"
              rgba="0.55 0.40 0.25 1" contype="1" conaffinity="1"/>
        <geom name="table_R_leg" type="box" pos="{TABLE_X_RIGHT} 0 {leg_half_z}"
              size="{TABLE_LEG_HALF} {TABLE_LEG_HALF} {leg_half_z}"
              rgba="0.55 0.40 0.25 1" contype="0" conaffinity="0"/>
    </worldbody>
</mujoco>
"""


def _enable_arm_contact_geoms(arm_mjcf: str) -> str:
    root = ET.fromstring(arm_mjcf)
    enabled = set()
    for geom in root.iter("geom"):
        n = geom.get("name")
        if n in CONTACT_GEOMS:
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            enabled.add(n)
    return ET.tostring(root, encoding="unicode") + "\n"


def _grayscale_arm_geoms(arm_mjcf: str) -> str:
    root = ET.fromstring(arm_mjcf)
    for geom in root.iter("geom"):
        n = geom.get("name") or ""
        if n == "tcp_marker":
            continue
        if n.startswith("L") and n.endswith("_capsule"):
            geom.set("rgba", "0.72 0.72 0.72 1")
        elif n == "palm_box":
            geom.set("rgba", "0.45 0.45 0.45 1")
        elif n == "arm_base_box":
            geom.set("rgba", "0.18 0.18 0.18 1")
        elif n in ("finger_left_box", "finger_right_box"):
            geom.set("rgba", "0.55 0.55 0.55 1")
    return ET.tostring(root, encoding="unicode") + "\n"


def _write_temp_mjcf(prefix: str, text: str) -> str:
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".xml",
                                     delete=False, mode="w") as tmp:
        tmp.write(text)
        return tmp.name


def main():
    arm_xml = _grayscale_arm_geoms(_enable_arm_contact_geoms(make_arm_gripper_mjcf()))
    arm_tmp = _write_temp_mjcf("grasp_mpm_arm_", arm_xml)
    tables_tmp = _write_temp_mjcf("grasp_mpm_tables_", _make_tables_only_mjcf())
    print(f"[grasp-mpm] arm mjcf: {arm_tmp}")
    print(f"[grasp-mpm] tables mjcf: {tables_tmp}")
    print(f"[grasp-mpm] tables x={TABLE_X_LEFT}/{TABLE_X_RIGHT} top z={TABLE_TOP_Z}; rope z={ROPE_REST_Z:.3f}")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3, substeps=10, requires_grad=False,   # CPIC requires this off
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.1, -0.10, -0.05),
            upper_bound=(0.7, 0.10, 0.50),
            grid_density=128,
            enable_CPIC=True,                            # critical for thin-finger contact
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(pos=(0.0, 0.0, -0.001)),
        surface=gs.surfaces.Default(color=(0.18, 0.20, 0.25, 1.0)),
    )
    arm = scene.add_entity(gs.morphs.MJCF(file=arm_tmp))
    tables = scene.add_entity(gs.morphs.MJCF(file=tables_tmp))

    rope = scene.add_entity(
        material=gs.materials.MPM.Elastic(rho=100, E=200),  # very soft, ropy slump
        morph=gs.morphs.Box(
            pos=(ROPE_CENTER_X, 0.0, ROPE_REST_Z),
            size=(ROPE_LEN, ROPE_CROSS, ROPE_CROSS),
        ),
        surface=gs.surfaces.Default(color=(0.86, 0.62, 0.20, 1.0)),
        vis_mode="particle",
    )

    cam = scene.add_camera(
        res=(640, 480),
        pos=(ROPE_CENTER_X, 1.0, 0.22),
        lookat=(ROPE_CENTER_X, 0.0, 0.16),
        up=(0.0, 0.0, 1.0),
        fov=42,
    )

    scene.build()
    scene.reset()

    print(f"[grasp-mpm] arm n_dofs={arm.n_dofs}")
    print(f"[grasp-mpm] rope n_particles={getattr(rope, 'n_particles', '?')}")

    initial_qpos = torch.zeros(arm.n_dofs, dtype=torch.float32)
    for i, q in enumerate(ARM_LPOSE_QPOS):
        initial_qpos[i] = q
    initial_qpos[7] = INITIAL_FINGER_OPEN
    initial_qpos[8] = INITIAL_FINGER_OPEN
    arm.set_dofs_position(initial_qpos)

    # ---------- Forward + record video ----------
    qvels = ([APPROACH_QVEL]*N_APPROACH + [CLOSE_QVEL]*N_CLOSE
             + [LIFT_QVEL]*N_LIFT)
    # CPIC requires requires_grad=False, so we run forward-only here. Backward
    # path was validated separately in scripts/rope_solver_probe/probe_mpm_elastic.py.
    v_tensors = [gs.tensor(qv, requires_grad=False) for qv in qvels]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mem_samples = [gpu_mem_mb()]

    qvel_snapshots = []
    rope_z_log = []
    frames = []
    status = "ok"
    t0 = time.time()
    try:
        for i in range(HORIZON):
            arm.set_dofs_velocity(v_tensors[i])
            scene.step()
            if (i + 1) % 5 == 0:
                qvel_snapshots.append(arm.get_dofs_velocity())
            # render every step (HORIZON ~90 frames @ 20fps = 4.5s video)
            rgb = cam.render()
            if isinstance(rgb, tuple):
                rgb = rgb[0]
            frames.append(np.asarray(rgb))
            # track rope mean z for diagnostic
            ss = scene.get_state()
            mpm_state = find_solver_state(ss, "MPMSolverState")
            if mpm_state is not None:
                rope_z_log.append(float(mpm_state.pos[..., 2].mean().detach().cpu().item()))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except gs.GenesisException as e:
        status = f"genesis_err:{str(e)[:120]}"
    except Exception as e:
        status = f"err:{repr(e)[:120]}"
    t_fwd = time.time() - t0
    fps_fwd = HORIZON / t_fwd if t_fwd > 0 else math.nan
    mem_samples.append(gpu_mem_mb())

    # ---------- Backward skipped — CPIC + requires_grad incompatible ----------
    grad_norms = [float("nan")] * HORIZON
    nan_count = HORIZON
    bwd_status = "skipped (CPIC mode, see rope_solver_probe for backward)"
    t_bwd = 0.0
    loss_v = float("nan")
    mem_samples.append(gpu_mem_mb())
    peak_proc = max([m for m in mem_samples if m is not None], default=None)
    peak_torch = (torch.cuda.max_memory_allocated() / 1024 / 1024
                  if torch.cuda.is_available() else None)

    # ---------- Render video ----------
    VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(VIDEO_PATH), fps=FPS, codec="libx264",
                            quality=8) as w:
        for f in frames:
            w.append_data(f)

    # ---------- Plot grad-norm sample steps ----------
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_steps = [i for i, g in enumerate(grad_norms)
                    if math.isfinite(g) and g > 1e-9]
    sample_y = [math.log10(grad_norms[i]) for i in sample_steps]
    plt.figure(figsize=(8, 4.5))
    if sample_steps:
        plt.plot(sample_steps, sample_y, marker="o", markersize=6, linewidth=1.6)
    plt.axvline(N_APPROACH - 0.5, color="0.35", ls="--", lw=1.0,
                label="approach→close")
    plt.axvline(N_APPROACH + N_CLOSE - 0.5, color="0.35", ls="--", lw=1.0,
                label="close→lift")
    plt.xlabel("step index (loss-sample steps only)")
    plt.ylabel("log10(grad_norm)")
    plt.title("Grad norm vs t — MPM Elastic E=200 cable + handwritten arm")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close()

    # ---------- Log ----------
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write("\n".join([
            f"timestamp={datetime.now().isoformat(timespec='seconds')}",
            f"arm n_dofs={arm.n_dofs}",
            f"rope n_particles={getattr(rope, 'n_particles', '?')}",
            f"forward status={status} fps={fps_fwd:.2f} loss={loss_v:.5f}",
            f"backward status={bwd_status} t={t_bwd:.2f}s grad_nan={nan_count}/{HORIZON}",
            f"peak_mem nvidia-smi={fmt_mb(peak_proc)} torch={fmt_mb(peak_torch)}",
            f"rope_z trajectory: {rope_z_log[::10]}",
            "",
        ]) + "\n")

    print(f"[summary] forward {status} fps={fps_fwd:.1f}")
    print(f"[summary] backward {bwd_status} t={t_bwd:.2f}s grad_nan={nan_count}/{HORIZON}")
    print(f"[summary] mem peak nvidia-smi={fmt_mb(peak_proc)} torch={fmt_mb(peak_torch)}")
    print(f"[summary] loss={loss_v:.5f}, grad finite/total={len(sample_steps)}/{HORIZON}")
    print(f"[summary] video: {VIDEO_PATH}")
    print(f"[summary] plot: {PLOT_PATH}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
