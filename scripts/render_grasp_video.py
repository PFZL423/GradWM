"""Render the scripted grasp policy to mp4 from a side camera (+Y looking
at the X-Z plane). Reuses geometry + policy from grasp_scene.py.

No requires_grad — pure forward rollout for visualization.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import imageio.v2 as imageio
import numpy as np
import torch

import genesis as gs

from grasp_scene import (
    _make_bridge_scene_mjcf,
    _enable_arm_contact_geoms,
    _grayscale_arm_geoms,
    _write_temp_mjcf,
    APPROACH_QVEL, CLOSE_QVEL, LIFT_QVEL,
    INITIAL_FINGER_OPEN, ARM_LPOSE_QPOS,
    TABLE_X_LEFT, TABLE_X_RIGHT, TABLE_TOP_Z, CABLE_REST_Z,
)
from make_arm_mjcf import make_arm_gripper_mjcf

OUT_PATH = Path("analysis/grasp_phase1.mp4")
RES = (640, 480)
FPS = 20
N_SETTLE = 30   # let cable settle on tables before recording

def main():
    arm_xml = _grayscale_arm_geoms(_enable_arm_contact_geoms(make_arm_gripper_mjcf()))
    arm_tmp = _write_temp_mjcf("render_grasp_arm_", arm_xml)
    bridge_tmp = _write_temp_mjcf("render_grasp_bridge_", _make_bridge_scene_mjcf())

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=4, substeps_local=4, requires_grad=False),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))
    arm = scene.add_entity(gs.morphs.MJCF(file=arm_tmp))
    bridge = scene.add_entity(gs.morphs.MJCF(file=bridge_tmp))

    # Side camera — both tables + arm in frame, slight elevation so cable lift
    # is visually obvious. Scene center now at x≈0.33 (arm palm + cable mid).
    cam = scene.add_camera(
        res=RES,
        pos=(0.33, 1.0, 0.22),
        lookat=(0.33, 0.0, 0.16),
        up=(0.0, 0.0, 1.0),
        fov=42,
    )

    scene.build()
    scene.reset()

    # Initial pose: L-pose + open fingers
    initial_qpos = torch.zeros(arm.n_dofs, dtype=torch.float32)
    for i, q in enumerate(ARM_LPOSE_QPOS):
        initial_qpos[i] = q
    initial_qpos[7] = INITIAL_FINGER_OPEN
    initial_qpos[8] = INITIAL_FINGER_OPEN
    arm.set_dofs_position(initial_qpos)

    frames: list[np.ndarray] = []

    def hold_zero_velocity():
        arm.set_dofs_velocity(torch.zeros(arm.n_dofs, dtype=torch.float32))

    def step_and_render():
        scene.step()
        rgb = cam.render()
        # Genesis returns either a single ndarray or a tuple — normalize
        if isinstance(rgb, tuple):
            rgb = rgb[0]
        frames.append(np.asarray(rgb))

    # 1) Settle phase — cable falls onto tables, arm holds L-pose
    print(f"[render] settle ({N_SETTLE} steps)")
    for _ in range(N_SETTLE):
        hold_zero_velocity()
        step_and_render()

    # 2) Approach (15 steps) — arm idle in L-pose, fingers open
    print(f"[render] approach (15 steps)")
    for _ in range(15):
        arm.set_dofs_velocity(torch.tensor(APPROACH_QVEL, dtype=torch.float32))
        step_and_render()

    # 3) Close (20 steps) — fingers wrap cable
    print(f"[render] close (20 steps)")
    for _ in range(20):
        arm.set_dofs_velocity(torch.tensor(CLOSE_QVEL, dtype=torch.float32))
        step_and_render()

    # 4) Lift (25 steps) — J2/J4/J6 reverse, palm rises
    print(f"[render] lift (25 steps)")
    for _ in range(25):
        arm.set_dofs_velocity(torch.tensor(LIFT_QVEL, dtype=torch.float32))
        step_and_render()

    # Write mp4
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[render] writing {len(frames)} frames @ {FPS} fps -> {OUT_PATH}")
    with imageio.get_writer(str(OUT_PATH), fps=FPS, codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)
    print(f"[render] done. duration ~{len(frames)/FPS:.1f}s, size={OUT_PATH.stat().st_size/1024:.1f} KB")

if __name__ == "__main__":
    raise SystemExit(main())
