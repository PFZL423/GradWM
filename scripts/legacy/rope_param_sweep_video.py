"""Rope parameter sweep — produces one MP4 per config for visual comparison.

R001-R009: radius=0.010 (2cm diameter)
C001-C009: radius=0.005 (1cm diameter) — same param matrix, half the radius

Run with --ids R001,C003 to test a subset.
Outputs go to analysis/rope_sweep/. Index written to rope_sweep_index.txt.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

OUT_DIR = Path("analysis/rope_sweep")
INDEX_PATH = OUT_DIR / "rope_sweep_index.txt"

# N doubled throughout (60/70/80 replacing 30/35/40); seg_mass keeps total ~14g
_BASE_PARAMS = [
    {"desc": "standard",       "N_CABLE_SEG": 60, "CABLE_DAMPING": 2e-4, "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 2.33e-4},
    {"desc": "low-damp",       "N_CABLE_SEG": 60, "CABLE_DAMPING": 1e-5, "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 2.33e-4},
    {"desc": "high-damp",      "N_CABLE_SEG": 60, "CABLE_DAMPING": 5e-3, "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 2.33e-4},
    {"desc": "heavy",          "N_CABLE_SEG": 60, "CABLE_DAMPING": 2e-4, "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 1.0e-3},
    {"desc": "light",          "N_CABLE_SEG": 60, "CABLE_DAMPING": 2e-4, "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 5.0e-5},
    {"desc": "N50",            "N_CABLE_SEG": 50, "CABLE_DAMPING": 2e-4, "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 2.80e-4},
    {"desc": "N80",            "N_CABLE_SEG": 80, "CABLE_DAMPING": 2e-4, "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 1.75e-4},
    {"desc": "combined",       "N_CABLE_SEG": 70, "CABLE_DAMPING": 6e-4, "CABLE_ARMATURE": 8e-5, "CABLE_SEG_MASS": 2.00e-4},
    {"desc": "heavy-highdamp", "N_CABLE_SEG": 80, "CABLE_DAMPING": 5e-3, "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 1.0e-3},
]

# D-series: based on C009 (N=80, damp=5e-3, mass=1e-3, arm=2e-5, r=0.005)
# sweep N / mass / damp / armature one dimension at a time
_BASE_N    = 80
_BASE_DAMP = 5e-3
_BASE_ARM  = 2e-5
_BASE_MASS = 1.0e-3

_D_PARAMS = [
    # baseline
    {"desc": "base",        "N_CABLE_SEG": _BASE_N,  "CABLE_DAMPING": _BASE_DAMP, "CABLE_ARMATURE": _BASE_ARM,  "CABLE_SEG_MASS": _BASE_MASS},
    # N sweep (mass scales to keep total ~80g)
    {"desc": "N60",         "N_CABLE_SEG": 60,        "CABLE_DAMPING": _BASE_DAMP, "CABLE_ARMATURE": _BASE_ARM,  "CABLE_SEG_MASS": 1.33e-3},
    {"desc": "N100",        "N_CABLE_SEG": 100,       "CABLE_DAMPING": _BASE_DAMP, "CABLE_ARMATURE": _BASE_ARM,  "CABLE_SEG_MASS": 8.0e-4},
    {"desc": "N120",        "N_CABLE_SEG": 120,       "CABLE_DAMPING": _BASE_DAMP, "CABLE_ARMATURE": _BASE_ARM,  "CABLE_SEG_MASS": 6.67e-4},
    # mass sweep
    {"desc": "mass-half",   "N_CABLE_SEG": _BASE_N,  "CABLE_DAMPING": _BASE_DAMP, "CABLE_ARMATURE": _BASE_ARM,  "CABLE_SEG_MASS": 5.0e-4},
    {"desc": "mass-2x",     "N_CABLE_SEG": _BASE_N,  "CABLE_DAMPING": _BASE_DAMP, "CABLE_ARMATURE": _BASE_ARM,  "CABLE_SEG_MASS": 2.0e-3},
    # damp sweep
    {"desc": "damp-low",    "N_CABLE_SEG": _BASE_N,  "CABLE_DAMPING": 5e-4,       "CABLE_ARMATURE": _BASE_ARM,  "CABLE_SEG_MASS": _BASE_MASS},
    {"desc": "damp-high",   "N_CABLE_SEG": _BASE_N,  "CABLE_DAMPING": 2e-2,       "CABLE_ARMATURE": _BASE_ARM,  "CABLE_SEG_MASS": _BASE_MASS},
    # armature sweep
    {"desc": "arm-low",     "N_CABLE_SEG": _BASE_N,  "CABLE_DAMPING": _BASE_DAMP, "CABLE_ARMATURE": 1e-6,       "CABLE_SEG_MASS": _BASE_MASS},
    {"desc": "arm-high",    "N_CABLE_SEG": _BASE_N,  "CABLE_DAMPING": _BASE_DAMP, "CABLE_ARMATURE": 2e-4,       "CABLE_SEG_MASS": _BASE_MASS},
    {"desc": "arm-xhigh",   "N_CABLE_SEG": _BASE_N,  "CABLE_DAMPING": _BASE_DAMP, "CABLE_ARMATURE": 2e-3,       "CABLE_SEG_MASS": _BASE_MASS},
]

# E-series: fine-tune around D005 (mass-half=5e-4) and D008 (damp-high=2e-2)
# N=80, arm=2e-5 fixed; damp slightly below 2e-2 to reduce stiffness
_E_PARAMS = [
    # mass fine sweep around 5e-4
    {"desc": "mass-3e-4",        "N_CABLE_SEG": 80, "CABLE_DAMPING": 5e-3,  "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 3.0e-4},
    {"desc": "mass-5e-4",        "N_CABLE_SEG": 80, "CABLE_DAMPING": 5e-3,  "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 5.0e-4},
    {"desc": "mass-7e-4",        "N_CABLE_SEG": 80, "CABLE_DAMPING": 5e-3,  "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 7.0e-4},
    # damp fine sweep below 2e-2 (D008 slightly stiff)
    {"desc": "damp-8e-3",        "N_CABLE_SEG": 80, "CABLE_DAMPING": 8e-3,  "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 5.0e-4},
    {"desc": "damp-1e-2",        "N_CABLE_SEG": 80, "CABLE_DAMPING": 1e-2,  "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 5.0e-4},
    {"desc": "damp-2e-2",        "N_CABLE_SEG": 80, "CABLE_DAMPING": 2e-2,  "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 5.0e-4},
    # combined best guesses
    {"desc": "combo-a",          "N_CABLE_SEG": 80, "CABLE_DAMPING": 8e-3,  "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 3.0e-4},
    {"desc": "combo-b",          "N_CABLE_SEG": 80, "CABLE_DAMPING": 1e-2,  "CABLE_ARMATURE": 2e-5, "CABLE_SEG_MASS": 7.0e-4},
]

SWEEP_MATRIX = (
    [{"id": f"R{i+1:03d}", "CABLE_SEG_RADIUS": 0.010, "finger_open": 0.013, "cable_rest_z_bump": 0.0,   **p} for i, p in enumerate(_BASE_PARAMS)] +
    [{"id": f"C{i+1:03d}", "CABLE_SEG_RADIUS": 0.005, "finger_open": 0.008, "cable_rest_z_bump": 0.013, **p} for i, p in enumerate(_BASE_PARAMS)] +
    [{"id": f"D{i+1:03d}", "CABLE_SEG_RADIUS": 0.005, "finger_open": 0.008, "cable_rest_z_bump": 0.013, **p} for i, p in enumerate(_D_PARAMS)] +
    [{"id": f"E{i+1:03d}", "CABLE_SEG_RADIUS": 0.005, "finger_open": 0.008, "cable_rest_z_bump": 0.013, **p} for i, p in enumerate(_E_PARAMS)]
)


def _run_worker(entry: dict):
    import tempfile
    import numpy as np
    import imageio.v2 as imageio
    import torch
    import genesis as gs
    import grasp_scene as gs_mod
    from grasp_scene import (
        _enable_arm_contact_geoms, _grayscale_arm_geoms, _write_temp_mjcf,
        APPROACH_QVEL, CLOSE_QVEL, LIFT_QVEL,
        INITIAL_FINGER_OPEN, ARM_LPOSE_QPOS,
    )
    from make_arm_mjcf import make_arm_gripper_mjcf

    cable_rest_z_bump = entry.get("cable_rest_z_bump", 0.0)
    patch_keys = ["N_CABLE_SEG", "CABLE_DAMPING", "CABLE_ARMATURE", "CABLE_SEG_MASS", "CABLE_SEG_RADIUS"]
    saved = {k: getattr(gs_mod, k) for k in patch_keys}
    saved_z = gs_mod.CABLE_REST_Z
    try:
        for k in patch_keys:
            setattr(gs_mod, k, entry[k])
        gs_mod.CABLE_REST_Z = saved_z + cable_rest_z_bump
        bridge_xml = gs_mod._make_bridge_scene_mjcf()
    finally:
        for k, v in saved.items():
            setattr(gs_mod, k, v)
        gs_mod.CABLE_REST_Z = saved_z

    # finger_open controls both the body y-offset AND the slide range upper limit.
    # This ensures the finger inner face can actually reach the rope surface.
    finger_open = entry["finger_open"]
    finger_open_init = finger_open * 0.6
    arm_xml = _grayscale_arm_geoms(_enable_arm_contact_geoms(
        make_arm_gripper_mjcf(finger_open=finger_open, finger_range=(0.0, finger_open))
    ))
    arm_tmp = _write_temp_mjcf("sweep_arm_", arm_xml)
    bridge_tmp = _write_temp_mjcf("sweep_bridge_", bridge_xml)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=4, substeps_local=4, requires_grad=False),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))
    arm = scene.add_entity(gs.morphs.MJCF(file=arm_tmp))
    scene.add_entity(gs.morphs.MJCF(file=bridge_tmp))

    cam = scene.add_camera(
        res=(640, 480),
        pos=(0.33, 1.0, 0.22),
        lookat=(0.33, 0.0, 0.16),
        up=(0.0, 0.0, 1.0),
        fov=42,
    )
    scene.build()
    scene.reset()

    initial_qpos = torch.zeros(arm.n_dofs, dtype=torch.float32)
    for i, q in enumerate(ARM_LPOSE_QPOS):
        initial_qpos[i] = q
    initial_qpos[7] = finger_open_init
    initial_qpos[8] = finger_open_init
    arm.set_dofs_position(initial_qpos)

    frames = []

    def step_render(vel):
        arm.set_dofs_velocity(torch.tensor(vel, dtype=torch.float32))
        scene.step()
        rgb = cam.render()
        if isinstance(rgb, tuple):
            rgb = rgb[0]
        frames.append(np.asarray(rgb))

    def step_render_close():
        q = arm.get_dofs_position()
        vel = list(CLOSE_QVEL)
        if q[7].item() >= finger_open:
            vel[7] = 0.0
        if q[8].item() >= finger_open:
            vel[8] = 0.0
        arm.set_dofs_velocity(torch.tensor(vel, dtype=torch.float32))
        scene.step()
        rgb = cam.render()
        if isinstance(rgb, tuple):
            rgb = rgb[0]
        frames.append(np.asarray(rgb))

    def step_render_lift():
        vel = list(LIFT_QVEL)
        q = arm.get_dofs_position()
        # maintain grip: keep pushing fingers closed unless already at limit
        vel[7] = 0.0 if q[7].item() >= finger_open else 1.5
        vel[8] = 0.0 if q[8].item() >= finger_open else 1.5
        arm.set_dofs_velocity(torch.tensor(vel, dtype=torch.float32))
        scene.step()
        rgb = cam.render()
        if isinstance(rgb, tuple):
            rgb = rgb[0]
        frames.append(np.asarray(rgb))

    zero_vel = [0.0] * arm.n_dofs

    for _ in range(30):
        step_render(zero_vel)
    for _ in range(15):
        step_render(APPROACH_QVEL)
    for _ in range(20):
        step_render_close()
    for _ in range(60):
        step_render_lift()
    for _ in range(90):
        step_render(zero_vel)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vid_name = f"rope_{entry['id']}_{entry['desc']}.mp4"
    vid_path = OUT_DIR / vid_name
    with imageio.get_writer(str(vid_path), fps=50, codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)

    Path(arm_tmp).unlink(missing_ok=True)
    Path(bridge_tmp).unlink(missing_ok=True)

    print("__RESULT__" + json.dumps({
        "id": entry["id"], "desc": entry["desc"],
        "video": vid_name, "n_frames": len(frames), "status": "ok",
    }))


def _write_index(results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# rope_param_sweep_video index",
        "# id  desc  N  radius  damping  armature  seg_mass  video  status",
        "",
    ]
    id_to_entry = {e["id"]: e for e in SWEEP_MATRIX}
    for r in results:
        e = id_to_entry.get(r.get("id"), {})
        lines.append(
            f"{r.get('id','?'):<6} {r.get('desc','?'):<16} "
            f"N={e.get('N_CABLE_SEG','?'):<3} "
            f"r={e.get('CABLE_SEG_RADIUS','?'):<6} "
            f"damp={e.get('CABLE_DAMPING','?'):<8} "
            f"arm={e.get('CABLE_ARMATURE','?'):<8} "
            f"mass={e.get('CABLE_SEG_MASS','?'):<10} "
            f"{r.get('video','?'):<44} {r.get('status','?')}"
        )
    INDEX_PATH.write_text("\n".join(lines) + "\n")
    print(f"[sweep] index -> {INDEX_PATH}")


def main():
    script_path = str(Path(__file__).resolve())

    ids_filter = None
    for arg in sys.argv[1:]:
        if arg.startswith("--ids"):
            val = arg.split("=", 1)[-1] if "=" in arg else sys.argv[sys.argv.index(arg) + 1]
            ids_filter = set(val.split(","))

    matrix = [e for e in SWEEP_MATRIX if ids_filter is None or e["id"] in ids_filter]

    results = []
    for entry in matrix:
        t0 = time.time()
        print(f"[sweep] starting {entry['id']} ({entry['desc']}) ...")
        cmd = [
            "conda", "run", "-n", "genesis", "--no-capture-output",
            "python", script_path, "--worker", json.dumps(entry),
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            marker = "__RESULT__"
            line = next((l for l in p.stdout.split("\n") if l.startswith(marker)), None)
            if line:
                r = json.loads(line[len(marker):])
            else:
                stderr_snip = p.stderr.strip()[-200:] if p.stderr else ""
                r = {"id": entry["id"], "desc": entry["desc"],
                     "video": "N/A", "n_frames": 0,
                     "status": f"no_result stderr={stderr_snip}"}
        except subprocess.TimeoutExpired:
            r = {"id": entry["id"], "desc": entry["desc"],
                 "video": "N/A", "n_frames": 0, "status": "timeout"}

        elapsed = time.time() - t0
        print(f"[sweep] {entry['id']} {r['status']} ({elapsed:.0f}s)")
        results.append(r)

    _write_index(results)
    print("\n[sweep] all done.")
    for r in results:
        print(f"  {r['id']}  {r['desc']:<16}  {r['status']}  {r.get('video','')}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        idx = sys.argv.index("--worker")
        entry = json.loads(sys.argv[idx + 1])
        raise SystemExit(_run_worker(entry))
    else:
        raise SystemExit(main())
