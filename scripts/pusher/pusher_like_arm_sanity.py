"""Forward-only sanity for a Genesis-native Pusher-like arm task.

This is the bridge task between the primitive free-body box push and the
future ARX arm setup: action is a multi-DOF arm qvel vector, while the task
loss is still a simple object-to-target distance.
"""
import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import genesis as gs
from genesis.utils.misc import qd_to_numpy

from make_arm_mjcf import TOTAL_DOFS, make_arm_gripper_mjcf


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
INITIAL_FINGER_OPEN = 0.005

DT = 2e-3
TABLE_POS = (0.34, 0.0, 0.0475)
TABLE_SIZE = (0.26, 0.16, 0.095)
TABLE_TOP_Z = TABLE_POS[2] + TABLE_SIZE[2] * 0.5
OBJECT_SIZE = (0.035, 0.035, 0.035)
OBJECT_POS = (0.335, 0.0, TABLE_TOP_Z + OBJECT_SIZE[2] * 0.5)
TARGET_POS = (0.43, 0.0, OBJECT_POS[2])


def _enable_arm_contact_geoms(arm_mjcf: str) -> str:
    root = ET.fromstring(arm_mjcf)
    enabled = set()
    for geom in root.iter("geom"):
        name = geom.get("name")
        if name in CONTACT_GEOMS:
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            enabled.add(name)
    missing = CONTACT_GEOMS - enabled
    if missing:
        raise ValueError(f"missing expected contact geoms: {sorted(missing)}")
    return ET.tostring(root, encoding="unicode") + "\n"


def _write_temp_mjcf(prefix: str, mjcf_text: str) -> str:
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".xml", delete=False, mode="w") as tmp:
        tmp.write(mjcf_text)
        return tmp.name


def _contact_count(scene) -> int:
    n_contacts = qd_to_numpy(scene.rigid_solver.collider._collider_state.n_contacts)
    return int(n_contacts.reshape(-1)[0])


def _entity_pos_np(ent) -> np.ndarray:
    return ent.get_state().pos.detach().cpu().numpy().reshape(-1, 3)[0].astype(float)


def _make_scene():
    arm_xml = _enable_arm_contact_geoms(make_arm_gripper_mjcf())
    arm_tmp = _write_temp_mjcf("pusher_like_arm_", arm_xml)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=4, substeps_local=4, requires_grad=False),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))
    scene.add_entity(
        gs.morphs.Box(pos=TABLE_POS, size=TABLE_SIZE, fixed=True),
        surface=gs.surfaces.Default(color=(0.40, 0.36, 0.30, 1.0)),
    )
    arm = scene.add_entity(gs.morphs.MJCF(file=arm_tmp))
    obj = scene.add_entity(
        gs.morphs.Box(pos=OBJECT_POS, size=OBJECT_SIZE),
        surface=gs.surfaces.Default(color=(0.86, 0.45, 0.22, 1.0)),
    )
    scene.build()
    Path(arm_tmp).unlink(missing_ok=True)
    return scene, arm, obj


def _set_initial_arm_pose(arm):
    qpos = torch.zeros(arm.n_dofs, dtype=torch.float32)
    for i, q in enumerate(ARM_LPOSE_QPOS):
        qpos[i] = q
    qpos[7] = INITIAL_FINGER_OPEN
    qpos[8] = INITIAL_FINGER_OPEN
    arm.set_dofs_position(qpos)


def _zeros():
    return torch.zeros(TOTAL_DOFS, dtype=torch.float32)


def _close_qvel(speed: float):
    qvel = _zeros()
    qvel[7] = speed
    qvel[8] = speed
    return qvel


def _qvel_from_dofs(name: str, dofs: dict[int, float]):
    qvel = _zeros()
    for dof, value in dofs.items():
        qvel[dof] = value
    return name, qvel


def _candidate_push_qvels(speed: float):
    return [
        _qvel_from_dofs("hold", {}),
        _qvel_from_dofs("j1+", {0: speed}),
        _qvel_from_dofs("j1-", {0: -speed}),
        _qvel_from_dofs("j2+", {1: speed}),
        _qvel_from_dofs("j2-", {1: -speed}),
        _qvel_from_dofs("j4+", {3: speed}),
        _qvel_from_dofs("j4-", {3: -speed}),
        _qvel_from_dofs("j6+", {5: speed}),
        _qvel_from_dofs("j6-", {5: -speed}),
        _qvel_from_dofs("j2j4j6+", {1: speed, 3: speed, 5: speed}),
        _qvel_from_dofs("j2j4j6-", {1: -speed, 3: -speed, 5: -speed}),
        _qvel_from_dofs("reach_mix_a", {1: -speed, 3: speed, 5: -speed}),
        _qvel_from_dofs("reach_mix_b", {1: speed, 3: -speed, 5: speed}),
    ]


def _loss_to_target(pos: np.ndarray) -> float:
    delta = pos - np.asarray(TARGET_POS, dtype=float)
    return float(np.dot(delta, delta))


def _rollout(scene, arm, obj, push_qvel, settle_steps, close_steps, push_steps, close_speed):
    scene.reset()
    _set_initial_arm_pose(arm)
    initial_pos = _entity_pos_np(obj)
    initial_loss = _loss_to_target(initial_pos)

    contacts = []
    first_total_contact_step = None
    max_total_contact_count = 0

    def step(qvel):
        nonlocal first_total_contact_step, max_total_contact_count
        arm.set_dofs_velocity(qvel)
        scene.step()
        contact_n = _contact_count(scene)
        contacts.append(contact_n)
        max_total_contact_count = max(max_total_contact_count, contact_n)
        if contact_n > 0 and first_total_contact_step is None:
            first_total_contact_step = len(contacts) - 1

    zero = _zeros()
    close = _close_qvel(close_speed)
    for _ in range(settle_steps):
        step(zero)
    for _ in range(close_steps):
        step(close)
    for _ in range(push_steps):
        step(push_qvel)

    final_pos = _entity_pos_np(obj)
    displacement = final_pos - initial_pos
    final_loss = _loss_to_target(final_pos)
    return {
        "initial_pos": initial_pos.tolist(),
        "final_pos": final_pos.tolist(),
        "target_pos": list(TARGET_POS),
        "displacement": displacement.tolist(),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_improvement": initial_loss - final_loss,
        "contact_count_note": "total scene contact count, not arm-object pair-specific",
        "first_total_contact_step": first_total_contact_step,
        "max_total_contact_count": max_total_contact_count,
        "contact_trace_tail": contacts[-20:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--close-steps", type=int, default=20)
    parser.add_argument("--push-steps", type=int, default=80)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--close-speed", type=float, default=1.5)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/2026-07-09_arx_pusher/pusher_like_arm_sanity.json"),
    )
    args = parser.parse_args()

    scene, arm, obj = _make_scene()
    if arm.n_dofs != TOTAL_DOFS:
        raise RuntimeError(f"expected arm n_dofs={TOTAL_DOFS}, got {arm.n_dofs}")

    rows = []
    for name, qvel in _candidate_push_qvels(args.speed):
        result = _rollout(
            scene,
            arm,
            obj,
            qvel,
            args.settle_steps,
            args.close_steps,
            args.push_steps,
            args.close_speed,
        )
        result["program"] = name
        result["push_qvel"] = [float(x) for x in qvel.tolist()]
        rows.append(result)
        disp = result["displacement"]
        print(
            f"[pusher-like] {name:11s} dx={disp[0]:+.5f} dy={disp[1]:+.5f} "
            f"d_loss={result['loss_improvement']:+.8f} "
            f"first_total_contact={result['first_total_contact_step']} "
            f"max_total_contact={result['max_total_contact_count']}"
        )

    hold = next((row for row in rows if row["program"] == "hold"), None)
    if hold is not None:
        for row in rows:
            row["extra_dx_vs_hold"] = row["displacement"][0] - hold["displacement"][0]
            row["extra_loss_improvement_vs_hold"] = row["loss_improvement"] - hold["loss_improvement"]

    ranked = sorted(rows, key=lambda r: (r["loss_improvement"], r["displacement"][0]), reverse=True)
    payload = {
        "description": "Genesis-native multi-DOF arm Pusher-like forward sanity",
        "dt": DT,
        "table_pos": TABLE_POS,
        "table_size": TABLE_SIZE,
        "object_size": OBJECT_SIZE,
        "object_pos": OBJECT_POS,
        "target_pos": TARGET_POS,
        "settle_steps": args.settle_steps,
        "close_steps": args.close_steps,
        "push_steps": args.push_steps,
        "speed": args.speed,
        "close_speed": args.close_speed,
        "contact_count_note": "contact counts are total scene counts; pair-specific arm-object contact extraction is a later diagnostic",
        "best_by_loss_improvement": ranked[0]["program"] if ranked else None,
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    if ranked:
        best = ranked[0]
        print(
            f"[summary] best={best['program']} "
            f"dx={best['displacement'][0]:+.5f} loss_improvement={best['loss_improvement']:+.8f}"
        )
    print(f"[summary] wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
