"""Forward-only A5 Pusher-like sanity using FK-scan candidate.

This script places a small box on the predicted horizontal sweep arc of the
A5 tip proxy and checks whether Genesis produces actual horizontal object
motion. It does not test gradients yet.
"""
import argparse
import json
import os
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

import torch
import genesis as gs

from arx_a5_diagnostics import _contact_count, _patch_a5_urdf


DT = 2e-3
DEFAULT_OUT = Path("analysis/2026-07-09_arx_pusher/a5_pusher_forward_sanity.json")
DEFAULT_QPOS = [0.0, 1.4, -0.4, 0.5, 0.0, 0.0]
DEFAULT_QVEL = [1.6, 0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_OBJ_POS = (0.306, 0.076, 0.120)
DEFAULT_PUSH_STEPS = 170
TABLE_SIZE = (0.30, 0.20, 0.10)
TABLE_POS = (0.30, 0.025, 0.05)
OBJECT_SIZE = (0.04, 0.04, 0.04)


def _parse_vec(text, expected, name):
    values = [float(x) for x in text.split(",") if x.strip()]
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"{name} expects {expected} comma-separated values, got {len(values)}")
    return values


def _flat_pos(obj):
    return obj.get_state().pos.detach().cpu().reshape(-1, 3)[0]


def _make_scene(obj_pos, requires_grad=False):
    patched, _, _ = _patch_a5_urdf()
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, substeps=4, substeps_local=4, requires_grad=requires_grad),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))
        scene.add_entity(
            gs.morphs.Box(pos=TABLE_POS, size=TABLE_SIZE, fixed=True),
            surface=gs.surfaces.Default(color=(0.40, 0.36, 0.30, 1.0)),
        )
        arm = scene.add_entity(gs.morphs.URDF(file=str(patched), fixed=True, pos=(0.0, 0.0, 0.05)))
        obj = scene.add_entity(
            gs.morphs.Box(pos=obj_pos, size=OBJECT_SIZE),
            surface=gs.surfaces.Default(color=(0.86, 0.45, 0.22, 1.0)),
        )
        scene.build()
    finally:
        patched.unlink(missing_ok=True)
    return scene, arm, obj


def _velocity_tensor(values, use_gs_tensor):
    if use_gs_tensor:
        return gs.tensor(values)
    return torch.tensor(values, dtype=torch.float32)


def _rollout(scene, arm, obj, qpos, qvel, settle_steps, push_steps, use_gs_tensor=False):
    scene.reset()
    arm.set_dofs_position(torch.tensor(qpos, dtype=torch.float32))

    zero = _velocity_tensor([0.0 for _ in qvel], use_gs_tensor)
    contacts = []
    for _ in range(settle_steps):
        arm.set_dofs_velocity(zero)
        scene.step()
        contacts.append(_contact_count(scene))

    initial_pos = _flat_pos(obj)
    push = _velocity_tensor(qvel, use_gs_tensor)
    for _ in range(push_steps):
        arm.set_dofs_velocity(push)
        scene.step()
        contacts.append(_contact_count(scene))

    final_pos = _flat_pos(obj)
    disp = final_pos - initial_pos
    return {
        "initial_pos": [float(x) for x in initial_pos.tolist()],
        "final_pos": [float(x) for x in final_pos.tolist()],
        "displacement": [float(x) for x in disp.tolist()],
        "horizontal_disp": float(disp[:2].norm().item()),
        "vertical_disp": float(disp[2].item()),
        "max_total_contact_count": max(contacts) if contacts else 0,
        "contact_trace_tail": contacts[-20:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-x", type=float, default=DEFAULT_OBJ_POS[0])
    parser.add_argument("--obj-y", type=float, default=DEFAULT_OBJ_POS[1])
    parser.add_argument("--obj-z", type=float, default=DEFAULT_OBJ_POS[2])
    parser.add_argument("--qpos", type=lambda x: _parse_vec(x, 6, "qpos"), default=DEFAULT_QPOS)
    parser.add_argument("--qvel", type=lambda x: _parse_vec(x, 6, "qvel"), default=DEFAULT_QVEL)
    parser.add_argument("--requires-grad", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--push-steps", type=int, default=DEFAULT_PUSH_STEPS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    obj_pos = (args.obj_x, args.obj_y, args.obj_z)
    scene, arm, obj = _make_scene(obj_pos, requires_grad=args.requires_grad)
    result = _rollout(
        scene,
        arm,
        obj,
        args.qpos,
        args.qvel,
        args.settle_steps,
        args.push_steps,
        use_gs_tensor=args.requires_grad,
    )
    payload = {
        "description": "A5 FK-candidate Pusher-like forward sanity",
        "qpos": args.qpos,
        "qvel": args.qvel,
        "object_pos": obj_pos,
        "object_size": OBJECT_SIZE,
        "table_pos": TABLE_POS,
        "table_size": TABLE_SIZE,
        "requires_grad": args.requires_grad,
        "settle_steps": args.settle_steps,
        "push_steps": args.push_steps,
        "result": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[a5-pusher] obj={obj_pos} hdisp={result['horizontal_disp']:.6f} "
        f"vdisp={result['vertical_disp']:.6f} max_contact={result['max_total_contact_count']}"
    )
    print(f"[a5-pusher] wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
