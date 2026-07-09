"""Check whether Genesis can restore a saved state and continue one step.

This is a forward-only continuation test for local-anchor data collection:
save an online state, reset the scene to that state, replay the next action,
and compare the restored transition with the original online transition.
"""
import argparse
import json
import os
import pickle
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

import numpy as np
import torch
import genesis as gs
from genesis.utils.misc import qd_to_numpy


def _write_temp_mjcf(text):
    with tempfile.NamedTemporaryFile(prefix="state_restore_two_box_", suffix=".xml", delete=False, mode="w") as f:
        f.write(text)
        return f.name


def _two_box_mjcf(object_x):
    return f"""<mujoco model="state_restore_two_box">
    <worldbody>
        <body name="pusher" pos="0 0 0.025">
            <freejoint/>
            <geom name="pusher_box" type="box" size="0.04 0.05 0.04"
                  mass="0.05" rgba="0.2 0.4 0.8 1" contype="1" conaffinity="1"/>
        </body>
        <body name="object" pos="{object_x:.8f} 0 0.025">
            <freejoint/>
            <geom name="object_box" type="box" size="0.04 0.05 0.04"
                  mass="0.05" rgba="0.8 0.4 0.2 1" contype="1" conaffinity="1"/>
        </body>
    </worldbody>
</mujoco>
"""


def _contact_count(scene):
    n_contacts = qd_to_numpy(scene.rigid_solver.collider._collider_state.n_contacts)
    return int(n_contacts.reshape(-1)[0])


def _flat_entity_pos(ent):
    return ent.get_state().pos.detach().cpu().numpy().reshape(-1, 3)


def _flat_qvel(ent):
    return ent.get_dofs_velocity().detach().cpu().numpy().reshape(-1)


def _rigid_solver_state(scene):
    state = scene.get_state()
    for solver_state in state.solvers_state:
        if solver_state is not None and solver_state.__class__.__name__ == "RigidSolverState":
            return state, solver_state
    raise RuntimeError("RigidSolverState not found in scene state")


def _snapshot(scene, ent):
    _, rigid = _rigid_solver_state(scene)
    return {
        "entity_pos": _flat_entity_pos(ent).tolist(),
        "entity_qvel": _flat_qvel(ent).tolist(),
        "qpos": rigid.qpos.detach().cpu().numpy().reshape(-1).tolist(),
        "dofs_vel": rigid.dofs_vel.detach().cpu().numpy().reshape(-1).tolist(),
        "dofs_acc": rigid.dofs_acc.detach().cpu().numpy().reshape(-1).tolist(),
        "links_pos": rigid.links_pos.detach().cpu().numpy().reshape(-1).tolist(),
        "links_quat": rigid.links_quat.detach().cpu().numpy().reshape(-1).tolist(),
        "contact_count": _contact_count(scene),
    }


def _max_abs_diff(a, b, key):
    arr_a = np.asarray(a[key], dtype=np.float64).reshape(-1)
    arr_b = np.asarray(b[key], dtype=np.float64).reshape(-1)
    if arr_a.shape != arr_b.shape:
        return None
    if arr_a.size == 0:
        return 0.0
    return float(np.max(np.abs(arr_a - arr_b)))


def _compare(a, b):
    keys = ["entity_pos", "entity_qvel", "qpos", "dofs_vel", "dofs_acc", "links_pos", "links_quat"]
    out = {f"{key}_max_abs": _max_abs_diff(a, b, key) for key in keys}
    out["contact_count_a"] = a["contact_count"]
    out["contact_count_b"] = b["contact_count"]
    out["contact_count_diff"] = int(b["contact_count"] - a["contact_count"])
    out["max_state_abs"] = max(v for k, v in out.items() if k.endswith("_max_abs") and v is not None)
    return out


def _set_action(ent, vx):
    full_v = torch.zeros(ent.n_dofs, dtype=torch.float32)
    full_v[0] = vx
    ent.set_dofs_velocity(full_v)


def _step_with_action(scene, ent, vx):
    _set_action(ent, vx)
    scene.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_state_picklable(state):
    state._scene = None
    for solver_state in state.solvers_state:
        if solver_state is not None:
            solver_state.serializable()
    return state


def _build_two_box_scene(case_name):
    object_x = 0.045 if case_name == "contact" else 0.20
    tmp = _write_temp_mjcf(_two_box_mjcf(object_x))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=4, substeps_local=4, requires_grad=False),
        show_viewer=False,
    )
    ent = scene.add_entity(gs.morphs.MJCF(file=tmp))
    scene.build()
    scene.reset()
    Path(tmp).unlink(missing_ok=True)
    return scene, ent


def _build_box_push_scene():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=4, substeps_local=4, requires_grad=False),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))
    pusher = scene.add_entity(gs.morphs.Box(size=(0.04, 0.05, 0.04), pos=(0.0, 0.0, 0.025)))
    scene.add_entity(gs.morphs.Box(size=(0.04, 0.05, 0.04), pos=(0.045, 0.0, 0.025)))
    scene.build()
    scene.reset()
    return scene, pusher


def _build_scene(case_name):
    if case_name == "box_push":
        return _build_box_push_scene()
    if case_name in ("no_contact", "contact"):
        return _build_two_box_scene(case_name)
    raise ValueError(f"unknown case: {case_name}")


def _run_load_state_worker(case_name, state_path, vx):
    scene, ent = _build_scene(case_name)
    with Path(state_path).open("rb") as f:
        state = pickle.load(f)
    scene.reset(state=state)
    restored_anchor = _snapshot(scene, ent)
    _step_with_action(scene, ent, vx)
    restored_next = _snapshot(scene, ent)
    print("__STATE_RESTORE_WORKER__" + json.dumps({
        "restored_anchor": restored_anchor,
        "restored_next": restored_next,
    }))


def _run_cross_process_restore(case_name, anchor_state, online_next, vx):
    with tempfile.NamedTemporaryFile(prefix="genesis_state_restore_", suffix=".pkl", delete=False) as f:
        state_path = Path(f.name)
    try:
        picklable_state = _make_state_picklable(anchor_state)
        with state_path.open("wb") as f:
            pickle.dump(picklable_state, f)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--load-state",
            str(state_path),
            "--case",
            case_name,
            "--vx",
            str(vx),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=os.environ.copy())
        worker_result = next(
            (
                json.loads(line[len("__STATE_RESTORE_WORKER__"):])
                for line in proc.stdout.splitlines()
                if line.startswith("__STATE_RESTORE_WORKER__")
            ),
            None,
        )
        if worker_result is None:
            return {
                "status": f"error:returncode={proc.returncode}",
                "stdout_tail": "\n".join(proc.stdout.splitlines()[-8:]),
                "stderr_tail": "\n".join(proc.stderr.splitlines()[-8:]),
            }
        return {
            "status": "ok",
            "restored_next_vs_online_next": _compare(online_next, worker_result["restored_next"]),
            "restored_anchor_contact_count": worker_result["restored_anchor"]["contact_count"],
        }
    except Exception as exc:
        return {"status": f"error:{type(exc).__name__}:{str(exc)[:160]}"}
    finally:
        state_path.unlink(missing_ok=True)


def _run_case(case_name, pre_steps, vx, delta_vx):
    scene, ent = _build_scene(case_name)

    contact_trace = []
    for _ in range(pre_steps):
        _step_with_action(scene, ent, vx)
        contact_trace.append(_contact_count(scene))

    anchor_state = scene.get_state()
    anchor_snapshot = _snapshot(scene, ent)

    _step_with_action(scene, ent, vx)
    online_next = _snapshot(scene, ent)

    scene.reset(state=anchor_state)
    restored_anchor = _snapshot(scene, ent)
    _step_with_action(scene, ent, vx)
    restored_next = _snapshot(scene, ent)

    scene.reset(state=anchor_state)
    _step_with_action(scene, ent, vx + delta_vx)
    perturbed_a = _snapshot(scene, ent)

    scene.reset(state=anchor_state)
    _step_with_action(scene, ent, vx + delta_vx)
    perturbed_b = _snapshot(scene, ent)

    cross_process = _run_cross_process_restore(case_name, anchor_state, online_next, vx)

    serializable_status = "not_checked"
    try:
        anchor_state.serializable()
        serializable_status = "ok"
    except Exception as exc:
        serializable_status = f"error:{type(exc).__name__}:{str(exc)[:120]}"

    return {
        "case": case_name,
        "pre_steps": pre_steps,
        "vx": vx,
        "delta_vx": delta_vx,
        "contact_trace": contact_trace,
        "anchor_contact_count": anchor_snapshot["contact_count"],
        "online_next_contact_count": online_next["contact_count"],
        "restored_anchor_vs_anchor": _compare(anchor_snapshot, restored_anchor),
        "restored_next_vs_online_next": _compare(online_next, restored_next),
        "perturbed_repeat_vs_repeat": _compare(perturbed_a, perturbed_b),
        "perturbed_delta_from_online": _compare(online_next, perturbed_a),
        "cross_process_manual_pickle_restore": cross_process,
        "serializable_status_after_test": serializable_status,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="no_contact")
    parser.add_argument("--load-state", type=Path, default=None)
    parser.add_argument("--case", default=None)
    parser.add_argument("--pre-steps", type=int, default=8)
    parser.add_argument("--vx", type=float, default=0.40)
    parser.add_argument("--delta-vx", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("analysis/genesis_state_restore_check.json"))
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    if args.load_state is not None:
        if args.case is None:
            raise ValueError("--case is required with --load-state")
        _run_load_state_worker(args.case, args.load_state, args.vx)
        return

    backend_name = "cpu" if gs.backend == gs.cpu else "gpu" if gs.backend == gs.gpu else str(gs.backend)
    results = []
    for case_name in [case.strip() for case in args.cases.split(",") if case.strip()]:
        results.append(_run_case(case_name, args.pre_steps, args.vx, args.delta_vx))

    payload = {
        "description": "Genesis same-process state restore continuation check",
        "backend": backend_name,
        "precision": "32",
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("__STATE_RESTORE__" + json.dumps(payload))


if __name__ == "__main__":
    raise SystemExit(main())
