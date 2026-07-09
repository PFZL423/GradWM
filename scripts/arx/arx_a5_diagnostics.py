"""Diagnostic suite for ARX A5 in Genesis.

This intentionally separates the "is the A5 model/gradient path healthy?"
checks from the contact-gradient gap checks:

1. import: URDF load + one forward step.
2. no_contact_backward: A5 alone, qvel trajectory loss, analytic vs FD.
3. far_box_gradient: object exists but is far away; object-action gradient
   should be zero or absent.
4. push_forward: put a box near the arm and sweep simple joint qvel programs;
   this is forward-only and does not assume contact gradients are correct.
"""
import argparse
import json
import math
import os
import subprocess
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

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_ROOT = REPO_ROOT / "external" / "ARX_Model" / "_extracted"
A5_ROOT = EXTRACTED_ROOT / "A5"
A5_URDF = A5_ROOT / "urdf" / "A5.urdf"
DEFAULT_OUT = Path("analysis/2026-07-09_arx_pusher/arx_a5_diagnostics.json")

A5_SAFE_QPOS = [0.0, 0.4, -0.8, 0.0, 0.0, 0.0]
DT = 2e-3


def _tail(text, n=20):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _patch_a5_urdf():
    if not A5_URDF.exists():
        raise FileNotFoundError(
            f"missing {A5_URDF}; run scripts/arx/arx_model_load_sanity.py or extract ARX_Model/A5/A5.7z first"
        )
    root = ET.fromstring(A5_URDF.read_text())
    mesh_paths = []
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        prefix = "package://A5/"
        if filename.startswith(prefix):
            abs_path = (A5_ROOT / filename[len(prefix):]).resolve()
            mesh.set("filename", str(abs_path))
            mesh_paths.append(str(abs_path))
        else:
            mesh_paths.append(filename)

    joints = []
    for joint in root.iter("joint"):
        joint_type = joint.get("type", "")
        if joint_type not in ("revolute", "continuous", "prismatic"):
            continue
        limit = joint.find("limit")
        lower = float(limit.get("lower", "-3.1415926")) if limit is not None else -3.1415926
        upper = float(limit.get("upper", "3.1415926")) if limit is not None else 3.1415926
        if joint_type == "continuous":
            lower, upper = -3.1415926, 3.1415926
        joints.append({
            "name": joint.get("name", ""),
            "type": joint_type,
            "lower": lower,
            "upper": upper,
            "mid": 0.5 * (lower + upper),
        })

    with tempfile.NamedTemporaryFile(prefix="arx_a5_diag_", suffix=".urdf", delete=False, mode="w") as tmp:
        tmp.write(ET.tostring(root, encoding="unicode") + "\n")
        patched = Path(tmp.name)
    return patched, mesh_paths, joints


def _worker_import():
    import torch
    import genesis as gs

    patched, mesh_paths, joints = _patch_a5_urdf()
    try:
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, substeps=4, substeps_local=4, requires_grad=False),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))
        arm = scene.add_entity(gs.morphs.URDF(file=str(patched), fixed=True, pos=(0.0, 0.0, 0.18)))
        scene.build()
        scene.reset()
        arm.set_dofs_position(torch.tensor(A5_SAFE_QPOS, dtype=torch.float32))
        for _ in range(5):
            scene.step()
        return {
            "check": "import",
            "status": "ok",
            "n_links": int(arm.n_links),
            "n_dofs": int(arm.n_dofs),
            "n_geoms": int(arm.n_geoms),
            "urdf_joints": joints,
            "genesis_joint_names": [j.name for j in arm.joints if getattr(j, "name", None)],
            "missing_meshes": [p for p in mesh_paths if p.startswith("/") and not Path(p).exists()],
        }
    finally:
        patched.unlink(missing_ok=True)


def _contact_count(scene):
    from genesis.utils.misc import qd_to_numpy

    n_contacts = qd_to_numpy(scene.rigid_solver.collider._collider_state.n_contacts)
    return int(n_contacts.reshape(-1)[0])


def _build_a5_scene(gs, requires_grad, with_far_box=False, with_push_box=False):
    patched, _, _ = _patch_a5_urdf()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=4, substeps_local=4, requires_grad=requires_grad),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))
    base_pos = (0.0, 0.0, 0.18)
    if with_push_box:
        base_pos = (0.0, 0.0, 0.05)
        scene.add_entity(
            gs.morphs.Box(pos=(0.14, 0.0, 0.05), size=(0.28, 0.18, 0.10), fixed=True),
            surface=gs.surfaces.Default(color=(0.40, 0.36, 0.30, 1.0)),
        )
    arm = scene.add_entity(gs.morphs.URDF(file=str(patched), fixed=True, pos=base_pos))
    obj = None
    if with_far_box:
        obj = scene.add_entity(
            gs.morphs.Box(pos=(0.80, 0.0, 0.025), size=(0.04, 0.04, 0.04)),
            surface=gs.surfaces.Default(color=(0.86, 0.45, 0.22, 1.0)),
        )
    if with_push_box:
        obj = scene.add_entity(
            gs.morphs.Box(pos=(0.14, 0.0, 0.125), size=(0.04, 0.04, 0.04)),
            surface=gs.surfaces.Default(color=(0.86, 0.45, 0.22, 1.0)),
        )
    scene.build()
    patched.unlink(missing_ok=True)
    return scene, arm, obj


def _set_safe_qpos(arm, torch):
    arm.set_dofs_position(torch.tensor(A5_SAFE_QPOS, dtype=torch.float32))


def _a5_velocities(torch):
    values = [
        [0.20, -0.10, 0.15, 0.05, 0.00, 0.00],
        [0.10, 0.20, -0.15, 0.00, 0.05, 0.00],
        [-0.15, 0.10, 0.20, -0.05, 0.00, 0.05],
        [0.00, -0.20, 0.10, 0.05, -0.05, 0.00],
        [0.15, 0.00, -0.10, 0.00, 0.05, -0.05],
        [-0.10, 0.15, 0.00, -0.05, 0.05, 0.00],
        [0.05, -0.15, 0.10, 0.00, 0.00, 0.05],
        [0.00, 0.10, -0.05, 0.05, 0.00, -0.05],
    ]
    return [torch.tensor(v, dtype=torch.float32) for v in values]


def _qvel_trajectory_loss(scene, arm, velocities, torch, gs, requires_grad_idx=None, perturb=None):
    scene.reset()
    _set_safe_qpos(arm, torch)
    v_tensors = []
    target = gs.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    loss = 0.0
    contact_counts = []
    for i, base_v in enumerate(velocities):
        v = base_v.clone()
        if perturb is not None and i == perturb["step"]:
            v[perturb["dof"]] += perturb["delta"]
        if i == requires_grad_idx:
            v = gs.tensor(v.detach().cpu().tolist(), requires_grad=True)
            v_tensors.append(v)
        else:
            v = gs.tensor(v.detach().cpu().tolist())
        arm.set_dofs_velocity(v)
        scene.step()
        contact_counts.append(_contact_count(scene))
        qvel = arm.get_dofs_velocity()
        loss = loss + (qvel - target).pow(2).sum()
    anchor_tensor = v_tensors[0] if v_tensors else None
    return loss, anchor_tensor, contact_counts


def _worker_no_contact_backward():
    import torch
    import genesis as gs

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene, arm, _ = _build_a5_scene(gs, requires_grad=True)
    velocities = _a5_velocities(torch)
    check_step = 3
    check_dof = 1
    eps = 1e-4

    loss, anchor_tensor, contact_counts = _qvel_trajectory_loss(
        scene, arm, velocities, torch, gs, requires_grad_idx=check_step
    )
    status = "ok"
    analytic = None
    try:
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if anchor_tensor.grad is None:
            status = "grad_none"
            analytic = 0.0
        elif torch.isnan(anchor_tensor.grad).any():
            status = "grad_nan"
        else:
            analytic = float(anchor_tensor.grad[check_dof].detach().cpu().item())
    except Exception as exc:
        status = f"backward_error:{type(exc).__name__}:{str(exc)[:160]}"

    loss_plus, _, _ = _qvel_trajectory_loss(
        scene, arm, velocities, torch, gs, perturb={"step": check_step, "dof": check_dof, "delta": eps}
    )
    loss_minus, _, _ = _qvel_trajectory_loss(
        scene, arm, velocities, torch, gs, perturb={"step": check_step, "dof": check_dof, "delta": -eps}
    )
    fd = float(((loss_plus - loss_minus) / (2 * eps)).detach().cpu().item())
    abs_error = None if analytic is None else abs(analytic - fd)
    rel_error = None if analytic is None else abs_error / (abs(fd) + 1e-12)
    return {
        "check": "no_contact_backward",
        "status": status,
        "loss": float(loss.detach().cpu().item()),
        "check_step": check_step,
        "check_dof": check_dof,
        "analytic_grad": analytic,
        "fd_grad": fd,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "max_total_contact_count": max(contact_counts) if contact_counts else 0,
        "observation_requires_grad": {
            "dofs_velocity": bool(getattr(arm.get_dofs_velocity(), "requires_grad", False)),
            "state_pos": bool(getattr(getattr(arm.get_state(), "pos", None), "requires_grad", False)),
        },
    }


def _flat_pos(obj):
    return obj.get_state().pos.reshape(-1, 3)[0]


def _object_loss_rollout(scene, arm, obj, velocities, torch, gs, requires_grad_idx=None, perturb=None):
    scene.reset()
    _set_safe_qpos(arm, torch)
    anchor_tensor = None
    contact_counts = []
    for i, base_v in enumerate(velocities):
        v = base_v.clone()
        if perturb is not None and i == perturb["step"]:
            v[perturb["dof"]] += perturb["delta"]
        if i == requires_grad_idx:
            v = gs.tensor(v.detach().cpu().tolist(), requires_grad=True)
            anchor_tensor = v
        else:
            v = gs.tensor(v.detach().cpu().tolist())
        arm.set_dofs_velocity(v)
        scene.step()
        contact_counts.append(_contact_count(scene))
    pos = _flat_pos(obj)
    target = gs.tensor([0.90, 0.0, 0.025])
    loss = (pos - target).pow(2).sum()
    return loss, anchor_tensor, contact_counts, pos


def _worker_far_box_gradient():
    import torch
    import genesis as gs

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene, arm, obj = _build_a5_scene(gs, requires_grad=True, with_far_box=True)
    velocities = _a5_velocities(torch)
    check_step = 3
    check_dof = 1
    eps = 1e-4

    loss, anchor_tensor, contact_counts, pos = _object_loss_rollout(
        scene, arm, obj, velocities, torch, gs, requires_grad_idx=check_step
    )
    status = "ok"
    analytic = 0.0
    try:
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if anchor_tensor.grad is None:
            status = "ok_no_grad_path"
        elif torch.isnan(anchor_tensor.grad).any():
            status = "grad_nan"
            analytic = float("nan")
        else:
            analytic = float(anchor_tensor.grad[check_dof].detach().cpu().item())
    except RuntimeError as exc:
        if "does not require grad" in str(exc):
            status = "ok_loss_has_no_grad_path"
            analytic = 0.0
        else:
            status = f"backward_error:{type(exc).__name__}:{str(exc)[:160]}"
            analytic = float("nan")
    except Exception as exc:
        status = f"backward_error:{type(exc).__name__}:{str(exc)[:160]}"
        analytic = float("nan")

    loss_plus, _, _, pos_plus = _object_loss_rollout(
        scene, arm, obj, velocities, torch, gs, perturb={"step": check_step, "dof": check_dof, "delta": eps}
    )
    loss_minus, _, _, pos_minus = _object_loss_rollout(
        scene, arm, obj, velocities, torch, gs, perturb={"step": check_step, "dof": check_dof, "delta": -eps}
    )
    fd = float(((loss_plus - loss_minus) / (2 * eps)).detach().cpu().item())
    obj_delta = (pos_plus - pos_minus).detach().cpu()
    return {
        "check": "far_box_gradient",
        "status": status,
        "loss": float(loss.detach().cpu().item()),
        "check_step": check_step,
        "check_dof": check_dof,
        "analytic_grad": analytic,
        "fd_grad": fd,
        "final_obj_pos": [float(x) for x in pos.detach().cpu().tolist()],
        "fd_obj_pos_delta_norm": float(obj_delta.norm().item()),
        "max_total_contact_count": max(contact_counts) if contact_counts else 0,
    }


def _push_candidates(torch, speed):
    candidates = []
    for dof in range(6):
        for sign in (-1.0, 1.0):
            v = torch.zeros(6, dtype=torch.float32)
            v[dof] = sign * speed
            candidates.append((f"j{dof + 1}{'+' if sign > 0 else '-'}", v))
    combos = [
        ("j2j3", {1: speed, 2: -speed}),
        ("j2j3_rev", {1: -speed, 2: speed}),
        ("j3j4", {2: speed, 3: speed}),
    ]
    for name, dofs in combos:
        v = torch.zeros(6, dtype=torch.float32)
        for dof, value in dofs.items():
            v[dof] = value
        candidates.append((name, v))
    return candidates


def _push_rollout(scene, arm, obj, torch, qvel, steps):
    scene.reset()
    _set_safe_qpos(arm, torch)
    initial_pos = _flat_pos(obj).detach().cpu()
    contacts = []
    for _ in range(steps):
        arm.set_dofs_velocity(qvel)
        scene.step()
        contacts.append(_contact_count(scene))
    final_pos = _flat_pos(obj).detach().cpu()
    disp = final_pos - initial_pos
    return {
        "initial_pos": [float(x) for x in initial_pos.tolist()],
        "final_pos": [float(x) for x in final_pos.tolist()],
        "displacement": [float(x) for x in disp.tolist()],
        "disp_norm": float(disp.norm().item()),
        "max_total_contact_count": max(contacts) if contacts else 0,
        "contact_trace_tail": contacts[-20:],
    }


def _worker_push_forward():
    import torch
    import genesis as gs

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene, arm, obj = _build_a5_scene(gs, requires_grad=False, with_push_box=True)
    rows = []
    for name, qvel in _push_candidates(torch, speed=1.0):
        result = _push_rollout(scene, arm, obj, torch, qvel, steps=80)
        result["program"] = name
        result["qvel"] = [float(x) for x in qvel.tolist()]
        rows.append(result)
    ranked = sorted(rows, key=lambda r: r["disp_norm"], reverse=True)
    return {
        "check": "push_forward",
        "status": "ok",
        "note": "forward-only; total contact count is not pair-specific",
        "best_program": ranked[0]["program"] if ranked else None,
        "best_disp_norm": ranked[0]["disp_norm"] if ranked else None,
        "results": rows,
    }


def _run_worker(name):
    if name == "import":
        return _worker_import()
    if name == "no_contact_backward":
        return _worker_no_contact_backward()
    if name == "far_box_gradient":
        return _worker_far_box_gradient()
    if name == "push_forward":
        return _worker_push_forward()
    raise ValueError(f"unknown worker: {name}")


def _orchestrate(args):
    script = str(Path(__file__).resolve())
    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    records = []
    for check in checks:
        cmd = [
            "conda", "run", "-n", "genesis", "--no-capture-output",
            "python", script, "--worker", check,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=os.environ.copy(),
        )
        record = next(
            (
                json.loads(line[len("__A5_DIAG__"):])
                for line in proc.stdout.splitlines()
                if line.startswith("__A5_DIAG__")
            ),
            None,
        )
        if record is None:
            record = {
                "check": check,
                "status": f"worker_error:returncode={proc.returncode}",
                "stdout_tail": _tail(proc.stdout),
                "stderr_tail": _tail(proc.stderr),
            }
        records.append(record)
        print(f"[a5-diag:{check}] status={record.get('status')}")
        if record.get("check") == "no_contact_backward":
            print(
                f"  analytic={record.get('analytic_grad')} fd={record.get('fd_grad')} "
                f"rel_error={record.get('rel_error')} max_contact={record.get('max_total_contact_count')}"
            )
        elif record.get("check") == "far_box_gradient":
            print(
                f"  analytic={record.get('analytic_grad')} fd={record.get('fd_grad')} "
                f"obj_delta_norm={record.get('fd_obj_pos_delta_norm')} "
                f"max_contact={record.get('max_total_contact_count')}"
            )
        elif record.get("check") == "push_forward":
            print(
                f"  best={record.get('best_program')} "
                f"best_disp_norm={record.get('best_disp_norm')}"
            )

    payload = {
        "description": "ARX A5 Genesis forward/backward diagnostic suite",
        "safe_qpos": A5_SAFE_QPOS,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[a5-diag] wrote {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=None)
    parser.add_argument(
        "--checks",
        default="import,no_contact_backward,far_box_gradient,push_forward",
        help="comma-separated checks for orchestrator mode",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.worker:
        result = _run_worker(args.worker)
        print("__A5_DIAG__" + json.dumps(result))
        return
    _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
