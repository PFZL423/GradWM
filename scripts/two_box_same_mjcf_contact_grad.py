"""Same-MJCF two-box contact gradient probe.

Tests whether object-position loss can backpropagate through rigid contact to
a pusher freejoint velocity when both boxes live in one MJCF entity.
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
from genesis.utils.misc import qd_to_numpy


def _write_temp_mjcf(text):
    with tempfile.NamedTemporaryFile(prefix="two_box_same_", suffix=".xml", delete=False, mode="w") as f:
        f.write(text)
        return f.name


def _two_box_mjcf():
    return """<mujoco model="two_box_same_mjcf">
    <worldbody>
        <body name="pusher" pos="0 0 0.025">
            <freejoint/>
            <geom name="pusher_box" type="box" size="0.04 0.05 0.04"
                  mass="0.05" rgba="0.2 0.4 0.8 1" contype="1" conaffinity="1"/>
        </body>
        <body name="object" pos="0.045 0 0.025">
            <freejoint/>
            <geom name="object_box" type="box" size="0.04 0.05 0.04"
                  mass="0.05" rgba="0.8 0.4 0.2 1" contype="1" conaffinity="1"/>
        </body>
    </worldbody>
</mujoco>
"""


def _object_pos(ent):
    pos = ent.get_state().pos.reshape(-1, 3)
    if pos.shape[0] < 2:
        state = ent.get_state()
        fields = [name for name in dir(state) if not name.startswith("_")]
        raise RuntimeError(f"entity state pos has shape {tuple(pos.shape)}; state fields={fields}")
    return pos[1]


def _contact_summary(scene):
    collider_state = scene.rigid_solver.collider._collider_state
    n_contacts = qd_to_numpy(collider_state.n_contacts)
    n = int(n_contacts.reshape(-1)[0])
    pairs = []
    if n:
        link_a = qd_to_numpy(collider_state.contact_data.link_a)
        link_b = qd_to_numpy(collider_state.contact_data.link_b)
        geom_a = qd_to_numpy(collider_state.contact_data.geom_a)
        geom_b = qd_to_numpy(collider_state.contact_data.geom_b)
        penetration = qd_to_numpy(collider_state.contact_data.penetration)
        for i in range(min(n, 4)):
            pairs.append({
                "link_a": int(link_a[i, 0]),
                "link_b": int(link_b[i, 0]),
                "geom_a": int(geom_a[i, 0]),
                "geom_b": int(geom_b[i, 0]),
                "penetration": float(penetration[i, 0]),
            })
    return {"n": n, "pairs": pairs}


def _tensor_info(t):
    return {
        "requires_grad": bool(getattr(t, "requires_grad", False)),
        "is_leaf": bool(getattr(t, "is_leaf", False)),
        "grad_fn": None if getattr(t, "grad_fn", None) is None else type(t.grad_fn).__name__,
    }


def run_once(horizon, vx, target_x, backward, loss_kind, drive_mode):
    tmp = _write_temp_mjcf(_two_box_mjcf())
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=2e-3, substeps=4, substeps_local=4, requires_grad=True),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))
        ent = scene.add_entity(gs.morphs.MJCF(file=tmp))
        scene.build()
        scene.reset()

        pusher_vel = [vx, 0.0, 0.0, 0.0, 0.0, 0.0]
        full_vel = pusher_vel + [0.0] * max(0, ent.n_dofs - 6)
        v_list = []
        contact_counts = []
        contact_pairs_first = None
        if drive_mode == "initial-only":
            full_v = gs.tensor(full_vel, requires_grad=True)
            ent.set_dofs_velocity(full_v)
            v_list.append(full_v)
            for _ in range(horizon):
                scene.step()
                summary = _contact_summary(scene)
                contact_counts.append(summary["n"])
                if contact_pairs_first is None and summary["n"]:
                    contact_pairs_first = summary["pairs"]
        elif drive_mode == "each-step":
            for _ in range(horizon):
                full_v = gs.tensor(full_vel, requires_grad=True)
                ent.set_dofs_velocity(full_v)
                scene.step()
                v_list.append(full_v)
                summary = _contact_summary(scene)
                contact_counts.append(summary["n"])
                if contact_pairs_first is None and summary["n"]:
                    contact_pairs_first = summary["pairs"]
        else:
            raise ValueError(drive_mode)

        qvel = ent.get_dofs_velocity()
        object_qvel = qvel[6:9]
        if loss_kind == "object-qvel":
            target = gs.tensor([target_x, 0.0, 0.0])
            loss = (object_qvel - target).pow(2).sum()
            pos = None
        elif loss_kind == "object-pos":
            pos = _object_pos(ent)
            target = gs.tensor([target_x, 0.0, 0.025])
            loss = (pos - target).pow(2).sum()
        else:
            raise ValueError(loss_kind)

        status = "not_backward"
        grad_norms = []
        if backward:
            status = "ok"
            try:
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                status = f"bwd_error:{repr(e)[:160]}"
            for v in v_list:
                if v.grad is None:
                    grad_norms.append(None)
                elif torch.isnan(v.grad).any():
                    grad_norms.append(float("nan"))
                else:
                    grad_norms.append(float(v.grad.norm().item()))

        finite = [g for g in grad_norms if g is not None and torch.isfinite(torch.tensor(g))]
        pusher_grad_sum = 0.0
        object_grad_sum = 0.0
        for v in v_list:
            if v.grad is None or torch.isnan(v.grad).any():
                continue
            pusher_grad_sum += float(v.grad[:6].norm().item())
            object_grad_sum += float(v.grad[6:].norm().item())
        return {
            "horizon": horizon,
            "vx": vx,
            "drive_mode": drive_mode,
            "n_links": ent.n_links,
            "n_dofs": ent.n_dofs,
            "loss_kind": loss_kind,
            "input_tensor_count": len(v_list),
            "input_tensor_info_first": None if not v_list else _tensor_info(v_list[0]),
            "input_tensor_info_last": None if not v_list else _tensor_info(v_list[-1]),
            "contact_counts": contact_counts,
            "contact_max": max(contact_counts) if contact_counts else 0,
            "contact_steps": sum(1 for n in contact_counts if n),
            "contact_pairs_first": contact_pairs_first,
            "object_pos": None if pos is None else [float(x) for x in pos.detach().cpu().tolist()],
            "object_qvel": [float(x) for x in object_qvel.detach().cpu().tolist()],
            "qvel_requires_grad": bool(getattr(object_qvel, "requires_grad", False)),
            "pos_requires_grad": None if pos is None else bool(getattr(pos, "requires_grad", False)),
            "loss": float(loss.item()),
            "status": status,
            "grad_none": sum(g is None for g in grad_norms),
            "grad_nan": sum(1 for g in grad_norms if isinstance(g, float) and not torch.isfinite(torch.tensor(g))),
            "grad_norm_first": grad_norms[0] if grad_norms else None,
            "grad_norm_last": grad_norms[-1] if grad_norms else None,
            "grad_norm_sum": float(sum(finite)) if finite else 0.0,
            "pusher_grad_norm_sum": pusher_grad_sum,
            "object_grad_norm_sum": object_grad_sum,
            "grad_first_tensor": None if not v_list or v_list[0].grad is None else [float(x) for x in v_list[0].grad.detach().cpu().tolist()],
            "grad_last_tensor": None if not v_list or v_list[-1].grad is None else [float(x) for x in v_list[-1].grad.detach().cpu().tolist()],
        }
    finally:
        Path(tmp).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--vx", type=float, default=0.40)
    parser.add_argument("--target-x", type=float, default=0.10)
    parser.add_argument("--loss-kind", choices=("object-qvel", "object-pos"), default="object-qvel")
    parser.add_argument("--drive-mode", choices=("each-step", "initial-only"), default="each-step")
    parser.add_argument("--no-backward", action="store_true")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    result = run_once(
        args.horizon,
        args.vx,
        args.target_x,
        backward=not args.no_backward,
        loss_kind=args.loss_kind,
        drive_mode=args.drive_mode,
    )
    print("__SAMEBOX__" + json.dumps(result))


if __name__ == "__main__":
    raise SystemExit(main())
