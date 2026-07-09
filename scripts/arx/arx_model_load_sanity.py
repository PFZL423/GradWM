"""Genesis load sanity for ARX A5 / ACone assets.

The ARX URDFs use ROS-style package:// mesh paths. This script writes a
temporary patched URDF with absolute mesh paths, loads the selected model in
Genesis, sets a mid-range qpos, and steps forward a few frames.
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

import torch
import genesis as gs
from genesis.utils.misc import qd_to_numpy


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_ROOT = REPO_ROOT / "external" / "ARX_Model" / "_extracted"

MODEL_CONFIGS = {
    "a5": {
        "root": EXTRACTED_ROOT / "A5",
        "urdf": EXTRACTED_ROOT / "A5" / "urdf" / "A5.urdf",
        "package": "A5",
        "pos": (-0.35, 0.0, 0.25),
    },
    "acone": {
        "root": EXTRACTED_ROOT / "acone",
        "urdf": EXTRACTED_ROOT / "acone" / "urdf" / "acone.urdf",
        "package": "acone",
        "pos": (0.35, 0.0, 0.35),
    },
}


def _patch_urdf(model_name: str) -> tuple[Path, list[str], list[dict]]:
    cfg = MODEL_CONFIGS[model_name]
    urdf_path = cfg["urdf"]
    if not urdf_path.exists():
        raise FileNotFoundError(f"missing URDF for {model_name}: {urdf_path}")

    root = ET.fromstring(urdf_path.read_text())
    mesh_paths = []
    prefix = f"package://{cfg['package']}/"
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        if filename.startswith(prefix):
            rel = filename[len(prefix):]
            abs_path = (cfg["root"] / rel).resolve()
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

    with tempfile.NamedTemporaryFile(
        prefix=f"arx_{model_name}_", suffix=".urdf", delete=False, mode="w"
    ) as tmp:
        tmp.write(ET.tostring(root, encoding="unicode") + "\n")
        patched = Path(tmp.name)
    return patched, mesh_paths, joints


def _entity_joint_names(entity) -> list[str]:
    names = []
    for joint in getattr(entity, "joints", []):
        name = getattr(joint, "name", None)
        if name:
            names.append(name)
    return names


def _entity_link_names(entity) -> list[str]:
    names = []
    for link in getattr(entity, "links", []):
        name = getattr(link, "name", None)
        if name:
            names.append(name)
    return names


def _entity_link_positions(entity) -> list[list[float]]:
    state_pos = getattr(entity.get_state(), "pos", None)
    pos = state_pos
    if pos is None and callable(getattr(entity, "get_links_pos", None)):
        # Non-differentiable, but useful for asset import placement diagnostics.
        pos = entity.get_links_pos()
    if pos is None:
        return []
    try:
        arr = qd_to_numpy(pos)
    except Exception:
        try:
            arr = pos.detach().cpu().numpy()
        except Exception:
            return []
    if arr is None:
        return []
    arr = arr.reshape(-1, 3)
    return [[float(v) for v in row.tolist()] for row in arr]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="a5,acone")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/2026-07-09_arx_pusher/arx_model_load_sanity.json"),
    )
    args = parser.parse_args()

    model_names = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    for name in model_names:
        if name not in MODEL_CONFIGS:
            raise ValueError(f"unknown model {name!r}; choices={sorted(MODEL_CONFIGS)}")

    patched_paths = []
    records = []
    try:
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")
        scene = gs.Scene(show_viewer=False)
        scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))

        entities = []
        for name in model_names:
            patched, mesh_paths, joints = _patch_urdf(name)
            patched_paths.append(patched)
            ent = scene.add_entity(
                gs.morphs.URDF(
                    file=str(patched),
                    fixed=True,
                    pos=MODEL_CONFIGS[name]["pos"],
                )
            )
            entities.append((name, ent, mesh_paths, joints))

        scene.build()
        scene.reset()

        for name, ent, mesh_paths, joints in entities:
            qpos = torch.tensor([j["mid"] for j in joints], dtype=torch.float32)
            if ent.n_dofs == len(joints):
                ent.set_dofs_position(qpos)
            else:
                qpos = torch.zeros(ent.n_dofs, dtype=torch.float32)
                ent.set_dofs_position(qpos)
            records.append({
                "model": name,
                "status": "build_ok",
                "n_links": int(ent.n_links),
                "n_dofs": int(ent.n_dofs),
                "n_geoms": int(ent.n_geoms),
                "urdf_joint_count": len(joints),
                "urdf_joints": joints,
                "genesis_joint_names": _entity_joint_names(ent),
                "genesis_link_names": _entity_link_names(ent),
                "link_positions_after_qpos": _entity_link_positions(ent),
                "mesh_count": len(mesh_paths),
                "missing_meshes": [p for p in mesh_paths if p.startswith("/") and not Path(p).exists()],
            })
            print(
                f"[arx:{name}] build ok n_links={ent.n_links} "
                f"n_dofs={ent.n_dofs} n_geoms={ent.n_geoms} urdf_joints={len(joints)}"
            )

        for _ in range(args.steps):
            scene.step()
        print(f"[arx] step ok steps={args.steps}")

    finally:
        for path in patched_paths:
            path.unlink(missing_ok=True)

    payload = {
        "description": "ARX A5 / ACone Genesis URDF load sanity",
        "models": model_names,
        "extracted_root": str(EXTRACTED_ROOT),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[arx] wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
