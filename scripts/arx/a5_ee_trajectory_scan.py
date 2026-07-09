"""FK scan for A5 end-effector horizontal sweep candidates.

The goal is to find initial qpos + joint qvel programs where the A5 tip proxy
moves mostly in the table plane. Those candidates are used to place a box in a
real A5 Pusher-like scene.
"""
import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
A5_URDF = REPO_ROOT / "external" / "ARX_Model" / "_extracted" / "A5" / "urdf" / "A5.urdf"
DEFAULT_OUT = Path("analysis/2026-07-09_arx_pusher/a5_ee_trajectory_scan.json")
DEFAULT_CSV = Path("analysis/2026-07-09_arx_pusher/a5_ee_trajectory_scan.csv")

BASE_POS = np.array([0.0, 0.0, 0.05], dtype=float)
TIP_OFFSET_LINK6 = np.array([0.08, 0.0, 0.0], dtype=float)
DT = 2e-3


def _rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    t = np.eye(4, dtype=float)
    t[:3, :3] = _rpy_matrix(rpy)
    t[:3, 3] = np.asarray(xyz, dtype=float)
    return t


def _parse_vec(raw, default):
    if raw is None:
        return tuple(default)
    return tuple(float(x) for x in raw.split())


def _parse_a5_chain():
    root = ET.fromstring(A5_URDF.read_text())
    joints = []
    for joint in root.iter("joint"):
        joint_type = joint.get("type", "")
        if joint_type not in ("revolute", "continuous", "prismatic"):
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        xyz = _parse_vec(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_vec(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        axis_xyz = _parse_vec(axis.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0))
        lower = float(limit.get("lower", "-3.1415926")) if limit is not None else -3.1415926
        upper = float(limit.get("upper", "3.1415926")) if limit is not None else 3.1415926
        if joint_type == "continuous":
            lower, upper = -math.pi, math.pi
        joints.append(
            {
                "name": joint.get("name", ""),
                "type": joint_type,
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis_xyz,
                "lower": lower,
                "upper": upper,
            }
        )
    if len(joints) != 6:
        raise RuntimeError(f"expected 6 A5 joints, got {len(joints)} from {A5_URDF}")
    return joints


def _fk_tip(joints, qpos, base_pos):
    t = _transform(base_pos, (0.0, 0.0, 0.0))
    for i, joint in enumerate(joints):
        t = t @ _transform(joint["xyz"], joint["rpy"])
        if joint["type"] in ("revolute", "continuous"):
            r = np.eye(4, dtype=float)
            r[:3, :3] = _axis_angle(joint["axis"], qpos[i])
            t = t @ r
        elif joint["type"] == "prismatic":
            p = np.eye(4, dtype=float)
            p[:3, 3] = np.asarray(joint["axis"], dtype=float) * qpos[i]
            t = t @ p
    tip_h = np.ones(4, dtype=float)
    tip_h[:3] = TIP_OFFSET_LINK6
    return (t @ tip_h)[:3]


def _candidate_qpos(joints):
    grids = [
        [-0.6, 0.0, 0.6],
        [0.2, 0.5, 0.8, 1.1, 1.4],
        [-0.4, -0.8, -1.2, -1.6],
        [-0.5, 0.0, 0.5],
        [0.0],
        [0.0],
    ]
    out = []
    for q1 in grids[0]:
        for q2 in grids[1]:
            for q3 in grids[2]:
                for q4 in grids[3]:
                    for q5 in grids[4]:
                        for q6 in grids[5]:
                            q = np.array([q1, q2, q3, q4, q5, q6], dtype=float)
                            if _inside_limits(joints, q):
                                out.append(q)
    return out


def _candidate_qvels(speed):
    out = []
    for i in range(6):
        for sign in (-1.0, 1.0):
            v = np.zeros(6, dtype=float)
            v[i] = sign * speed
            out.append((f"j{i + 1}{'+' if sign > 0 else '-'}", v))
    combos = [
        ("j2+j3-", {1: speed, 2: -speed}),
        ("j2-j3+", {1: -speed, 2: speed}),
        ("j2+j4+", {1: speed, 3: speed}),
        ("j2-j4-", {1: -speed, 3: -speed}),
        ("j3+j4+", {2: speed, 3: speed}),
        ("j3-j4-", {2: -speed, 3: -speed}),
        ("j1+j2+", {0: speed, 1: speed}),
        ("j1-j2-", {0: -speed, 1: -speed}),
    ]
    for name, dofs in combos:
        v = np.zeros(6, dtype=float)
        for dof, value in dofs.items():
            v[dof] = value
        out.append((name, v))
    return out


def _inside_limits(joints, qpos, margin=1e-6):
    for q, joint in zip(qpos, joints):
        if q < joint["lower"] - margin or q > joint["upper"] + margin:
            return False
    return True


def _score(start, end, target_z):
    disp = end - start
    horizontal = float(np.linalg.norm(disp[:2]))
    vertical = float(abs(disp[2]))
    start_z_penalty = float(abs(start[2] - target_z))
    return horizontal - 2.0 * vertical - 0.25 * start_z_penalty


def _row(program, qpos, qvel, start, end, score):
    disp = end - start
    horizontal = float(np.linalg.norm(disp[:2]))
    vertical = float(abs(disp[2]))
    return {
        "program": program,
        "score": score,
        "qpos": [float(x) for x in qpos.tolist()],
        "qvel": [float(x) for x in qvel.tolist()],
        "start_tip": [float(x) for x in start.tolist()],
        "end_tip": [float(x) for x in end.tolist()],
        "disp": [float(x) for x in disp.tolist()],
        "horizontal_disp": horizontal,
        "vertical_abs_disp": vertical,
        "vertical_over_horizontal": vertical / (horizontal + 1e-12),
    }


def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "program",
        "score",
        "horizontal_disp",
        "vertical_abs_disp",
        "vertical_over_horizontal",
        "qpos",
        "qvel",
        "start_tip",
        "end_tip",
        "disp",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            flat = dict(row)
            flat["rank"] = i + 1
            for key in ("qpos", "qvel", "start_tip", "end_tip", "disp"):
                flat[key] = json.dumps(flat[key])
            writer.writerow(flat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--target-z", type=float, default=0.12)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    joints = _parse_a5_chain()
    horizon = args.steps * DT
    rows = []
    for qpos in _candidate_qpos(joints):
        start = _fk_tip(joints, qpos, BASE_POS)
        for program, qvel in _candidate_qvels(args.speed):
            q_next = qpos + qvel * horizon
            if not _inside_limits(joints, q_next):
                continue
            end = _fk_tip(joints, q_next, BASE_POS)
            score = _score(start, end, args.target_z)
            rows.append(_row(program, qpos, qvel, start, end, score))

    rows.sort(key=lambda r: r["score"], reverse=True)
    top = rows[: args.top_k]
    payload = {
        "description": "FK scan for A5 tip-proxy horizontal sweep candidates",
        "urdf": str(A5_URDF),
        "base_pos": [float(x) for x in BASE_POS.tolist()],
        "tip_offset_link6": [float(x) for x in TIP_OFFSET_LINK6.tolist()],
        "dt": DT,
        "steps": args.steps,
        "horizon_seconds": horizon,
        "speed": args.speed,
        "target_z": args.target_z,
        "num_candidates": len(rows),
        "top": top,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(rows, args.csv)

    print(
        f"[a5-ee-scan] candidates={len(rows)} top_k={len(top)} "
        f"horizon={horizon:.3f}s out={args.out}"
    )
    for i, row in enumerate(top[:10], start=1):
        print(
            f"  #{i:02d} {row['program']:8s} score={row['score']:+.5f} "
            f"hxy={row['horizontal_disp']:.5f} dz={row['vertical_abs_disp']:.5f} "
            f"start={row['start_tip']} disp={row['disp']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
