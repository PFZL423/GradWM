"""Compare arm input gradients for the scripted grasp with and without rope.

This is a narrow reconciliation test for the close-onset grad drop seen in
earlier rope plots. Each scene runs in a fresh subprocess because Genesis does
not support repeated build/backward reliably in one process.
"""
import argparse
import csv
import json
import math
import os
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
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import genesis as gs
from genesis.utils.misc import qd_to_numpy

from grasp_scene import APPROACH_QVEL, CLOSE_QVEL, LIFT_QVEL, TARGET_QVEL
from traj_opt_grasp import _build_scene, _reset_pose_and_settle, _maybe_clamp_fingers

N_APPROACH = 15
N_CLOSE = 20
N_LIFT = 25
HORIZON = N_APPROACH + N_CLOSE + N_LIFT
DEFAULT_CSV = Path("analysis/rope_no_rope_grad_compare.csv")
DEFAULT_PNG = Path("analysis/rope_no_rope_grad_compare.png")


def _phase(step):
    if step < N_APPROACH:
        return "approach"
    if step < N_APPROACH + N_CLOSE:
        return "close"
    return "lift"


def _velocity_program():
    return (
        [APPROACH_QVEL for _ in range(N_APPROACH)]
        + [CLOSE_QVEL for _ in range(N_CLOSE)]
        + [LIFT_QVEL for _ in range(N_LIFT)]
    )


def _contact_count(scene):
    try:
        n_contacts = qd_to_numpy(scene.rigid_solver.collider._collider_state.n_contacts)
        return int(n_contacts.reshape(-1)[0])
    except Exception:
        return None


def _run_scene(scene_kind, sample_every, clamp_fingers):
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    arm_tmp = env_tmp = None
    try:
        scene, arm, arm_tmp, env_tmp = _build_scene(scene_kind)
        _reset_pose_and_settle(scene, arm)

        target = gs.tensor(TARGET_QVEL)
        base_tensors = [gs.tensor(v, requires_grad=True) for v in _velocity_program()]
        used_tensors = list(base_tensors)
        snapshots = []
        contact_counts = []

        for i, v in enumerate(base_tensors):
            used_v = v
            if clamp_fingers and _phase(i) == "close":
                used_v = _maybe_clamp_fingers(arm, v)
            used_tensors[i] = used_v
            arm.set_dofs_velocity(used_v)
            scene.step()
            contact_counts.append(_contact_count(scene))
            if (i + 1) % sample_every == 0:
                snapshots.append(arm.get_dofs_velocity())

        loss = sum((q - target).pow(2).sum() for q in snapshots)
        status = "ok"
        try:
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            status = f"bwd_error:{repr(e)[:160]}"

        grad_norms = []
        grad_nan = 0
        for v in used_tensors:
            if v.grad is None or torch.isnan(v.grad).any():
                grad_norms.append(float("nan"))
                grad_nan += 1
            else:
                grad_norms.append(float(v.grad.norm().item()))

        print("__RNRC__" + json.dumps({
            "scene": scene_kind,
            "status": status,
            "loss": float(loss.item()),
            "sample_every": sample_every,
            "clamp_fingers": clamp_fingers,
            "grad_nan": grad_nan,
            "grad_norms": grad_norms,
            "contact_counts": contact_counts,
        }))
    finally:
        if arm_tmp:
            Path(arm_tmp).unlink(missing_ok=True)
        if env_tmp:
            Path(env_tmp).unlink(missing_ok=True)


def _tail(text, n=14):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _run_worker(script_path, scene_kind, args):
    cmd = [
        "conda", "run", "-n", "genesis", "--no-capture-output",
        "python", script_path,
        "--worker", scene_kind,
        "--sample-every", str(args.sample_every),
    ]
    if args.no_clamp_fingers:
        cmd.append("--no-clamp-fingers")
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=args.timeout,
        env=os.environ.copy(),
    )
    result = next(
        (json.loads(line[len("__RNRC__"):])
         for line in p.stdout.splitlines() if line.startswith("__RNRC__")),
        None,
    )
    if result is None:
        print(f"[worker-error] scene={scene_kind} returncode={p.returncode}")
        if _tail(p.stdout):
            print("[worker-stdout-tail]")
            print(_tail(p.stdout))
        if _tail(p.stderr):
            print("[worker-stderr-tail]")
            print(_tail(p.stderr))
        raise RuntimeError(f"{scene_kind} worker did not return a result")
    return result


def _write_csv(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        for step, grad in enumerate(result["grad_norms"]):
            rows.append({
                "scene": result["scene"],
                "step": step,
                "phase": _phase(step),
                "grad_norm": grad,
                "log10_grad_norm": math.log10(grad) if math.isfinite(grad) and grad > 0 else float("nan"),
                "contact_count": result["contact_counts"][step],
                "loss": result["loss"],
                "status": result["status"],
            })
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene", "step", "phase", "grad_norm", "log10_grad_norm",
                "contact_count", "loss", "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.0, 4.8))
    for result in results:
        xs = list(range(len(result["grad_norms"])))
        ys = [
            math.log10(g) if math.isfinite(g) and g > 0 else float("nan")
            for g in result["grad_norms"]
        ]
        plt.plot(xs, ys, marker="o", markersize=3.4, linewidth=1.3, label=result["scene"])
    plt.axvline(N_APPROACH - 0.5, color="0.35", linestyle="--", linewidth=1.0)
    plt.axvline(N_APPROACH + N_CLOSE - 0.5, color="0.50", linestyle="--", linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("log10(input grad norm)")
    plt.title("Scripted grasp gradient: rope vs no-rope")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def _window_stats(results, start, end):
    by_scene = {r["scene"]: r for r in results}
    rows = []
    for step in range(start, end + 1):
        rope = by_scene["rope"]["grad_norms"][step]
        no_rope = by_scene["no-rope"]["grad_norms"][step]
        ratio = rope / no_rope if math.isfinite(rope) and math.isfinite(no_rope) and no_rope != 0 else float("nan")
        rows.append((step, _phase(step), rope, no_rope, ratio))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("rope", "no-rope"), default=None)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--no-clamp-fingers", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--window-start", type=int, default=12)
    parser.add_argument("--window-end", type=int, default=20)
    args = parser.parse_args()

    if args.worker:
        _run_scene(args.worker, args.sample_every, clamp_fingers=not args.no_clamp_fingers)
        return

    script_path = str(Path(__file__).resolve())
    results = []
    for scene_kind in ("rope", "no-rope"):
        print(f"[rope-no-rope] running {scene_kind}")
        result = _run_worker(script_path, scene_kind, args)
        print(
            f"  scene={scene_kind} loss={result['loss']:.6f} "
            f"status={result['status']} grad_nan={result['grad_nan']}/{HORIZON}"
        )
        results.append(result)

    _write_csv(results, args.csv)
    _write_plot(results, args.plot)

    print("\nClose-onset window:")
    for step, phase, rope, no_rope, ratio in _window_stats(results, args.window_start, args.window_end):
        print(
            f"  step={step:02d} {phase:8s} "
            f"rope={rope:.6g} no_rope={no_rope:.6g} ratio={ratio:.6g}"
        )
    print(f"\nWrote {args.csv}")
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    raise SystemExit(main())
