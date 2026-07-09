"""Trajectory optimization for a simple rigid-box push contact task.

The loss is directly on the free object's differentiable position, so this is
a cleaner rigid-contact comparison than arm-qvel-only rope losses.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import genesis as gs

OUT_CSV = Path("analysis/box_push_loss.csv")
OUT_PNG = Path("analysis/box_push_loss.png")
OUT_QVEL = Path("analysis/box_push_qvel.json")


def _make_initial_velocities(horizon, mode, init_vx):
    if mode == "zero":
        return torch.zeros((horizon, 6), dtype=torch.float32)
    if mode == "push":
        return torch.tensor([[init_vx, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(horizon)], dtype=torch.float32)
    raise ValueError(mode)


def _flat_pos(pos):
    return pos.reshape(-1, 3)[0]


def _worker_mode(request_path):
    payload = json.loads(Path(request_path).read_text())
    velocities = payload["velocities"]
    loss_mode = payload["loss_mode"]
    sample_every = int(payload["sample_every"])
    control_weight = float(payload["control_weight"])
    smooth_weight = float(payload["smooth_weight"])

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    target = gs.tensor(payload["target"])

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=4, substeps_local=4, requires_grad=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))

    pusher = scene.add_entity(gs.morphs.Box(size=(0.04, 0.05, 0.04), pos=(0.0, 0.0, 0.025)))
    obj = scene.add_entity(gs.morphs.Box(size=(0.04, 0.05, 0.04), pos=(0.045, 0.0, 0.025)))
    scene.build()
    scene.reset()

    v_tensors = [gs.tensor(v, requires_grad=True) for v in velocities]
    obj_pos_samples = []
    for i, v in enumerate(v_tensors):
        pusher.set_dofs_velocity(v)
        scene.step()
        if loss_mode == "trajectory" and ((i + 1) % sample_every == 0):
            obj_pos_samples.append(_flat_pos(obj.get_state().pos))

    final_pos = _flat_pos(obj.get_state().pos)
    if loss_mode == "final":
        loss = (final_pos - target).pow(2).sum()
    elif loss_mode == "trajectory":
        loss = sum((pos - target).pow(2).sum() for pos in obj_pos_samples)
    else:
        raise ValueError(loss_mode)

    if control_weight:
        loss = loss + control_weight * sum(v.pow(2).sum() for v in v_tensors)
    if smooth_weight and len(v_tensors) > 1:
        loss = loss + smooth_weight * sum(
            (v_tensors[i] - v_tensors[i - 1]).pow(2).sum()
            for i in range(1, len(v_tensors))
        )

    status = "ok"
    try:
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as e:
        status = f"bwd_error:{repr(e)[:160]}"

    grads = []
    grad_nan = 0
    grad_norm_sq = 0.0
    for v in v_tensors:
        if v.grad is None or torch.isnan(v.grad).any():
            grad_nan += 1
            g = torch.zeros(6, dtype=torch.float32)
        else:
            g = v.grad.detach().float().cpu()
            grad_norm_sq += float((g * g).sum().item())
        grads.append(g.tolist())

    print("__BOXPUSH__" + json.dumps({
        "status": status,
        "loss": float(loss.item()),
        "grad": grads,
        "grad_nan": grad_nan,
        "grad_norm": math.sqrt(grad_norm_sq),
        "final_obj_pos": [float(x) for x in final_pos.detach().cpu().tolist()],
    }))


def _tail(text, n=12):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _run_worker(script_path, velocities, args):
    with tempfile.NamedTemporaryFile(prefix="box_push_req_", suffix=".json", delete=False, mode="w") as f:
        json.dump({
            "velocities": velocities.detach().cpu().tolist(),
            "loss_mode": args.loss,
            "sample_every": args.sample_every,
            "control_weight": args.control_weight,
            "smooth_weight": args.smooth_weight,
            "target": [args.target_x, args.target_y, args.target_z],
        }, f)
        request_path = f.name
    try:
        cmd = [
            "conda", "run", "-n", "genesis", "--no-capture-output",
            "python", script_path, "--worker", request_path,
        ]
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=os.environ.copy(),
        )
    finally:
        Path(request_path).unlink(missing_ok=True)

    result = next(
        (json.loads(line[len("__BOXPUSH__"):])
         for line in p.stdout.splitlines() if line.startswith("__BOXPUSH__")),
        None,
    )
    if result is None:
        print(f"[worker-error] returncode={p.returncode}")
        if _tail(p.stdout):
            print("[worker-stdout-tail]")
            print(_tail(p.stdout))
        if _tail(p.stderr):
            print("[worker-stderr-tail]")
            print(_tail(p.stderr))
        raise RuntimeError("box push worker did not return a result")
    return result


def _write_outputs(rows, velocities, csv_path, png_path, qvel_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["iter", "loss", "grad_norm", "grad_nan", "status", "final_obj_pos"],
        )
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(6.5, 4.0))
    plt.plot([r["iter"] for r in rows], [r["loss"] for r in rows], marker="o")
    plt.xlabel("Adam iteration")
    plt.ylabel("object position loss")
    plt.title("Rigid box push trajectory optimization")
    plt.grid(True, alpha=0.3)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close()

    qvel_path.write_text(json.dumps({
        "shape": list(velocities.shape),
        "qvel": velocities.detach().cpu().tolist(),
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=None)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--init", choices=("push", "zero"), default="push")
    parser.add_argument("--init-vx", type=float, default=0.40)
    parser.add_argument("--loss", choices=("final", "trajectory"), default="final")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--target-x", type=float, default=0.16)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.025)
    parser.add_argument("--max-vel", type=float, default=1.5)
    parser.add_argument("--control-weight", type=float, default=0.0)
    parser.add_argument("--smooth-weight", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--csv", type=Path, default=OUT_CSV)
    parser.add_argument("--plot", type=Path, default=OUT_PNG)
    parser.add_argument("--qvel-out", type=Path, default=OUT_QVEL)
    args = parser.parse_args()

    if args.worker:
        _worker_mode(args.worker)
        return

    script_path = str(Path(__file__).resolve())
    velocities = _make_initial_velocities(args.horizon, args.init, args.init_vx)
    velocities.requires_grad_(True)
    opt = torch.optim.Adam([velocities], lr=args.lr)
    rows = []

    print(
        f"Box push opt: H={args.horizon} iters={args.iters} lr={args.lr} "
        f"init={args.init} init_vx={args.init_vx} loss={args.loss} "
        f"target=({args.target_x},{args.target_y},{args.target_z})"
    )
    for it in range(args.iters + 1):
        result = _run_worker(script_path, velocities.detach(), args)
        row = {
            "iter": it,
            "loss": result["loss"],
            "grad_norm": result["grad_norm"],
            "grad_nan": result["grad_nan"],
            "status": result["status"],
            "final_obj_pos": result["final_obj_pos"],
        }
        rows.append(row)
        print(
            f"  iter={it:03d} loss={row['loss']:.8f} grad_norm={row['grad_norm']:.8f} "
            f"grad_nan={row['grad_nan']}/{args.horizon} pos={row['final_obj_pos']} status={row['status']}"
        )
        if it == args.iters:
            break
        if result["status"] != "ok" or result["grad_nan"]:
            print("  stopping: worker returned non-ok status or NaN gradients")
            break

        opt.zero_grad(set_to_none=True)
        velocities.grad = torch.tensor(result["grad"], dtype=torch.float32)
        opt.step()
        with torch.no_grad():
            velocities.clamp_(-args.max_vel, args.max_vel)

    _write_outputs(rows, velocities.detach(), args.csv, args.plot, args.qvel_out)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.plot}")
    print(f"Wrote {args.qvel_out}")


if __name__ == "__main__":
    raise SystemExit(main())
