"""Single-anchor, many-direction FD-vs-autograd diagnostic for rigid box push.

Picks one contact-frame anchor in the box-push scene, samples many
perturbation directions and scales around the pusher's anchor-step velocity,
and compares the finite-difference (FD) directional derivative of an
object-position probe scalar against Genesis autograd's directional
derivative (dot of the analytic gradient with the same direction).

Produces a scatter plot (FD vs analytic) and a CSV of per-(direction, scale)
samples. Unlike scripts/legacy/grad_fd_check.py (per-DOF axis-aligned FD on the
rope/grasp scene at a few hardcoded steps), this sweeps many directions and
scales on the cleaner box-push contact scene to visualize how FD/analytic
agreement varies with perturbation direction and magnitude.

Each run does exactly one gs.init() / scene.build() / .backward() inside a
single subprocess worker (Genesis constraint: no multiple backward per
process); everything else replays via plain scene.reset() (no state=...).
"""
import argparse
import csv
import json
import math
import os
import subprocess
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

DEFAULT_HORIZON = 40
DEFAULT_INIT_VX = 0.40
DEFAULT_SCALES = "1e-5,3e-5,1e-4,3e-4,1e-3"
DEFAULT_CSV = Path("analysis/2026-07-09_arx_pusher/single_frame_fd_diag.csv")
DEFAULT_PNG = Path("analysis/2026-07-09_arx_pusher/single_frame_fd_diag.png")


# ---------------------------------------------------------------------------
# Worker-side code (only imports torch/genesis inside the conda `genesis` env)
# ---------------------------------------------------------------------------

def _make_velocity_program(horizon, init_vx):
    return [[init_vx, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(horizon)]


def _build_scene(gs):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=4, substeps_local=4, requires_grad=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))
    pusher = scene.add_entity(gs.morphs.Box(size=(0.04, 0.05, 0.04), pos=(0.0, 0.0, 0.025)))
    obj = scene.add_entity(gs.morphs.Box(size=(0.04, 0.05, 0.04), pos=(0.045, 0.0, 0.025)))
    scene.build()
    return scene, pusher, obj


def _contact_summary(scene):
    from genesis.utils.misc import qd_to_numpy
    collider_state = scene.rigid_solver.collider._collider_state
    n_contacts = qd_to_numpy(collider_state.n_contacts)
    n = int(n_contacts.reshape(-1)[0])
    return {"n": n}


def _find_anchor_step(scene, pusher, velocities, gs):
    scene.reset()
    for i, v in enumerate(velocities):
        pusher.set_dofs_velocity(gs.tensor(v))
        scene.step()
        if _contact_summary(scene)["n"] > 0:
            return i
    return None


def _flat_pos(pos):
    return pos.reshape(-1, 3)[0]


def _probe_scalar(obj, loss_kind, target, torch):
    pos = _flat_pos(obj.get_state().pos)
    if loss_kind == "disp_x":
        return pos[0]
    if loss_kind == "mse_target":
        t = target if hasattr(target, "shape") else None
        return (pos - t).pow(2).sum()
    raise ValueError(loss_kind)


def _rollout_analytic(scene, pusher, obj, velocities, anchor_step, loss_kind, target, gs, torch):
    scene.reset()
    anchor_tensor = None
    for i, v in enumerate(velocities):
        rg = (i == anchor_step)
        vt = gs.tensor(v, requires_grad=rg)
        if rg:
            anchor_tensor = vt
        pusher.set_dofs_velocity(vt)
        scene.step()

    probe = _probe_scalar(obj, loss_kind, target, torch)
    status = "ok"
    grad6 = None
    try:
        probe.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if anchor_tensor.grad is None or torch.isnan(anchor_tensor.grad).any():
            status = "bwd_nan"
        else:
            grad6 = [float(x) for x in anchor_tensor.grad.detach().cpu().tolist()]
    except Exception as e:
        msg = repr(e)
        marker = "Nan grad in qpos or dofs_vel found at step "
        if marker in str(e):
            idx = str(e).split(marker, 1)[1].strip().split()[0].rstrip(".,:")
            status = f"bwd_nan_at_step:{idx}"
        else:
            status = f"bwd_error:{msg[:160]}"

    return status, float(probe.item()), grad6


def _rollout_forward(scene, pusher, obj, velocities, anchor_step, perturb_vec, loss_kind, target, gs, torch):
    scene.reset()
    for i, v in enumerate(velocities):
        vv = list(v)
        if i == anchor_step:
            vv = [vv[k] + perturb_vec[k] for k in range(6)]
        pusher.set_dofs_velocity(gs.tensor(vv))
        scene.step()
    probe = _probe_scalar(obj, loss_kind, target, torch)
    return float(probe.detach().item())


def _worker_mode(request_path):
    import torch
    import genesis as gs

    payload = json.loads(Path(request_path).read_text())
    horizon = int(payload["horizon"])
    init_vx = float(payload["init_vx"])
    anchor_step_req = payload.get("anchor_step")
    loss_kind = payload["loss_kind"]
    target_list = payload.get("target")
    directions = payload["directions"]
    scales = payload["scales"]

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene, pusher, obj = _build_scene(gs)
    velocities = _make_velocity_program(horizon, init_vx)
    target = gs.tensor(target_list) if (loss_kind == "mse_target" and target_list is not None) else None

    if anchor_step_req is None:
        anchor_step = _find_anchor_step(scene, pusher, velocities, gs)
    else:
        anchor_step = int(anchor_step_req)

    if anchor_step is None:
        print("__FDDIAG__" + json.dumps({"status": "no_contact", "anchor_step": None, "rows": []}))
        return

    status, analytic_probe_value, analytic_grad = _rollout_analytic(
        scene, pusher, obj, velocities, anchor_step, loss_kind, target, gs, torch
    )

    rows = []
    if status == "ok" and analytic_grad is not None:
        for d_idx, direction in enumerate(directions):
            analytic_directional = sum(analytic_grad[k] * direction[k] for k in range(6))
            for scale in scales:
                perturb_plus = [scale * direction[k] for k in range(6)]
                perturb_minus = [-scale * direction[k] for k in range(6)]
                val_plus = _rollout_forward(scene, pusher, obj, velocities, anchor_step, perturb_plus, loss_kind, target, gs, torch)
                val_minus = _rollout_forward(scene, pusher, obj, velocities, anchor_step, perturb_minus, loss_kind, target, gs, torch)
                fd_directional = (val_plus - val_minus) / (2 * scale)
                abs_error = abs(fd_directional - analytic_directional)
                rel_error = abs_error / (abs(fd_directional) + 1e-12)
                rows.append({
                    "direction_idx": d_idx,
                    "direction": direction,
                    "scale": scale,
                    "val_plus": val_plus,
                    "val_minus": val_minus,
                    "fd_directional": fd_directional,
                    "analytic_directional": analytic_directional,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                })

    print("__FDDIAG__" + json.dumps({
        "status": status,
        "anchor_step": anchor_step,
        "analytic_grad": analytic_grad,
        "analytic_probe_value": analytic_probe_value,
        "rows": rows,
    }))


# ---------------------------------------------------------------------------
# Orchestrator-side code (no torch/genesis import required here)
# ---------------------------------------------------------------------------

def _sample_directions(num_directions, subspace, seed, random_directions):
    import numpy as np
    rng = np.random.default_rng(seed)
    dirs = []
    if subspace == "vxvy":
        axis_dirs = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        if random_directions:
            thetas = rng.uniform(0, 2 * math.pi, size=num_directions)
        else:
            thetas = np.linspace(0, 2 * math.pi, num_directions, endpoint=False)
        seen = set()
        for vx, vy in axis_dirs:
            key = (round(vx, 6), round(vy, 6))
            if key not in seen:
                seen.add(key)
                dirs.append([vx, vy, 0.0, 0.0, 0.0, 0.0])
        for theta in thetas:
            vx, vy = math.cos(theta), math.sin(theta)
            key = (round(vx, 6), round(vy, 6))
            if key in seen:
                continue
            seen.add(key)
            dirs.append([vx, vy, 0.0, 0.0, 0.0, 0.0])
        dirs = dirs[:max(num_directions, len(axis_dirs))]
    elif subspace == "full6":
        for _ in range(num_directions):
            v = rng.normal(size=6)
            v = v / (np.linalg.norm(v) + 1e-12)
            dirs.append(v.tolist())
    else:
        raise ValueError(subspace)
    return dirs


def _tail(text, n=12):
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _run_worker(script_path, request_dict, timeout):
    with tempfile.NamedTemporaryFile(prefix="fd_diag_req_", suffix=".json", delete=False, mode="w") as f:
        json.dump(request_dict, f)
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
            timeout=timeout,
            env=os.environ.copy(),
        )
    finally:
        Path(request_path).unlink(missing_ok=True)

    result = next(
        (json.loads(line[len("__FDDIAG__"):])
         for line in p.stdout.splitlines() if line.startswith("__FDDIAG__")),
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
        raise RuntimeError("single_frame_fd_diag worker did not return a result")
    return result


def _tag_by_rel_error(rel_error):
    if rel_error < 0.1:
        return "GOOD"
    if rel_error < 0.5:
        return "OK"
    if rel_error < 1.0:
        return "WEAK"
    return "BAD"


def _write_csv(rows, anchor_step, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "anchor_step", "direction_idx", "dir_vx", "dir_vy", "dir_vz",
        "dir_wx", "dir_wy", "dir_wz", "scale", "fd_directional",
        "analytic_directional", "abs_error", "rel_error", "tag",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            d = r["direction"]
            writer.writerow({
                "anchor_step": anchor_step,
                "direction_idx": r["direction_idx"],
                "dir_vx": d[0], "dir_vy": d[1], "dir_vz": d[2],
                "dir_wx": d[3], "dir_wy": d[4], "dir_wz": d[5],
                "scale": r["scale"],
                "fd_directional": r["fd_directional"],
                "analytic_directional": r["analytic_directional"],
                "abs_error": r["abs_error"],
                "rel_error": r["rel_error"],
                "tag": r["tag"],
            })


def _write_scatter_png(rows, anchor_step, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = sorted({r["scale"] for r in rows})
    cmap = plt.get_cmap("viridis")
    color_for_scale = {s: cmap(i / max(1, len(scales) - 1)) for i, s in enumerate(scales)}

    plt.figure(figsize=(6.5, 6.0))
    xs_all = [r["fd_directional"] for r in rows]
    ys_all = [r["analytic_directional"] for r in rows]

    for s in scales:
        xs = [r["fd_directional"] for r in rows if r["scale"] == s]
        ys = [r["analytic_directional"] for r in rows if r["scale"] == s]
        plt.scatter(xs, ys, color=color_for_scale[s], label=f"scale={s:g}", alpha=0.8, s=30)

    highlight_x, highlight_y = [], []
    for r in rows:
        if abs(r["analytic_directional"]) < 1e-6 and abs(r["fd_directional"]) > 0.05:
            highlight_x.append(r["fd_directional"])
            highlight_y.append(r["analytic_directional"])
    if highlight_x:
        plt.scatter(highlight_x, highlight_y, facecolors="none", edgecolors="red", s=90,
                    linewidths=1.5, label=f"|analytic|~0 & |FD|>0.05 (n={len(highlight_x)})")

    if xs_all:
        lo, hi = min(xs_all + ys_all), max(xs_all + ys_all)
        pad = 0.05 * (hi - lo + 1e-9)
        plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", label="y = x")

    plt.xlabel("FD directional derivative")
    plt.ylabel("Analytic (autograd) directional derivative")
    plt.title(f"Single-frame FD vs autograd @ anchor_step={anchor_step} (box push)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--init-vx", type=float, default=DEFAULT_INIT_VX)
    parser.add_argument("--anchor-step", type=int, default=None)
    parser.add_argument("--loss-kind", choices=("disp_x", "mse_target"), default="disp_x")
    parser.add_argument("--target-x", type=float, default=0.16)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.025)
    parser.add_argument("--num-directions", type=int, default=16)
    parser.add_argument("--direction-subspace", choices=("vxvy", "full6"), default="vxvy")
    parser.add_argument("--random-directions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PNG)
    args = parser.parse_args()

    if args.worker:
        _worker_mode(args.worker)
        return

    script_path = str(Path(__file__).resolve())
    directions = _sample_directions(args.num_directions, args.direction_subspace, args.seed, args.random_directions)
    scales = [float(s) for s in args.scales.split(",") if s.strip()]

    request = {
        "horizon": args.horizon,
        "init_vx": args.init_vx,
        "anchor_step": args.anchor_step,
        "loss_kind": args.loss_kind,
        "target": [args.target_x, args.target_y, args.target_z],
        "directions": directions,
        "scales": scales,
    }

    print(
        f"Single-frame FD diag: H={args.horizon} init_vx={args.init_vx} "
        f"anchor_step={'auto' if args.anchor_step is None else args.anchor_step} "
        f"loss_kind={args.loss_kind} directions={len(directions)} scales={scales}"
    )
    result = _run_worker(script_path, request, args.timeout)

    if result["status"] == "no_contact":
        print("No contact detected within horizon. Try increasing --horizon or --init-vx.")
        return

    if result["status"] != "ok":
        print(f"Worker returned non-ok status: {result['status']}")
        return

    rows = result["rows"]
    for r in rows:
        r["tag"] = _tag_by_rel_error(r["rel_error"])

    anchor_step = result["anchor_step"]
    _write_csv(rows, anchor_step, args.csv)
    _write_scatter_png(rows, anchor_step, args.plot)

    analytic_grad = result["analytic_grad"]
    analytic_norm = math.sqrt(sum(g * g for g in analytic_grad)) if analytic_grad else float("nan")

    print(f"\nAnchor step: {anchor_step}")
    print(f"Analytic gradient (6-vec): {analytic_grad}")
    print(f"Analytic gradient norm: {analytic_norm:.8f}  near-zero: {analytic_norm < 1e-6}")

    print("\nSummary by scale:")
    for s in scales:
        subset = [r for r in rows if r["scale"] == s]
        if not subset:
            continue
        errs = [r["rel_error"] for r in subset]
        mean_err = sum(errs) / len(errs)
        sorted_errs = sorted(errs)
        median_err = sorted_errs[len(sorted_errs) // 2]
        counts = {"GOOD": 0, "OK": 0, "WEAK": 0, "BAD": 0}
        for r in subset:
            counts[r["tag"]] += 1
        print(
            f"  scale={s:.0e}  mean_rel_err={mean_err:.4f}  median_rel_err={median_err:.4f}  "
            f"GOOD={counts['GOOD']} OK={counts['OK']} WEAK={counts['WEAK']} BAD={counts['BAD']}"
        )

    print(f"\nWrote {args.csv}")
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    raise SystemExit(main())
