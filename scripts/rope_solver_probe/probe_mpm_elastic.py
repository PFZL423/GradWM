"""MPM-Elastic rope solver matrix probe.

Produces one datapoint for logs/rope_solver_matrix.csv and one side-camera
video at analysis/rope_solver_probe/mpm_elastic.mp4. MPM backward uses horizon
20 because first contact-aware backward can be very slow on RTX 4060-class GPUs.
"""

import csv
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "genesis-world"))

import imageio.v2 as imageio
import numpy as np
import torch
import trimesh

import genesis as gs


OUT = Path("analysis/rope_solver_probe")
LOG = Path("logs/rope_mpm_elastic.log")
CSV_PATH = Path("logs/rope_solver_matrix.csv")
CSV_FIELDS = [
    "solver",
    "n_particles",
    "fwd_fps",
    "bwd_s",
    "grad_norm",
    "grad_nan",
    "sag_mm",
    "peak_mem_mb",
    "grad_status",
]

ROPE_LEN = 0.45
ROPE_SIZE = (ROPE_LEN, 0.012, 0.012)
TABLE_X = 0.30                              # tables further apart so the longer rope spans them
TABLE_TOP_Z = 0.14
TABLE_SIZE = (0.11, 0.10, 0.02)
ROPE_CENTER_Z = TABLE_TOP_Z + ROPE_SIZE[2] * 0.5 + 0.004
FINGER_SIZE = (0.04, 0.02, 0.02)
FINGER_START = (0.0, 0.0, ROPE_CENTER_Z + ROPE_SIZE[2] * 0.5 + FINGER_SIZE[2] * 0.5 + 0.001)
PUSH_QVEL = (0.0, 0.0, -0.02)               # gentler push so deformation has time to develop
BACKWARD_HORIZON = 20
N_SETTLE = 60
N_PUSH = 80
FPS = 20


def gpu_mem_mb():
    try:
        pid = os.getpid()
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if int(parts[0]) == pid:
                return float(parts[1])
    except Exception:
        pass
    return None


def find_solver_state(scene_state, type_name: str):
    """Return solvers_state[i] whose class.__name__ == type_name, else None."""
    for s in scene_state.solvers_state:
        if s is None or type(s).__name__ != type_name:
            continue
        pos = getattr(s, "pos", None)
        if pos is None or (len(pos.shape) >= 2 and int(pos.shape[1]) > 0):
            return s
    return None


def render_clip(cam, scene, n_frames, drive_fn) -> list[np.ndarray]:
    """drive_fn(t)->None is called before each scene.step()."""
    frames = []
    for t in range(n_frames):
        drive_fn(t)
        scene.step()
        rgb = cam.render()
        if isinstance(rgb, tuple):
            rgb = rgb[0]
        frames.append(np.asarray(rgb))
    return frames


def log_lines(lines):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)


def append_csv(row):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def fmt(value):
    if value is None:
        return "nan"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not math.isfinite(v) else f"{v:.6g}"


def add_tables(scene):
    scene.add_entity(
        gs.morphs.Plane(pos=(0.0, 0.0, -0.001)),
        surface=gs.surfaces.Default(color=(0.18, 0.20, 0.25, 1.0)),
    )
    z = TABLE_TOP_Z - TABLE_SIZE[2] * 0.5
    for x in (-TABLE_X, TABLE_X):
        scene.add_entity(
            gs.morphs.Box(pos=(x, 0.0, z), size=TABLE_SIZE, fixed=True),
            surface=gs.surfaces.Default(color=(0.55, 0.40, 0.25, 1.0)),
        )


def set_finger_qvel(finger, qvel):
    vel = qvel.reshape(1, 3)
    ang = torch.zeros((1, 3), device=qvel.device, dtype=qvel.dtype)
    finger.set_velocity(vel=vel, ang=ang)


def active_positions(state):
    pos = state.pos
    active = getattr(state, "active", None)
    if active is not None and tuple(active.shape) == tuple(pos.shape[:-1]):
        return pos[active == 1]
    return pos.reshape(-1, 3)


def solver_positions_np(scene):
    state = find_solver_state(scene.get_state(), "MPMSolverState")
    if state is None:
        return None
    pos = state.pos.detach().cpu().numpy()
    active = getattr(state, "active", None)
    if active is not None and tuple(active.shape) == tuple(state.pos.shape[:-1]):
        mask = active.detach().cpu().numpy().astype(bool)
        return pos[mask]
    return pos.reshape(-1, 3)


def particle_count(scene):
    state = find_solver_state(scene.get_state(), "MPMSolverState")
    if state is None:
        return 0
    active = getattr(state, "active", None)
    if active is not None and tuple(active.shape) == tuple(state.pos.shape[:-1]):
        return int(active.detach().sum().item())
    return int(state.pos.shape[1])


def sag_mm_from_pos(pos):
    if pos is None or len(pos) < 6:
        return math.nan
    order = np.argsort(pos[:, 0])
    slab = max(3, min(len(order) // 12, len(order) // 3))
    left = pos[order[:slab], 2].mean()
    right = pos[order[-slab:], 2].mean()
    mid = pos[order[len(order) // 2 - slab // 2 : len(order) // 2 + slab // 2 + 1], 2].mean()
    return float(((left + right) * 0.5 - mid) * 1000.0)


def build_scene():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=10, requires_grad=True),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.55, -0.10, -0.05),
            upper_bound=(0.55, 0.10, 0.50),
            grid_density=128,         # default 64; bumped to 128 for thin-geom resolution. 200 caused backward NaN.
            # NB: enable_CPIC=True would help thin-object coupling but is incompatible with requires_grad in 1.1.1.
        ),
        show_viewer=False,
    )
    add_tables(scene)
    rope = scene.add_entity(
        material=gs.materials.MPM.Elastic(rho=100, E=2e2),  # very soft (E=200 vs default 3e5) so gravity sag is visible mid-span
        morph=gs.morphs.Box(pos=(0.0, 0.0, ROPE_CENTER_Z), size=ROPE_SIZE),
        surface=gs.surfaces.Default(color=(0.86, 0.62, 0.20, 1.0)),
        vis_mode="particle",
    )
    finger_obj = tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w").name
    trimesh.creation.box(extents=FINGER_SIZE).export(finger_obj)
    finger = scene.add_entity(
        material=gs.materials.Tool(friction=8.0),
        morph=gs.morphs.Mesh(file=finger_obj, pos=FINGER_START, scale=max(FINGER_SIZE)),  # ToolEntity normalizes mesh to unit cube; scale=max_extent → real size
        surface=gs.surfaces.Default(color=(0.55, 0.57, 0.60, 1.0)),
    )
    cam = scene.add_camera(
        res=(640, 480),
        pos=(0.0, 1.0, 0.45),
        lookat=(0.0, 0.0, 0.14),
        up=(0.0, 0.0, 1.0),
        fov=42,
    )
    scene.build(n_envs=1)
    scene.reset()
    return scene, rope, finger, cam


def run_backward_once(label, scene, finger):
    scene.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    mem_samples = [gpu_mem_mb()]
    qvel = gs.tensor(PUSH_QVEL, requires_grad=True)
    loss = None
    status = "ok"
    t_fwd = 0.0
    t_bwd = math.nan
    try:
        t0 = time.time()
        for _ in range(BACKWARD_HORIZON):
            set_finger_qvel(finger, qvel)
            scene.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fwd = time.time() - t0
        mem_samples.append(gpu_mem_mb())
        state = find_solver_state(scene.get_state(), "MPMSolverState")
        if state is None:
            status = "no_solver_state"
        else:
            pos = active_positions(state)
            log_lines([f"[{label}] MPMSolverState.pos shape={tuple(state.pos.shape)} rg={state.pos.requires_grad}"])
            if pos.numel() == 0:
                status = "no_active_particles"
            elif not pos.requires_grad:
                status = "state_pos_rg_false"
            else:
                loss = pos[:, 2].mean()
                t0 = time.time()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_bwd = time.time() - t0
                mem_samples.append(gpu_mem_mb())
    except gs.GenesisException as e:
        msg = str(e)
        status = msg.replace("Nan grad in qpos or dofs_vel found at step ", "nan@step")
    except Exception as e:
        status = f"error:{repr(e)[:120]}"

    grad = qvel.grad
    if grad is None:
        grad_norm = math.nan
        grad_nan_frac = 1.0
        grad_nan_count = 1
        grad_total = 1
        grad_status = status if status != "ok" else "no_contact_path"
    else:
        bad = torch.isnan(grad) | torch.isinf(grad)
        grad_nan_count = int(bad.sum().item())
        grad_total = int(grad.numel())
        grad_nan_frac = float(grad_nan_count) / float(grad_total)
        grad_norm = float("nan") if bad.any() else float(grad.norm().item())
        if status != "ok":
            grad_status = status
            grad_nan_frac = max(grad_nan_frac, 1.0)
        elif grad_norm <= 1e-12:
            grad_status = "no_contact_path"
            grad_nan_frac = 1.0
        else:
            grad_status = "ok"
    peak_mem = max([m for m in mem_samples if m is not None], default=math.nan)
    fwd_fps = BACKWARD_HORIZON / t_fwd if t_fwd > 0 else math.nan
    log_lines(
        [
            f"[{label}] status={status} fwd_fps={fmt(fwd_fps)} bwd_s={fmt(t_bwd)} "
            f"grad_norm={fmt(grad_norm)} grad_nan={grad_nan_count}/{grad_total} "
            f"csv_grad_nan={fmt(grad_nan_frac)} peak_mem_mb={fmt(peak_mem)}",
        ]
    )
    return {
        "status": status,
        "grad_status": grad_status,
        "fwd_fps": fwd_fps,
        "bwd_s": t_bwd,
        "grad_norm": grad_norm,
        "grad_nan": grad_nan_frac,
        "peak_mem_mb": peak_mem,
        "loss": math.nan if loss is None else float(loss.detach().cpu().item()),
    }


def measure_sag(scene, finger):
    scene.reset()
    zero = torch.zeros(3, dtype=torch.float32)
    try:
        for _ in range(N_SETTLE):
            set_finger_qvel(finger, zero)
            scene.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as e:
        log_lines([f"[sag] failed: {repr(e)[:120]}"])
        return math.nan
    return sag_mm_from_pos(solver_positions_np(scene))


def render_video(scene, finger, cam):
    OUT.mkdir(parents=True, exist_ok=True)
    scene.reset()
    push = torch.tensor(PUSH_QVEL, dtype=torch.float32)
    zero = torch.zeros(3, dtype=torch.float32)

    def drive(t):
        set_finger_qvel(finger, zero if t < N_SETTLE else push)

    frames = render_clip(cam, scene, N_SETTLE + N_PUSH, drive)
    path = OUT / "mpm_elastic.mp4"
    with imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    log_lines([f"[render] wrote {path} frames={len(frames)} fps={FPS}"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log_lines([f"--- mpm-elastic matrix probe ts={time.strftime('%Y-%m-%dT%H:%M:%S')} ---"])

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene, rope, finger, cam = build_scene()
    n_particles = particle_count(scene)
    log_lines([f"[probe] rope n_particles={n_particles} entity_n={getattr(rope, 'n_particles', '?')}"])

    warmup = run_backward_once("warmup", scene, finger)
    steady = run_backward_once("steady", scene, finger)
    sag_mm = measure_sag(scene, finger)
    render_video(scene, finger, cam)
    peak_mem = max(
        [m for m in (warmup["peak_mem_mb"], steady["peak_mem_mb"], gpu_mem_mb()) if m is not None],
        default=math.nan,
    )

    append_csv(
        {
            "solver": "MPM-Elastic",
            "n_particles": n_particles,
            "fwd_fps": fmt(steady["fwd_fps"]),
            "bwd_s": fmt(steady["bwd_s"]),
            "grad_norm": fmt(steady["grad_norm"]),
            "grad_nan": fmt(steady["grad_nan"]),
            "sag_mm": fmt(sag_mm),
            "peak_mem_mb": fmt(peak_mem),
            "grad_status": steady["grad_status"],
        }
    )
    log_lines([f"[csv] appended {CSV_PATH}", "--- mpm-elastic matrix probe done ---", ""])


if __name__ == "__main__":
    raise SystemExit(main())
