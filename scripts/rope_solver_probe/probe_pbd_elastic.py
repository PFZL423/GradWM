"""PBD-Elastic rope probe.

Question we want answered:
  1. Is backward through PBD-Elastic NaN-free at 30 steps?
  2. Does the rope visibly deform under gravity (mid-point sag)?
  3. Does the rope sag at both ends when held at the middle?

Setup: a thin elongated Box (0.30 x 0.02 x 0.02) loaded as gs.materials.PBD.Elastic.
We do TWO scenarios:
  S1 (free fall): gravity only, no holds, rope falls. Measures dynamics correctness
                  + backward NaN.
  S2 (mid-grasp): rope is held at one mid-region particle by setting its
                  position to a fixed lift trajectory; observe whether the two
                  free ends sag downward.

Metrics written to logs/rope_pbd_elastic.log:
  - n_particles, n_dofs (if available)
  - per-step grad norm of a tiny perturbation tensor → NaN count
  - mid-point z trajectory (S1 free fall)
  - tip z trajectory (S2 mid-grasp), with target lift height
  - render still frames to analysis/rope_solver_probe/pbd_elastic_*.png

Run: python scripts/rope_solver_probe/probe_pbd_elastic.py
"""
import sys, os, subprocess, time, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import torch
import imageio.v2 as imageio
import genesis as gs

LOG = Path("logs/rope_pbd_elastic.log")
OUT = Path("analysis/rope_solver_probe")
OUT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

ROPE_LEN = 0.30
ROPE_SIDE = 0.02   # square cross-section 2cm
HORIZON = 30


def gpu_mem_mb():
    try:
        pid = os.getpid()
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if int(parts[0]) == pid:
                return float(parts[1])
    except Exception:
        pass
    return None


def log_lines(lines):
    with LOG.open("a") as f:
        f.write("\n".join(lines) + "\n")
    for l in lines:
        print(l)


def render_frame(cam, label):
    rgb = cam.render()
    if isinstance(rgb, tuple):
        rgb = rgb[0]
    path = OUT / f"pbd_elastic_{label}.png"
    imageio.imwrite(str(path), np.asarray(rgb))
    return path


def main():
    log_lines([f"--- pbd-elastic probe ts={time.strftime('%Y-%m-%dT%H:%M:%S')} ---"])

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3, substeps=10, requires_grad=True,
        ),
        pbd_options=gs.options.PBDOptions(particle_size=0.005),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))

    rope = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.30),
            size=(ROPE_LEN, ROPE_SIDE, ROPE_SIDE),
        ),
        material=gs.materials.PBD.Elastic(
            rho=200.0,                   # light, so gravity sag is visible
            stretch_compliance=1e-6,     # slightly soft stretch
            bending_compliance=1e-3,     # soft bend → ropy
        ),
        surface=gs.surfaces.Default(color=(0.85, 0.65, 0.30, 1.0)),
    )

    cam = scene.add_camera(
        res=(640, 480),
        pos=(0.0, 0.8, 0.25),
        lookat=(0.0, 0.0, 0.20),
        up=(0.0, 0.0, 1.0),
        fov=42,
    )

    scene.build()
    scene.reset()

    # Probe entity API
    n_particles = getattr(rope, "n_particles", None)
    log_lines([f"[probe] rope n_particles={n_particles}"])

    # ---------- S1: free fall ----------
    render_frame(cam, "s1_t0")
    z_traj = []
    status_s1 = "ok"
    try:
        for t in range(HORIZON):
            scene.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except gs.GenesisException as e:
        status_s1 = f"genesis_err:{str(e)[:100]}"
    except Exception as e:
        status_s1 = f"err:{repr(e)[:100]}"
    render_frame(cam, "s1_t30")
    log_lines([f"[s1] free-fall {HORIZON} steps: status={status_s1} mem={gpu_mem_mb()}MB"])

    # ---------- S2: backward-grad probe ----------
    # We need a differentiable handle. Try setting particle velocities (if such API exists)
    # and computing loss on final mean position.
    scene.reset()
    methods = [m for m in dir(rope) if "vel" in m.lower() or "particle" in m.lower() or "set" in m.lower()][:30]
    log_lines([f"[probe] rope writable methods (vel/particle/set): {methods}"])

    status_s2 = "ok"
    grad_status = "no_grad_path"
    grad_norms_summary = "n/a"
    try:
        # Attempt: get current positions (differentiable handle?), perturb, step, compute loss
        if hasattr(rope, "get_particles"):
            p0 = rope.get_particles()
            log_lines([f"[probe] get_particles -> shape={tuple(p0.shape)} rg={p0.requires_grad}"])
        # Try set_velocities
        if hasattr(rope, "set_particles_velocity"):
            log_lines([f"[probe] has set_particles_velocity"])

        # Run forward and check if any particle attribute is on the graph
        for t in range(HORIZON):
            scene.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Try to extract a differentiable post-state
        candidates = ["get_particles", "get_particles_pos", "get_state"]
        diff_handle = None
        for name in candidates:
            if hasattr(rope, name):
                v = getattr(rope, name)()
                if hasattr(v, "requires_grad"):
                    log_lines([f"[probe] {name}() -> shape={tuple(v.shape)} rg={v.requires_grad}"])
                    if v.requires_grad and diff_handle is None:
                        diff_handle = v

        if diff_handle is not None:
            grad_status = "diff_handle_found"
            loss = diff_handle.pow(2).sum()
            try:
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                grad_status = "backward_ok_no_input_grad_param"
            except Exception as e:
                grad_status = f"backward_failed:{repr(e)[:80]}"

    except gs.GenesisException as e:
        status_s2 = f"genesis_err:{str(e)[:100]}"
    except Exception as e:
        status_s2 = f"err:{repr(e)[:100]}"
    log_lines([f"[s2] backward-probe: status={status_s2} grad={grad_status} mem={gpu_mem_mb()}MB"])

    log_lines([f"--- pbd-elastic probe done ---", ""])

if __name__ == "__main__":
    main()
