"""FEM-Elastic rope probe.

Uses gs.materials.FEM.Elastic on a Box morph; Genesis tetrahedralizes it.
Same questions as the PBD probes: does it deform realistically + can backward
flow through it cleanly?

FEM has differentiable input/output (set_pos/set_vel + get_state hooks
requires_grad-aware tensors per fem_entity.py lines 675-681).

Run: python scripts/rope_solver_probe/probe_fem_elastic.py
"""
import sys, os, subprocess, time, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import imageio.v2 as imageio
import torch
import genesis as gs

LOG = Path("logs/rope_fem_elastic.log")
OUT = Path("analysis/rope_solver_probe")
OUT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

ROPE_LEN = 0.30
ROPE_SIDE = 0.015
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
    path = OUT / f"fem_elastic_{label}.png"
    imageio.imwrite(str(path), np.asarray(rgb))
    return path


def main():
    log_lines([f"--- fem-elastic probe ts={time.strftime('%Y-%m-%dT%H:%M:%S')} ---"])

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3, substeps=4, substeps_local=4, requires_grad=True,
        ),
        fem_options=gs.options.FEMOptions(use_implicit_solver=False),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))

    rope = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.30),
            size=(ROPE_LEN, ROPE_SIDE, ROPE_SIDE),
        ),
        material=gs.materials.FEM.Elastic(
            model="stable_neohookean",
            E=5e3,                 # softer than default 1e5
            nu=0.4,
            rho=200.0,
        ),
        surface=gs.surfaces.Default(color=(0.85, 0.65, 0.30, 1.0)),
    )

    cam = scene.add_camera(
        res=(640, 480),
        pos=(0.0, 0.8, 0.20),
        lookat=(0.0, 0.0, 0.20),
        up=(0.0, 0.0, 1.0),
        fov=42,
    )

    scene.build()
    scene.reset()

    n_v = getattr(rope, "n_vertices", None) or getattr(rope, "n_particles", None)
    log_lines([f"[probe] rope n_vertices={n_v}"])

    state0 = rope.get_state()
    log_lines([f"[probe] state attrs: {[a for a in dir(state0) if not a.startswith('_')][:20]}"])
    if hasattr(state0, "pos"):
        log_lines([f"[probe] state.pos shape={tuple(state0.pos.shape)} rg={state0.pos.requires_grad}"])
    if hasattr(state0, "vel"):
        log_lines([f"[probe] state.vel shape={tuple(state0.vel.shape)} rg={state0.vel.requires_grad}"])

    # ---------- S1: free fall ----------
    render_frame(cam, "s1_t0")
    status = "ok"
    try:
        for t in range(HORIZON):
            scene.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except gs.GenesisException as e:
        status = f"genesis_err:{str(e)[:100]}"
    except Exception as e:
        status = f"err:{repr(e)[:100]}"
    render_frame(cam, "s1_t30")
    log_lines([f"[s1] free-fall: status={status} mem={gpu_mem_mb()}MB"])

    if status == "ok":
        s1 = rope.get_state()
        if hasattr(s1, "pos"):
            pos = s1.pos.detach().cpu().numpy()
            if pos.ndim == 3:
                pos = pos[0]   # squeeze env dim
            x_ords = np.argsort(pos[:, 0])
            n_slab = max(3, len(x_ords) // 30)
            left_id = x_ords[:n_slab]
            right_id = x_ords[-n_slab:]
            mid_id = x_ords[len(x_ords)//2 - n_slab//2 : len(x_ords)//2 + n_slab//2 + 1]
            z_left  = pos[left_id, 2].mean()
            z_right = pos[right_id, 2].mean()
            z_mid   = pos[mid_id, 2].mean()
            log_lines([
                f"[s1] z_left={z_left:.4f}, z_mid={z_mid:.4f}, z_right={z_right:.4f}",
                f"[s1] sag (ends_avg - mid) = {(z_left+z_right)/2 - z_mid:.4f} m",
            ])

    # ---------- S2: backward sanity with set_velocity input ----------
    scene.reset()
    # Try perturbing velocity at t=0 with a differentiable tensor
    try:
        n_v_int = rope.n_vertices if hasattr(rope, "n_vertices") else (rope.n_particles if hasattr(rope, "n_particles") else None)
        if n_v_int is None:
            log_lines(["[s2] skip — cannot determine n_vertices"])
        else:
            v_input = gs.tensor(torch.zeros(n_v_int, 3), requires_grad=True)
            try:
                rope.set_velocity(v_input)
                log_lines(["[s2] set_velocity(diff_input) accepted"])
                forward_ok = True
                try:
                    for t in range(HORIZON):
                        scene.step()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except gs.GenesisException as e:
                    forward_ok = False
                    log_lines([f"[s2] forward genesis_err:{str(e)[:100]}"])
                except Exception as e:
                    forward_ok = False
                    log_lines([f"[s2] forward err:{repr(e)[:100]}"])

                if forward_ok:
                    s = rope.get_state()
                    out_pos = s.pos
                    log_lines([f"[s2] post-forward state.pos rg={out_pos.requires_grad}"])
                    if out_pos.requires_grad:
                        loss = out_pos.pow(2).sum()
                        try:
                            loss.backward()
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            grad = v_input.grad
                            if grad is None:
                                log_lines(["[s2] backward ok but v_input.grad is None"])
                            else:
                                nan_count = int(torch.isnan(grad).sum().item())
                                gnorm = grad.norm().item()
                                log_lines([
                                    f"[s2] backward OK. v_input.grad norm={gnorm:.5g}, NaN={nan_count}/{grad.numel()}",
                                ])
                        except Exception as e:
                            log_lines([f"[s2] backward failed: {repr(e)[:100]}"])
                    else:
                        log_lines(["[s2] state.pos rg=False — no grad path"])
            except Exception as e:
                log_lines([f"[s2] set_velocity failed: {repr(e)[:100]}"])
    except Exception as e:
        log_lines([f"[s2] outer err: {repr(e)[:100]}"])

    log_lines([f"[s2] mem={gpu_mem_mb()}MB"])
    log_lines([f"--- fem-elastic probe done ---", ""])

if __name__ == "__main__":
    main()
