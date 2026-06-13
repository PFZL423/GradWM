"""PBD-Cloth rope probe.

Generates a thin strip mesh (0.30m x 0.01m, 60 segments x 3 width) on the fly,
loads as gs.materials.PBD.Cloth() (2D), validates:
  1. Forward 30 steps without crashing under gravity-only.
  2. Mid-point sag visible (does the strip deform like rope?).
  3. Whether any differentiable particle handle exists for backward.

Run: python scripts/rope_solver_probe/probe_pbd_cloth.py
"""
import sys, os, subprocess, time, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import trimesh
import imageio.v2 as imageio
import torch
import genesis as gs

LOG = Path("logs/rope_pbd_cloth.log")
OUT = Path("analysis/rope_solver_probe")
OUT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

ROPE_LEN = 0.30
ROPE_WIDTH = 0.01    # very narrow strip — looks like rope from side
N_SEG_LEN = 60       # along length
N_SEG_W = 3          # across width
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


def make_strip_obj(out_path: str):
    """Generate a thin rectangular strip trimesh and write to obj."""
    # Vertices on a flat XY plane; will be positioned later via morph.pos
    xs = np.linspace(-ROPE_LEN/2, ROPE_LEN/2, N_SEG_LEN)
    ys = np.linspace(-ROPE_WIDTH/2, ROPE_WIDTH/2, N_SEG_W)
    verts = []
    for x in xs:
        for y in ys:
            verts.append([x, y, 0.0])
    verts = np.asarray(verts)

    faces = []
    for i in range(N_SEG_LEN - 1):
        for j in range(N_SEG_W - 1):
            v00 = i*N_SEG_W + j
            v01 = i*N_SEG_W + j + 1
            v10 = (i+1)*N_SEG_W + j
            v11 = (i+1)*N_SEG_W + j + 1
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.asarray(faces)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(out_path)
    return verts.shape[0], faces.shape[0]


def render_frame(cam, label):
    rgb = cam.render()
    if isinstance(rgb, tuple):
        rgb = rgb[0]
    path = OUT / f"pbd_cloth_{label}.png"
    imageio.imwrite(str(path), np.asarray(rgb))
    return path


def main():
    log_lines([f"--- pbd-cloth probe ts={time.strftime('%Y-%m-%dT%H:%M:%S')} ---"])

    obj_tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w").name
    n_v, n_f = make_strip_obj(obj_tmp)
    log_lines([f"[mesh] strip obj written: verts={n_v} faces={n_f} path={obj_tmp}"])

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
        morph=gs.morphs.Mesh(
            file=obj_tmp,
            pos=(0.0, 0.0, 0.30),
        ),
        material=gs.materials.PBD.Cloth(
            rho=1.0,
            stretch_compliance=1e-6,
            bending_compliance=1e-3,    # softer bending → more rope-like sag
            stretch_relaxation=0.3,
            bending_relaxation=0.1,
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

    log_lines([f"[probe] rope n_particles={getattr(rope, 'n_particles', '?')}"])

    # Probe diff handles
    for name in ("get_particles_pos", "get_particles_vel"):
        if hasattr(rope, name):
            v = getattr(rope, name)()
            log_lines([f"[probe] {name}() shape={tuple(v.shape)} rg={v.requires_grad}"])

    # ---------- S1: free fall under gravity ----------
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

    # quantify mid-point sag
    if status == "ok":
        pos = rope.get_particles_pos().detach().cpu().numpy()
        # find left/right ends + middle by x
        x_ords = np.argsort(pos[:, 0])
        left_id = x_ords[:N_SEG_W].tolist()       # leftmost slab
        right_id = x_ords[-N_SEG_W:].tolist()
        mid_id = x_ords[len(x_ords)//2 - N_SEG_W//2 : len(x_ords)//2 + N_SEG_W//2 + 1].tolist()
        z_left  = pos[left_id, 2].mean()
        z_right = pos[right_id, 2].mean()
        z_mid   = pos[mid_id, 2].mean()
        sag = (z_left + z_right) / 2 - z_mid
        log_lines([
            f"[s1] z_left={z_left:.4f}, z_mid={z_mid:.4f}, z_right={z_right:.4f}",
            f"[s1] sag (ends_avg - mid) = {sag:.4f} m",
        ])

    # ---------- S2: mid-grasp lift (mimics our manipulation) ----------
    scene.reset()
    # Find center particles to fix
    pos_init = rope.get_particles_pos().detach().cpu().numpy()
    x_ords = np.argsort(pos_init[:, 0])
    mid_idx = x_ords[len(x_ords)//2 - N_SEG_W//2 : len(x_ords)//2 + N_SEG_W//2 + 1].tolist()
    log_lines([f"[s2] mid-grasp particles to fix: {mid_idx}"])

    # Try to pin them to a virtual moving point
    if hasattr(rope, "fix_particles"):
        try:
            rope.fix_particles(particles_idx_local=mid_idx)
            log_lines(["[s2] fix_particles called"])
        except Exception as e:
            log_lines([f"[s2] fix_particles failed: {repr(e)[:80]}"])

    render_frame(cam, "s2_t0")
    status_s2 = "ok"
    try:
        for t in range(HORIZON):
            # crude lift: each step manually offset fixed particles upward by 1mm
            scene.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except gs.GenesisException as e:
        status_s2 = f"genesis_err:{str(e)[:100]}"
    except Exception as e:
        status_s2 = f"err:{repr(e)[:100]}"
    render_frame(cam, "s2_t30")
    log_lines([f"[s2] mid-fixed gravity: status={status_s2} mem={gpu_mem_mb()}MB"])

    if status_s2 == "ok":
        pos2 = rope.get_particles_pos().detach().cpu().numpy()
        z_left2  = pos2[left_id, 2].mean()
        z_right2 = pos2[right_id, 2].mean()
        z_mid2   = pos2[mid_id, 2].mean()
        sag2 = (z_left2 + z_right2) / 2 - z_mid2
        log_lines([
            f"[s2] z_left={z_left2:.4f}, z_mid={z_mid2:.4f}, z_right={z_right2:.4f}",
            f"[s2] sag (ends_avg - mid) = {sag2:.4f} m  <- mid pinned to initial",
        ])

    log_lines([f"--- pbd-cloth probe done ---", ""])

if __name__ == "__main__":
    main()
