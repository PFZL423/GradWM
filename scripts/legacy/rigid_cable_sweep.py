"""Sweep rigid cable over segment count × damping × armature.

For each (N, damping, armature):
  - Build a horizontal cable (length 0.30m) hanging across two static tables
    with gap 0.20m, two ends RESTING on tables (no weld).
  - 60-step gravity-only forward + 30-step backward sanity (check NaN).
  - Measure: midpoint sag (mm) — how much does the cable visibly droop?
  - Measure: forward fps + backward NaN status.

Output: logs/rigid_cable_sweep.csv with columns
  N, damping, armature, sag_mm, fwd_fps, bwd_status, peak_mem_mb

Render side-camera mp4 of two representative configs (low N + tight damping
vs high N + low damping) for visual comparison.
"""
import csv
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import imageio.v2 as imageio
import numpy as np
import torch
import genesis as gs

LOG = Path("logs/rigid_cable_sweep.log")
CSV_PATH = Path("logs/rigid_cable_sweep.csv")
OUT = Path("analysis/rigid_cable_sweep")
OUT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

CABLE_LEN = 0.30
SEG_RADIUS = 0.010
SEG_LEN = 0.020             # capsule cylinder length (excluding hemisphere caps)
TABLE_X_LEFT = 0.10
TABLE_X_RIGHT = 0.40
TABLE_GAP = TABLE_X_RIGHT - TABLE_X_LEFT
# Cable spans the gap with a few cm of slack on each table top, so the rope
# can sag visibly. Total chain length = (N-1) * spacing + ~capsule diameter,
# we pick spacing per N so total length = TABLE_GAP * 1.4 (40% slack).
TABLE_TOP_Z = 0.14
TABLE_TOP_HALF = 0.04
TABLE_TOP_HALF_Y = 0.05
TABLE_TOP_THICK = 0.01
TABLE_LEG_HALF = 0.025
CABLE_REST_Z = TABLE_TOP_Z + TABLE_TOP_THICK + 0.012  # just above table top
HORIZON_FWD = 60
HORIZON_BWD = 30


def gpu_mem_mb():
    try:
        pid = os.getpid()
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if int(parts[0]) == pid:
                return float(parts[1])
    except Exception:
        pass
    return None


def make_cable_mjcf(n_segments: int, damping: float, armature: float,
                    seg_mass: float = 0.001) -> str:
    """Horizontal cable, no weld, lays free under gravity. Total chain length
    = TABLE_GAP * 1.4 so there's enough slack to sag in the middle."""
    target_total = TABLE_GAP * 1.4   # ~0.42m for 0.30m gap
    spacing = target_total / max(1, n_segments - 1) if n_segments > 1 else 0.0
    halflen = SEG_LEN * 0.5
    # Start cable at left table top inboard edge; lay along +x
    x0 = TABLE_X_LEFT
    z0 = CABLE_REST_Z
    bodies_open: list[str] = []
    bodies_close: list[str] = []
    for i in range(n_segments):
        if i == 0:
            bodies_open.append(
                f'<body name="B{i}" pos="{x0} 0 {z0}">\n'
                f'  <freejoint/>\n'
                f'  <geom type="capsule" euler="0 90 0" size="{SEG_RADIUS} {halflen}" '
                f'mass="{seg_mass}" rgba="0.86 0.62 0.20 1" contype="1" conaffinity="1"/>'
            )
        else:
            bodies_open.append(
                f'<body name="B{i}" pos="{spacing} 0 0">\n'
                f'  <joint type="ball" damping="{damping}" armature="{armature}"/>\n'
                f'  <geom type="capsule" euler="0 90 0" size="{SEG_RADIUS} {halflen}" '
                f'mass="{seg_mass}" rgba="0.86 0.62 0.20 1" contype="1" conaffinity="1"/>'
            )
        bodies_close.append("</body>")
    nested = "\n".join(bodies_open) + "\n" + "\n".join(bodies_close)

    leg_top_z = TABLE_TOP_Z - TABLE_TOP_THICK
    leg_half_z = leg_top_z * 0.5

    return f"""<mujoco model="cable_sweep_{n_segments}">
    <worldbody>
        <geom name="table_L_top" type="box" pos="{TABLE_X_LEFT} 0 {TABLE_TOP_Z}"
              size="{TABLE_TOP_HALF} {TABLE_TOP_HALF_Y} {TABLE_TOP_THICK}"
              rgba="0.55 0.40 0.25 1" contype="1" conaffinity="1"/>
        <geom name="table_L_leg" type="box" pos="{TABLE_X_LEFT} 0 {leg_half_z}"
              size="{TABLE_LEG_HALF} {TABLE_LEG_HALF} {leg_half_z}"
              rgba="0.55 0.40 0.25 1" contype="0" conaffinity="0"/>
        <geom name="table_R_top" type="box" pos="{TABLE_X_RIGHT} 0 {TABLE_TOP_Z}"
              size="{TABLE_TOP_HALF} {TABLE_TOP_HALF_Y} {TABLE_TOP_THICK}"
              rgba="0.55 0.40 0.25 1" contype="1" conaffinity="1"/>
        <geom name="table_R_leg" type="box" pos="{TABLE_X_RIGHT} 0 {leg_half_z}"
              size="{TABLE_LEG_HALF} {TABLE_LEG_HALF} {leg_half_z}"
              rgba="0.55 0.40 0.25 1" contype="0" conaffinity="0"/>
        {nested}
    </worldbody>
</mujoco>
"""


def run_one(n: int, damping: float, armature: float, render: bool = False):
    mjcf = make_cable_mjcf(n, damping, armature)
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
    tmp.write(mjcf)
    tmp.close()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3, substeps=4, substeps_local=4, requires_grad=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(pos=(0, 0, -0.001)),
        surface=gs.surfaces.Default(color=(0.18, 0.20, 0.25, 1)),
    )
    cable = scene.add_entity(gs.morphs.MJCF(file=tmp.name))
    if render:
        cam = scene.add_camera(
            res=(640, 360),
            pos=((TABLE_X_LEFT + TABLE_X_RIGHT) * 0.5, 0.7, 0.20),
            lookat=((TABLE_X_LEFT + TABLE_X_RIGHT) * 0.5, 0.0, 0.13),
            up=(0, 0, 1), fov=42,
        )
    else:
        cam = None
    scene.build()
    scene.reset()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---------- Forward ----------
    frames = []
    fwd_status = "ok"
    t0 = time.time()
    try:
        for _ in range(HORIZON_FWD):
            scene.step()
            if cam is not None:
                rgb = cam.render()
                if isinstance(rgb, tuple):
                    rgb = rgb[0]
                frames.append(np.asarray(rgb))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except gs.GenesisException as e:
        fwd_status = f"genesis_err:{str(e)[:80]}"
    except Exception as e:
        fwd_status = f"err:{repr(e)[:80]}"
    t_fwd = time.time() - t0
    fps = HORIZON_FWD / t_fwd if t_fwd > 0 else math.nan

    # ---------- Sag measurement: read end-of-forward link positions ----------
    sag_mm = math.nan
    if fwd_status == "ok":
        links_pos = cable.get_links_pos().detach().cpu().numpy()
        if links_pos.ndim == 3:
            links_pos = links_pos[0]
        order = np.argsort(links_pos[:, 0])
        slab = max(2, len(order) // 12)
        z_left = links_pos[order[:slab], 2].mean()
        z_right = links_pos[order[-slab:], 2].mean()
        z_mid = links_pos[order[len(order) // 2 - slab // 2 : len(order) // 2 + slab // 2 + 1], 2].mean()
        sag_mm = float(((z_left + z_right) * 0.5 - z_mid) * 1000.0)

    # ---------- Backward sanity ----------
    scene.reset()
    bwd_status = "ok"
    nan_count = 0
    grad_norm = math.nan
    if fwd_status == "ok":
        # set a freejoint v0 input and let backward compute grad via state.pos
        try:
            for _ in range(HORIZON_BWD):
                scene.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            state = cable.get_state()
            if hasattr(state, "pos") and state.pos.requires_grad:
                loss = state.pos.pow(2).sum()
                try:
                    loss.backward()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    bwd_status = "ok"
                except gs.GenesisException as e:
                    msg = str(e)
                    bwd_status = (msg.replace("Nan grad in qpos or dofs_vel found at step ", "nan@step")
                                  if "Nan grad" in msg else f"genesis_err:{msg[:80]}")
                except Exception as e:
                    bwd_status = f"err:{repr(e)[:80]}"
            else:
                bwd_status = "no_grad_path"
        except gs.GenesisException as e:
            msg = str(e)
            bwd_status = (msg.replace("Nan grad in qpos or dofs_vel found at step ", "nan@step")
                          if "Nan grad" in msg else f"genesis_err:{msg[:80]}")
        except Exception as e:
            bwd_status = f"err:{repr(e)[:80]}"

    # nvidia-smi process query catches taichi GPU buffers (PyTorch
    # max_memory_allocated only sees its own tensors and undercounts heavily
    # for Genesis scenes — observed 0.05MB report for a real ~100MB scene).
    proc_mem = gpu_mem_mb()
    torch_mem = (torch.cuda.max_memory_allocated() / 1024 / 1024
                 if torch.cuda.is_available() else None)
    peak_mem = proc_mem if proc_mem is not None else torch_mem

    if cam is not None and frames:
        path = OUT / f"cable_N{n}_damp{damping}_arm{armature}.mp4"
        with imageio.get_writer(str(path), fps=20, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)

    os.unlink(tmp.name)
    # gs.destroy() invalidates init; we re-init at top of run_one instead.
    return {
        "N": n,
        "damping": damping,
        "armature": armature,
        "sag_mm": sag_mm,
        "fwd_fps": fps,
        "fwd_status": fwd_status,
        "bwd_status": bwd_status,
        "peak_mem_mb": peak_mem if peak_mem is not None else proc_mem,
    }


def main():
    # Each config runs in a fresh subprocess: Genesis 1.1.1 doesn't cleanly
    # support multiple scene builds in one process (gs.destroy invalidates
    # init; without it, second scene.build hits internal duplicate-id errors).

    fields = ["N", "damping", "armature", "sag_mm", "fwd_fps",
              "fwd_status", "bwd_status", "peak_mem_mb"]

    if "--worker" in sys.argv:
        # subprocess mode: read N/damping/armature from argv, run, print json
        import json
        idx = sys.argv.index("--worker")
        N = int(sys.argv[idx + 1])
        damping = float(sys.argv[idx + 2])
        armature = float(sys.argv[idx + 3])
        render = bool(int(sys.argv[idx + 4]))
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        try:
            r = run_one(N, damping, armature, render=render)
        except Exception as e:
            r = {"N": N, "damping": damping, "armature": armature,
                 "sag_mm": math.nan, "fwd_fps": math.nan,
                 "fwd_status": f"outer_err:{repr(e)[:100]}",
                 "bwd_status": "skipped", "peak_mem_mb": math.nan}
        # Sanitize non-jsonable
        clean = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                 for k, v in r.items()}
        print("__RESULT__" + json.dumps(clean))
        return 0

    new_csv = not CSV_PATH.exists()
    f_csv = CSV_PATH.open("a", newline="")
    writer = csv.DictWriter(f_csv, fieldnames=fields)
    if new_csv:
        writer.writeheader()
    f_log = LOG.open("a")

    def log(line):
        print(line)
        f_log.write(line + "\n")
        f_log.flush()

    log(f"=== rigid cable sweep ts={time.strftime('%Y-%m-%dT%H:%M:%S')} ===")

    n_values = [16, 30, 50, 75, 100]
    da_combos = [(0.01, 0.001), (0.001, 0.0001), (0.0001, 0.0)]

    import json
    script_path = str(Path(__file__).resolve())
    for N in n_values:
        for damping, armature in da_combos:
            render = (N == 16 and damping == 0.01) or (N == 100 and damping == 0.0001)
            cmd = ["conda", "run", "-n", "genesis", "--no-capture-output",
                   "python", script_path, "--worker",
                   str(N), str(damping), str(armature), str(int(render))]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                marker = "__RESULT__"
                line = next((l for l in p.stdout.split("\n") if l.startswith(marker)), None)
                if line:
                    r = json.loads(line[len(marker):])
                    # Re-fill nan from None
                    for k in ("sag_mm", "fwd_fps", "peak_mem_mb"):
                        if r.get(k) is None:
                            r[k] = math.nan
                else:
                    r = {"N": N, "damping": damping, "armature": armature,
                         "sag_mm": math.nan, "fwd_fps": math.nan,
                         "fwd_status": f"no_result_marker stderr={p.stderr[:100]}",
                         "bwd_status": "skipped", "peak_mem_mb": math.nan}
            except subprocess.TimeoutExpired:
                r = {"N": N, "damping": damping, "armature": armature,
                     "sag_mm": math.nan, "fwd_fps": math.nan,
                     "fwd_status": "timeout", "bwd_status": "skipped",
                     "peak_mem_mb": math.nan}

            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                             for k, v in r.items()})
            f_csv.flush()
            sag = r.get("sag_mm") or math.nan
            fps = r.get("fwd_fps") or math.nan
            log(f"N={r['N']:3d} damp={r['damping']} arm={r['armature']} "
                f"fps={fps:6.1f} sag={sag:6.2f}mm "
                f"fwd={str(r['fwd_status'])[:25]} bwd={str(r['bwd_status'])[:25]} "
                f"mem={r['peak_mem_mb']}MB")

    f_csv.close()
    f_log.close()
    print(f"[done] csv={CSV_PATH}  log={LOG}  videos in {OUT}")

if __name__ == "__main__":
    raise SystemExit(main())
