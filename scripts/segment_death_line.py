"""
Find the segment-count death line for Genesis 1.1.1 rigid+autograd backward.

Earlier sweeps showed: 5 segments (18 DoF) backward OK; 26 segments (84 DoF)
backward NaN immediately. Goal: find the exact N where backward starts to NaN.

Generates a temp MJCF for each N in {6, 8, 10, 12, 15, 18, 20, 22, 25}, attaches
a weld to anchor (matches D/F structure), runs gravity-only forward 30 steps,
loss = (tail-link pos)^2, loss.backward(). Records backward status and NaN
step (if any).
"""

import os
import subprocess
import tempfile
import time

import torch

import genesis as gs


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


def make_cable_mjcf(n_segments: int, with_weld: bool = True) -> str:
    """Generate an n-segment ball-jointed cable, optionally welded at the top."""
    seg_len = 0.025
    spacing = 0.055

    bodies_open = []
    bodies_close = []
    for i in range(n_segments):
        if i == 0:
            bodies_open.append(
                f'<body name="B{i}" pos="0 0 1.0">\n'
                f'  <freejoint/>\n'
                f'  <geom type="capsule" size="0.005 {seg_len}" mass="0.001" rgba="0.8 0.6 0.4 1"/>'
            )
        else:
            bodies_open.append(
                f'<body name="B{i}" pos="0 0 -{spacing}">\n'
                f'  <joint type="ball" damping="0.01" armature="0.001"/>\n'
                f'  <geom type="capsule" size="0.005 {seg_len}" mass="0.001" rgba="0.8 0.6 0.4 1"/>'
            )
        bodies_close.append("</body>")

    nested_body = "\n".join(bodies_open) + "\n" + "\n".join(bodies_close)

    weld_block = ""
    if with_weld:
        weld_block = '\n    <equality>\n        <weld body1="B0" body2="anchor"/>\n    </equality>'

    anchor_body = (
        '<body name="anchor" pos="0 0 1.07">\n'
        '  <geom type="sphere" size="0.01" rgba="1 0 0 1" contype="0" conaffinity="0"/>\n'
        '</body>'
    )

    return (
        f'<mujoco model="cable_{n_segments}seg">\n'
        f'    <worldbody>\n'
        f'        {anchor_body}\n'
        f'        {nested_body}\n'
        f'    </worldbody>{weld_block}\n'
        f'</mujoco>\n'
    )


def run_for_n(n_segments: int, horizon: int = 30):
    mjcf_text = make_cable_mjcf(n_segments, with_weld=True)
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"cable_{n_segments}seg_",
        suffix=".xml",
        delete=False,
        mode="w",
    )
    tmp.write(mjcf_text)
    tmp.flush()
    tmp.close()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3, substeps=4, substeps_local=4, requires_grad=True
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    ent = scene.add_entity(gs.morphs.MJCF(file=tmp.name))
    scene.build()
    scene.reset()

    t0 = time.time()
    for _ in range(horizon):
        scene.step()
    torch.cuda.synchronize()
    t_fwd = time.time() - t0

    pos = ent.get_state().pos
    tail_pos = pos[-1] if pos.dim() == 2 else pos[0, -1]
    loss = torch.pow(tail_pos - gs.tensor([0.0, 0.0, 0.0]), 2).sum()

    status = "ok"
    t0 = time.time()
    try:
        loss.backward()
        torch.cuda.synchronize()
    except gs.GenesisException as e:
        msg = str(e)
        if "Nan grad" in msg:
            status = msg.replace("Nan grad in qpos or dofs_vel found at step ", "nan@step")
        else:
            status = f"genesis_error:{msg[:80]}"
    except Exception as e:
        status = f"error:{repr(e)[:80]}"
    t_bwd = time.time() - t0

    mem = gpu_mem_mb()
    print(
        f"[N={n_segments:>3}] n_dofs={ent.n_dofs:>3}  fwd={t_fwd:.2f}s  loss={loss.item():.5f}  bwd={t_bwd:.2f}s  status={status}  mem={mem}MB"
    )

    os.unlink(tmp.name)
    return status


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # Already known: 5 OK, 26 NaN. Probe in between.
    for n in [6, 8, 10, 12, 15, 18, 20, 22, 25]:
        try:
            run_for_n(n, horizon=30)
        except Exception as e:
            print(f"[N={n}] OUTER FAIL: {e!r}")


if __name__ == "__main__":
    main()
