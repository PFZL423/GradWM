"""
Sanity: 50-segment hand-written ball-jointed cable, requires_grad backward.
Already known: 30 seg OK. 50 seg untested.
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


def make_handwritten_cable(n_segments: int) -> str:
    seg_len = 0.025
    spacing = 0.055

    open_ = []
    close_ = []
    for i in range(n_segments):
        if i == 0:
            open_.append(
                f'<body name="B{i}" pos="0 0 1.5">\n'
                f'  <freejoint/>\n'
                f'  <geom type="capsule" size="0.005 {seg_len}" mass="0.001" rgba="0.8 0.6 0.4 1"/>'
            )
        else:
            open_.append(
                f'<body name="B{i}" pos="0 0 -{spacing}">\n'
                f'  <joint type="ball" damping="0.01" armature="0.001"/>\n'
                f'  <geom type="capsule" size="0.005 {seg_len}" mass="0.001" rgba="0.8 0.6 0.4 1"/>'
            )
        close_.append("</body>")
    nested = "\n".join(open_) + "\n" + "\n".join(close_)
    return (
        f'<mujoco model="cable_{n_segments}seg">\n'
        f'    <worldbody>\n'
        f'        <body name="anchor" pos="0 0 1.57">\n'
        f'          <geom type="sphere" size="0.01" rgba="1 0 0 1" contype="0" conaffinity="0"/>\n'
        f'        </body>\n'
        f'        {nested}\n'
        f'    </worldbody>\n'
        f'    <equality>\n'
        f'        <weld body1="B0" body2="anchor"/>\n'
        f'    </equality>\n'
        f'</mujoco>\n'
    )


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    mjcf = make_handwritten_cable(50)
    tmp = tempfile.NamedTemporaryFile(prefix="cable_50seg_", suffix=".xml", delete=False, mode="w")
    tmp.write(mjcf)
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

    print(f"[50seg] n_links={ent.n_links}  n_dofs={ent.n_dofs}")

    horizon = 30
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
        status = msg.replace("Nan grad in qpos or dofs_vel found at step ", "nan@step") if "Nan grad" in msg else f"genesis_error:{msg[:100]}"
    except Exception as e:
        status = f"error:{repr(e)[:100]}"
    t_bwd = time.time() - t0

    print(f"[50seg] fwd={t_fwd:.2f}s  loss={loss.item():.5f}  bwd={t_bwd:.2f}s  status={status}  mem={gpu_mem_mb()}MB")


if __name__ == "__main__":
    main()
