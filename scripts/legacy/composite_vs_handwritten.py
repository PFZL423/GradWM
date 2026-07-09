"""
Hypothesis: NaN is not "26 segments break". The 26-seg cable.xml uses
MuJoCo's `<composite type="cable">` macro, which expands to joints of a
different kind from our hand-written 25-seg ball-jointed chain.

Test: write a 26-segment hand-written ball-jointed chain (same idiom as
5/6/.../25 seg) — if backward succeeds, the prior conclusion was wrong:
NaN comes from `composite cable` expansion, not from "26 segments".
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


def make_handwritten_cable(n_segments: int, with_weld: bool = True) -> str:
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

    nested = "\n".join(bodies_open) + "\n" + "\n".join(bodies_close)
    weld = '\n    <equality>\n        <weld body1="B0" body2="anchor"/>\n    </equality>' if with_weld else ""
    return (
        f'<mujoco model="cable_{n_segments}seg_handwritten">\n'
        f'    <worldbody>\n'
        f'        <body name="anchor" pos="0 0 1.07">\n'
        f'          <geom type="sphere" size="0.01" rgba="1 0 0 1" contype="0" conaffinity="0"/>\n'
        f'        </body>\n'
        f'        {nested}\n'
        f'    </worldbody>{weld}\n'
        f'</mujoco>\n'
    )


def run(label: str, mjcf_text: str = None, mjcf_file: str = None, horizon: int = 30):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3, substeps=4, substeps_local=4, requires_grad=True
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    if mjcf_text is not None:
        tmp = tempfile.NamedTemporaryFile(prefix=label + "_", suffix=".xml", delete=False, mode="w")
        tmp.write(mjcf_text)
        tmp.flush()
        tmp.close()
        ent = scene.add_entity(gs.morphs.MJCF(file=tmp.name))
    else:
        ent = scene.add_entity(gs.morphs.MJCF(file=mjcf_file))

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
        status = msg.replace("Nan grad in qpos or dofs_vel found at step ", "nan@step") if "Nan grad" in msg else f"genesis_error:{msg[:100]}"
    except Exception as e:
        status = f"error:{repr(e)[:100]}"
    t_bwd = time.time() - t0

    print(
        f"[{label}] n_links={ent.n_links}  n_dofs={ent.n_dofs}  fwd={t_fwd:.2f}s  "
        f"loss={loss.item():.5f}  bwd={t_bwd:.2f}s  status={status}  mem={gpu_mem_mb()}MB"
    )


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    print("=== handwritten 26-segment ball-jointed cable (with weld) ===")
    run("hand26_weld", mjcf_text=make_handwritten_cable(26, with_weld=True))

    print("\n=== handwritten 26-segment ball-jointed cable (no weld) ===")
    run("hand26_noweld", mjcf_text=make_handwritten_cable(26, with_weld=False))

    print("\n=== bundled cable.xml (composite, 26 segments, with weld) ===")
    run("composite26", mjcf_file="xml/cable.xml")

    print("\n=== handwritten 30-segment ball-jointed cable (with weld) ===")
    run("hand30_weld", mjcf_text=make_handwritten_cable(30, with_weld=True))


if __name__ == "__main__":
    main()
