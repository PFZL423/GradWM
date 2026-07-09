"""
Critical test: does rigid-rigid contact backward work in Genesis 1.1.1?

All our backward-OK cases so far had no actual contact. Push a free Box onto
a hand-written 5-segment ball-jointed cable so they truly collide, set
requires_grad=True, run forward 30 steps, loss = (cable tail pos)^2,
loss.backward(). Three sub-cases:

  - no_contact : Box dropped far from cable (gravity only, sanity baseline)
  - light      : Box dropped above cable, falls and gently lands on it
  - hard       : Box velocity-driven sideways into cable (sustained contact)

Pass = backward OK in light/hard. Fail = NaN in either.
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


def cable_5seg_mjcf() -> str:
    seg_len = 0.025
    spacing = 0.055
    bodies_o = []
    bodies_c = []
    for i in range(5):
        if i == 0:
            bodies_o.append(
                f'<body name="B0" pos="0 0 1.0">\n'
                f'  <freejoint/>\n'
                f'  <geom type="capsule" size="0.005 {seg_len}" mass="0.001" rgba="0.8 0.6 0.4 1"/>'
            )
        else:
            bodies_o.append(
                f'<body name="B{i}" pos="0 0 -{spacing}">\n'
                f'  <joint type="ball" damping="0.01" armature="0.001"/>\n'
                f'  <geom type="capsule" size="0.005 {seg_len}" mass="0.001" rgba="0.8 0.6 0.4 1"/>'
            )
        bodies_c.append("</body>")
    nested = "\n".join(bodies_o) + "\n" + "\n".join(bodies_c)
    return (
        f'<mujoco model="cable_5seg">\n'
        f'    <worldbody>\n'
        f'        <body name="anchor" pos="0 0 1.07">\n'
        f'          <geom type="sphere" size="0.01" rgba="1 0 0 1" contype="0" conaffinity="0"/>\n'
        f'        </body>\n'
        f'        {nested}\n'
        f'    </worldbody>\n'
        f'    <equality>\n'
        f'        <weld body1="B0" body2="anchor"/>\n'
        f'    </equality>\n'
        f'</mujoco>\n'
    )


def run(label, contact_mode):
    tmp = tempfile.NamedTemporaryFile(prefix="cable5_", suffix=".xml", delete=False, mode="w")
    tmp.write(cable_5seg_mjcf())
    tmp.flush()
    tmp.close()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3, substeps=4, substeps_local=4, requires_grad=True
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cable = scene.add_entity(gs.morphs.MJCF(file=tmp.name))

    if contact_mode == "no_contact":
        # Box far away from cable; sanity baseline
        box_pos = (0.5, 0.5, 0.5)
    elif contact_mode == "light":
        # Box right above the middle of cable; falls under gravity, lands on it
        box_pos = (0.0, 0.0, 0.85)
    elif contact_mode == "hard":
        # Box right next to mid-cable; we push it sideways in run loop
        box_pos = (-0.05, 0.0, 0.85)
    else:
        raise ValueError(contact_mode)
    box = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=box_pos))

    scene.build()
    scene.reset()

    horizon = 30
    t0 = time.time()
    for _ in range(horizon):
        if contact_mode == "hard":
            # Drive box +x at 0.5 m/s into the cable, sustained contact
            v = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            box.set_dofs_velocity(v)
        scene.step()
    torch.cuda.synchronize()
    t_fwd = time.time() - t0

    pos = cable.get_state().pos
    tail = pos[-1] if pos.dim() == 2 else pos[0, -1]
    loss = torch.pow(tail - gs.tensor([0.0, 0.0, 0.0]), 2).sum()

    status = "ok"
    t0 = time.time()
    try:
        loss.backward()
        torch.cuda.synchronize()
    except gs.GenesisException as e:
        msg = str(e)
        status = msg.replace("Nan grad in qpos or dofs_vel found at step ", "nan@") if "Nan grad" in msg else f"genesis_err:{msg[:80]}"
    except Exception as e:
        status = f"err:{repr(e)[:80]}"
    t_bwd = time.time() - t0

    print(
        f"[{label}] fwd={t_fwd:.2f}s  loss={loss.item():.5f}  bwd={t_bwd:.2f}s  status={status}  mem={gpu_mem_mb()}MB"
    )


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    print("=== rigid-rigid contact backward test ===")
    for mode in ["no_contact", "light", "hard"]:
        run(mode, mode)


if __name__ == "__main__":
    main()
