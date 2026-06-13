"""
Sanity: Franka hand.xml alone, requires_grad backward.

hand.xml has a tendon coupling and an <equality><joint joint1=.. joint2=..>
constraint. We've ruled out body-body weld as a NaN cause, but not joint-joint
coupling. Pin the hand in space (it has no root joint), close fingers via
set_dofs_position, run 30 steps, backward.
"""

import os
import subprocess
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


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3, substeps=4, substeps_local=4, requires_grad=True
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    # panda.xml backward NaN@29 — try panda_no_tendon.xml which has neither
    # tendon nor equality (panda.xml has both, like cable.xml).
    hand = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda_no_tendon.xml"))
    scene.build()
    scene.reset()

    print(f"[hand] n_links={hand.n_links}  n_dofs={hand.n_dofs}")

    # Set qpos to a safe value inside joint limits (matches franka_cube.py).
    # joint4 has range [-3.0718, -0.0698] so qpos0=0 is invalid.
    qpos_safe = torch.tensor([
        -1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04
    ], dtype=torch.float32)
    hand.set_dofs_position(qpos_safe)

    horizon = 30
    t0 = time.time()
    for _ in range(horizon):
        scene.step()
    torch.cuda.synchronize()
    t_fwd = time.time() - t0

    # Loss: position of left finger (link 1) — keep simple, just need a non-trivial backward
    pos = hand.get_state().pos
    target = pos[-1] if pos.dim() == 2 else pos[0, -1]
    loss = torch.pow(target - gs.tensor([0.0, 0.0, 0.0]), 2).sum()

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

    print(f"[hand] fwd={t_fwd:.2f}s  loss={loss.item():.5f}  bwd={t_bwd:.2f}s  status={status}  mem={gpu_mem_mb()}MB")


if __name__ == "__main__":
    main()
