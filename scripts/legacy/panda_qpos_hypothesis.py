"""
Subagent #1's hypothesis: Genesis NaN root cause is `qd_rotvec_to_quat`
producing 0/0 when rotvec ≈ 0. Panda's default home qpos has all 7 revolute
joints at 0 → triggers it.

Test: explicitly set panda qpos to nonzero values BEFORE the first step.
If backward NaN goes away, hypothesis confirmed and panda is usable.

Three variants:
  - default qpos (control)
  - safe nonzero qpos set once before step
  - safe nonzero qpos re-set every step (in case Genesis resets qpos0)
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


SAFE_QPOS = torch.tensor(
    [0.3, 0.3, 0.3, -1.5, 0.3, 1.5, 0.3, 0.04, 0.04],
    dtype=torch.float32,
)


def run(label, qpos_strategy):
    """qpos_strategy: 'default' / 'set_once' / 'set_every_step'"""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3, substeps=4, substeps_local=4, requires_grad=True
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    panda = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda_no_tendon.xml"))
    scene.build()
    scene.reset()

    if qpos_strategy != "default":
        panda.set_dofs_position(SAFE_QPOS)

    horizon = 30
    t0 = time.time()
    try:
        for _ in range(horizon):
            if qpos_strategy == "set_every_step":
                panda.set_dofs_position(SAFE_QPOS)
            scene.step()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[{label}] FORWARD FAIL: {repr(e)[:200]}")
        return
    t_fwd = time.time() - t0

    pos = panda.get_state().pos
    tail = pos[-1] if pos.dim() == 2 else pos[0, -1]
    loss = torch.pow(tail - gs.tensor([0.5, 0.0, 0.5]), 2).sum()

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

    print(f"[{label}] fwd={t_fwd:.2f}s  loss={loss.item():.5f}  bwd={t_bwd:.2f}s  status={status}  mem={gpu_mem_mb()}MB")


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    print("=== panda qpos hypothesis test ===")
    run("default", "default")
    run("set_once", "set_once")
    run("set_every_step", "set_every_step")


if __name__ == "__main__":
    main()
