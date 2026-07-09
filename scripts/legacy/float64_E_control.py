"""
Float64 control: re-run cell E (26-segment cable, no weld, horizon=30) with
`precision="64"` instead of "32". External adversarial review pointed out
Genesis' only official differentiable rigid test uses precision=64, while we
have been running float32 throughout.

Predicted outcomes:
  - If backward succeeds   -> NaN was fp32 overflow, e.g. in qd_rotvec_to_quat
                              for ball joints. Verdict shifts toward (B/A-doc).
  - If still nan@step29    -> structural Genesis bug (~90% confidence).
"""

import os
import subprocess
import time

import torch

import genesis as gs


SCENES_DIR = "/home/ubuntu/genisis/scenes"


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


def run(label: str, precision: str):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3, substeps=4, substeps_local=4, requires_grad=True
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    ent = scene.add_entity(gs.morphs.MJCF(file=f"{SCENES_DIR}/cable_no_weld.xml"))
    scene.build()
    scene.reset()

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
        if "Nan grad" in msg:
            status = msg.replace("Nan grad in qpos or dofs_vel found at step ", "nan@step")
        else:
            status = f"genesis_error:{msg[:80]}"
    except Exception as e:
        status = f"error:{repr(e)[:80]}"
    t_bwd = time.time() - t0

    print(
        f"[{label} precision={precision}] n_dofs={ent.n_dofs:>3}  fwd={t_fwd:.2f}s  "
        f"loss={loss.item():.5f}  bwd={t_bwd:.2f}s  status={status}  mem={gpu_mem_mb()}MB"
    )


def main():
    # The decisive test: precision="64".
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    run("E_64", "64")


if __name__ == "__main__":
    main()
