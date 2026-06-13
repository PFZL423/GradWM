"""
NaN root-cause isolation: 5-cell controlled experiment (A-E).

Each cell adds exactly one ingredient on top of the previous, so a
flip from "ok" to "nan" pinpoints which ingredient breaks the rigid
+ requires_grad backward path.

  A : single free Box, gravity only, 30 steps -> baseline
  B : 2 capsules + 1 ball joint, no weld   -> isolates ball joint
  C : 2 capsules + 1 ball joint + 1 weld   -> adds weld constraint
  D : 5-segment cable (composite-equivalent, with weld)  -> short rope
  E : 26-segment cable.xml with welds removed            -> long chain, no weld

Forward 30 steps under gravity (no external action), loss = (tail-link pos)^2,
loss.backward(). Report: forward_ok, backward_status (ok / nan@stepN / error),
forward_time, backward_time, peak nvidia-smi memory.

Run from genesis-world/ so xml/cable.xml resolves:
  cd /home/ubuntu/genisis/external/genesis-world
  python /home/ubuntu/genisis/scripts/nan_isolation_abcde.py
"""

import os
import subprocess
import time

import torch

import genesis as gs


SCENES_DIR = "/home/ubuntu/genisis/scenes"
DT = 2e-3
SUBSTEPS = 4
HORIZON = 30


def gpu_mem_mb():
    try:
        pid = os.getpid()
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if int(parts[0]) == pid:
                return float(parts[1])
    except Exception:
        pass
    return None


def make_scene():
    return gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=SUBSTEPS,
            substeps_local=SUBSTEPS,
            requires_grad=True,
        ),
        show_viewer=False,
    )


def add_entity_for(label):
    """Build a one-entity scene per label, return (scene, entity)."""
    scene = make_scene()
    scene.add_entity(gs.morphs.Plane())

    if label == "A":
        ent = scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, 0.5))
        )
    elif label == "B":
        ent = scene.add_entity(
            gs.morphs.MJCF(file=f"{SCENES_DIR}/two_capsule_ball.xml")
        )
    elif label == "C":
        ent = scene.add_entity(
            gs.morphs.MJCF(file=f"{SCENES_DIR}/two_capsule_ball_weld.xml")
        )
    elif label == "D":
        ent = scene.add_entity(
            gs.morphs.MJCF(file=f"{SCENES_DIR}/cable_5seg.xml")
        )
    elif label == "E":
        ent = scene.add_entity(
            gs.morphs.MJCF(file=f"{SCENES_DIR}/cable_no_weld.xml")
        )
    else:
        raise ValueError(label)

    scene.build()
    return scene, ent


def run_cell(label):
    print(f"\n=== cell {label} ===")
    scene, ent = add_entity_for(label)
    print(f"[{label}] entity n_links={ent.n_links}  n_dofs={ent.n_dofs}")
    scene.reset()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Forward: pure gravity for HORIZON steps. No external velocity / control.
    fwd_err = None
    t0 = time.time()
    try:
        for _ in range(HORIZON):
            scene.step()
        torch.cuda.synchronize()
    except Exception as e:
        fwd_err = repr(e)
    t_fwd = time.time() - t0

    if fwd_err:
        print(f"[{label}] forward FAILED in {t_fwd:.2f}s: {fwd_err}")
        return

    # Loss = (tail-link position)^2 . tail-link is the last link in the entity.
    tail_state = ent.get_state()
    pos = tail_state.pos
    if pos.dim() == 2:        # [n_links, 3]
        tail_pos = pos[-1]
    else:                     # [n_envs, n_links, 3]
        tail_pos = pos[0, -1]
    goal = gs.tensor([0.0, 0.0, 0.0])
    loss = torch.pow(tail_pos - goal, 2).sum()
    print(f"[{label}] forward {HORIZON} steps: {t_fwd:.2f}s  loss={loss.item():.5f}")

    # Backward
    bwd_status = "ok"
    bwd_detail = None
    t0 = time.time()
    try:
        loss.backward()
        torch.cuda.synchronize()
    except gs.GenesisException as e:
        msg = str(e)
        if "Nan grad" in msg:
            bwd_status = msg.replace(
                "Nan grad in qpos or dofs_vel found at step ", "nan@step"
            )
        else:
            bwd_status = "genesis_error"
            bwd_detail = msg
    except Exception as e:
        bwd_status = "error"
        bwd_detail = repr(e)
    t_bwd = time.time() - t0

    peak_torch = torch.cuda.max_memory_allocated() / 1024 / 1024
    peak_proc = gpu_mem_mb()

    print(f"[{label}] backward: {t_bwd:.2f}s  status={bwd_status}")
    if bwd_detail:
        print(f"[{label}] detail: {bwd_detail[:200]}")
    print(f"[{label}] peak torch alloc: {peak_torch:.1f} MB  peak nvidia-smi: {peak_proc} MB")


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # Run A-C first (each is small and fast); D-E gated on early results.
    for label in ["A", "B", "C", "D", "E"]:
        try:
            run_cell(label)
        except Exception as e:
            print(f"[{label}] OUTER FAIL: {e!r}")


if __name__ == "__main__":
    main()
