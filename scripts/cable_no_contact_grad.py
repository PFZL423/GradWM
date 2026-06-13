"""
NaN root-cause isolation: does the rigid backward NaN come from contact or
from the articulated-cable chain itself?

Setup: same `xml/cable.xml`, same dt/substeps/horizon as the original sanity
check, but no cube — cable just hangs and swings under gravity. Loss uses
tail-link position. If backward still NaNs, the articulated chain itself is
the culprit. If it succeeds, contact is the culprit.

A second variant injects a small initial perturbation via the cable's own
free-root velocity (still autograd-traced), so we have a non-zero gradient
path even without external bodies.
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


def build_scene(dt, substeps):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            substeps_local=substeps,
            requires_grad=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    cable = scene.add_entity(gs.morphs.MJCF(file="xml/cable.xml"))
    scene.build()
    return scene, cable


def run_no_contact(label: str, dt: float, substeps: int, horizon: int, perturb: bool):
    scene, cable = build_scene(dt, substeps)
    print(f"[{label}] cable n_links={cable.n_links}  n_dofs={cable.n_dofs}  dt={dt}  substeps={substeps}  horizon={horizon}  perturb={perturb}")

    scene.reset()

    # Optionally seed a small initial velocity on the cable's free root, with autograd.
    if perturb:
        v_init = gs.tensor([0.05, 0.0, 0.0], requires_grad=True)
        full = torch.cat([v_init, torch.zeros(cable.n_dofs - 3, device=v_init.device, dtype=v_init.dtype)])
        cable.set_dofs_velocity(full)
        v_list = [v_init]
    else:
        # No autograd-traced input — backward will trivially have nothing to send.
        # We still want to see whether the chain rule through cable physics raises.
        v_init = gs.tensor([0.05, 0.0, 0.0], requires_grad=True)
        v_list = [v_init]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for _ in range(horizon):
        scene.step()
    torch.cuda.synchronize()
    t_fwd = time.time() - t0

    tail_state = cable.get_state()
    tail_pos = tail_state.pos[-1] if tail_state.pos.dim() == 2 else tail_state.pos[0, -1]
    goal = gs.tensor([0.0, 0.0, 0.0])
    loss = torch.pow(tail_pos - goal, 2).sum()

    err = None
    t_bwd = None
    grad_info = None
    try:
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.time() - t0
        grad_info = (
            v_init.grad.norm().item() if v_init.grad is not None else None,
            (torch.isnan(v_init.grad).any().item() if v_init.grad is not None else None),
        )
    except Exception as e:
        err = repr(e)

    peak_torch = torch.cuda.max_memory_allocated() / 1024 / 1024
    peak_proc = gpu_mem_mb()

    print(f"[{label}] forward {horizon}: {t_fwd:.2f}s  loss={loss.item():.5f}")
    if err:
        print(f"[{label}] backward FAILED: {err}")
    else:
        print(f"[{label}] backward: {t_bwd:.2f}s  v_init grad: {grad_info}")
    print(f"[{label}] peak torch: {peak_torch:.1f} MB  peak nvidia-smi: {peak_proc} MB")
    return err is None


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # Run 1: no-contact, no perturb (cable just hangs + swings under gravity).
    print("=== run 1: no-contact, no autograd input ===")
    run_no_contact("hang", dt=2e-3, substeps=4, horizon=30, perturb=False)


if __name__ == "__main__":
    main()
