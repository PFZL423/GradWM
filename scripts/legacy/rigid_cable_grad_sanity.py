"""
Phase 1 sanity check: rigid + ball joint + contact + autograd.

Reuses Genesis' bundled `xml/cable.xml` (26-segment capsule + ball joints)
as the rope, drops a cube next to it, and pushes the cube via a
requires_grad velocity tensor for `horizon` steps. Loss = squared distance
between cable's tail link and a goal position. Then loss.backward().

What we want out of it:
  - Does forward run with requires_grad=True for rigid solver?
  - Does backward complete (= rigid+contact gradient path is wired)?
  - Per-step grad norms — health/NaN/spike at contact moments
  - Steady-state forward fps + backward fps after JIT warm-up
  - Peak GPU memory via nvidia-smi (PyTorch's torch.cuda.max_memory_allocated
    misses Taichi/Quadrants allocations)

Run from genesis-world repo so MJCF asset 'xml/cable.xml' resolves:
  cd /home/ubuntu/genisis/external/genesis-world
  python /home/ubuntu/genisis/scripts/rigid_cable_grad_sanity.py
"""

import os
import subprocess
import time

import torch

import genesis as gs


def gpu_mem_mb():
    """Process-level GPU memory in MB via nvidia-smi (sees Taichi too)."""
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


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    horizon = 30
    dt = 2e-3
    substeps = 4

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

    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(-0.30, -0.26, 0.05),
        )
    )

    scene.build()

    print(f"[sanity] cable n_links={cable.n_links}  n_dofs={cable.n_dofs}")
    print(f"[sanity] cube n_links={cube.n_links}  n_dofs={cube.n_dofs}")

    def run_once(label: str, take_backward: bool):
        scene.reset()

        v_list = [
            gs.tensor([0.40, 0.0, 0.0], requires_grad=True) for _ in range(horizon)
        ]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_fwd_start = time.time()
        for v_i in v_list:
            full_vel = torch.cat([v_i, torch.zeros(3, device=v_i.device, dtype=v_i.dtype)])
            cube.set_dofs_velocity(full_vel)
            scene.step()

        # loss: cable's tail link xy position vs a goal.
        # cable bodies in MJCF are named B_first ... B_last, so grab the last one.
        tail_state = cable.get_state()
        # cable.get_state() returns a state with `pos` shape [n_links, 3] (or [n_envs, n_links, 3]).
        tail_pos = tail_state.pos[-1] if tail_state.pos.dim() == 2 else tail_state.pos[0, -1]
        goal = gs.tensor([-0.30, 0.0, 0.06])
        loss = torch.pow(tail_pos - goal, 2).sum()

        torch.cuda.synchronize()
        t_fwd = time.time() - t_fwd_start

        if take_backward:
            t_bwd_start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            t_bwd = time.time() - t_bwd_start

            grad_norms = [
                v.grad.norm().item() if v.grad is not None else None for v in v_list
            ]
            nan_count = sum(
                1 for v in v_list if v.grad is not None and torch.isnan(v.grad).any()
            )
        else:
            t_bwd = None
            grad_norms = None
            nan_count = None

        peak_torch = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else None
        )
        peak_proc = gpu_mem_mb()

        print(f"[sanity:{label}] forward {horizon} steps: {t_fwd:.3f}s -> {horizon/t_fwd:.1f} step/s  loss={loss.item():.5f}")
        if take_backward:
            print(f"[sanity:{label}] backward: {t_bwd:.3f}s")
            print(f"[sanity:{label}] grad NaN: {nan_count}/{horizon}")
            print(f"[sanity:{label}] grad norms: {[round(g, 4) if g is not None else None for g in grad_norms]}")
        print(f"[sanity:{label}] peak torch alloc: {peak_torch:.1f} MB  peak nvidia-smi: {peak_proc} MB")

    # warmup: pay JIT compile cost (forward + backward each compile separately)
    print("[sanity] === warmup run (JIT compile, expect 100s+) ===")
    run_once("warmup", take_backward=True)

    print("[sanity] === steady-state run #1 ===")
    run_once("steady1", take_backward=True)

    print("[sanity] === steady-state run #2 ===")
    run_once("steady2", take_backward=True)

    print("[sanity] === forward-only steady-state ===")
    run_once("fwd-only", take_backward=False)


if __name__ == "__main__":
    main()
