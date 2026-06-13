"""
Smaller variant of Genesis' examples/differentiable_push.py.

Goal: figure out whether the earlier 600s timeout was the 4060 being slow,
or something else (jit compile, init loop). Shrinks horizon, n_envs, and
particle count so backward should finish on a 4060 8GB if the path works
at all.

Reports forward / backward / total wallclock and peak GPU memory.
"""

import os
import time

import torch

import genesis as gs


def main():
    gs.init(precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
            requires_grad=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 0.55),
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    stick = scene.add_entity(
        material=gs.materials.Tool(friction=8.0),
        morph=gs.morphs.Mesh(
            file="meshes/stirrer.obj",
            scale=0.6,
            pos=(0.5, 0.5, 0.05),
            euler=(90.0, 0.0, 0.0),
        ),
    )

    obj1 = scene.add_entity(
        material=gs.materials.MPM.Elastic(rho=500),
        morph=gs.morphs.Box(
            lower=(0.25, 0.15, 0.05),
            upper=(0.35, 0.25, 0.12),
        ),
        vis_mode="particle",
    )

    scene.build(n_envs=1)

    horizon = 20
    v_list = [
        gs.tensor([[0.0, 1.0, 0.0]], requires_grad=True) for _ in range(horizon)
    ]

    scene.reset()
    init_pos = gs.tensor([[0.3, 0.1, 0.18]], requires_grad=True)
    stick.set_position(init_pos)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    loss = 0.0
    for i, v_i in enumerate(v_list):
        stick.set_velocity(vel=v_i)
        scene.step()
        if i == horizon - 1:
            goal = gs.tensor([0.5, 0.8, 0.05])
            mpm_state = scene.get_state().solvers_state[
                scene.solvers.index(scene.mpm_solver)
            ]
            loss = torch.pow(
                mpm_state.pos[mpm_state.active == 1] - goal, 2
            ).sum()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_fwd = time.time() - t0
    print(f"[diff_push_small] forward {horizon} steps: {t_fwd:.2f}s  loss={loss.item():.4f}")

    t0 = time.time()
    loss.backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_bwd = time.time() - t0
    print(f"[diff_push_small] backward: {t_bwd:.2f}s")

    grad_norms = [v.grad.norm().item() if v.grad is not None else None for v in v_list]
    print(f"[diff_push_small] per-step grad norms: {grad_norms}")
    print(f"[diff_push_small] grad-NaN count: {sum(1 for v in v_list if v.grad is not None and torch.isnan(v.grad).any())} / {horizon}")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"[diff_push_small] peak CUDA alloc: {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
