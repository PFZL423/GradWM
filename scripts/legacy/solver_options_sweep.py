"""
Solver-options sweep: can we rescue Genesis 1.1.1 backward NaN by
flipping one or more rigid_options/sim_options knobs?

Targets (both reproduce NaN at default settings):
  - panda_no_tendon.xml (9 DoF, has joint range, self-collision, actuators)
  - xml/cable.xml       (84 DoF, composite expansion + 2 free joints)

Knobs tested (each is one knob change vs baseline, keeping all others default):
  baseline   : default rigid_options
  cg         : constraint_solver=CG instead of Newton
  euler      : integrator=Euler
  no_jlim    : enable_joint_limit=False
  no_self    : enable_self_collision=False
  no_constr  : disable_constraint=True (entire constraint solver off)
  combo      : enable_joint_limit=False + enable_self_collision=False
                 + disable_constraint=True (matches Genesis own
                 test_differentiable_rigid setup)

Float64 lives in a separate script (precision must be set at gs.init).
"""

import argparse
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


def make_rigid_options(knob: str):
    if knob == "baseline":
        return gs.options.RigidOptions()
    if knob == "cg":
        return gs.options.RigidOptions(constraint_solver=gs.constraint_solver.CG)
    if knob == "euler":
        return gs.options.RigidOptions(integrator=gs.integrator.Euler)
    if knob == "no_jlim":
        return gs.options.RigidOptions(enable_joint_limit=False)
    if knob == "no_self":
        return gs.options.RigidOptions(enable_self_collision=False)
    if knob == "no_constr":
        return gs.options.RigidOptions(disable_constraint=True)
    if knob == "combo":
        return gs.options.RigidOptions(
            enable_joint_limit=False,
            enable_self_collision=False,
            disable_constraint=True,
        )
    raise ValueError(knob)


def run_one(asset_name: str, mjcf_file: str, knob: str, qpos_safe=None):
    print(f"\n--- {asset_name} | knob={knob} ---", flush=True)
    try:
        rigid_opts = make_rigid_options(knob)
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=2e-3, substeps=4, substeps_local=4, requires_grad=True
            ),
            rigid_options=rigid_opts,
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        ent = scene.add_entity(gs.morphs.MJCF(file=mjcf_file))
        scene.build()
        scene.reset()
    except Exception as e:
        print(f"  BUILD FAIL: {repr(e)[:200]}")
        return

    if qpos_safe is not None:
        try:
            ent.set_dofs_position(qpos_safe)
        except Exception:
            pass

    horizon = 30
    t0 = time.time()
    try:
        for _ in range(horizon):
            scene.step()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  FORWARD FAIL: {repr(e)[:200]}")
        return
    t_fwd = time.time() - t0

    pos = ent.get_state().pos
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
        f"  fwd={t_fwd:.2f}s  loss={loss.item():.5f}  bwd={t_bwd:.2f}s  status={status}  "
        f"mem={gpu_mem_mb()}MB  n_dofs={ent.n_dofs}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["cable", "panda", "both"], default="both")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    knobs = ["baseline", "cg", "euler", "no_jlim", "no_self", "no_constr", "combo"]

    qpos_panda_safe = torch.tensor(
        [-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04],
        dtype=torch.float32,
    )

    if args.target in ("cable", "both"):
        print("\n========== CABLE.XML (composite, 84 DoF) ==========")
        for k in knobs:
            try:
                run_one("cable", "xml/cable.xml", k, qpos_safe=None)
            except Exception as e:
                print(f"  OUTER: {repr(e)[:200]}")

    if args.target in ("panda", "both"):
        print("\n========== PANDA_NO_TENDON.XML (9 DoF) ==========")
        for k in knobs:
            try:
                run_one("panda", "xml/franka_emika_panda/panda_no_tendon.xml", k, qpos_safe=qpos_panda_safe)
            except Exception as e:
                print(f"  OUTER: {repr(e)[:200]}")


if __name__ == "__main__":
    main()
