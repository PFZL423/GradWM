"""
Minimal reproducer for the real culprit.

cable.xml expands via mujoco composite to:
  - njnt=26: 2 free joints (`obj`, `B_first`) + 24 ball joints
  - nq=110, nv=84

Our handwritten 26-seg ball cable has only 1 free joint and backward succeeds.
Hypothesis: Genesis 1.1.1 NaNs in backward when an MJCF file declares 2+
free joints in the same entity.

Also test minimum: 2 free Boxes — but they would need to be in one MJCF entity
to reproduce, or attached together.
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


# Variant 1: two free bodies in one MJCF entity, no chain.
TWO_FREE = """<mujoco model="two_free">
    <worldbody>
        <body name="A" pos="0 0 0.5">
            <freejoint/>
            <geom type="box" size="0.02 0.02 0.02" mass="0.001" rgba="0.8 0.6 0.4 1"/>
        </body>
        <body name="B" pos="0.1 0 0.5">
            <freejoint/>
            <geom type="box" size="0.02 0.02 0.02" mass="0.001" rgba="0.6 0.8 0.4 1"/>
        </body>
    </worldbody>
</mujoco>
"""

# Variant 2: 1 free body + 1 short ball-jointed chain attached via no-op,
#            mimicking the cable.xml's "obj + B_first..B_last" structure
#            where obj has its own free joint but is a separate body.
ONE_FREE_PLUS_CHAIN = """<mujoco model="one_free_plus_chain">
    <worldbody>
        <body name="obj" pos="0.3 0 0.5">
            <freejoint/>
            <geom type="sphere" size="0.01" mass="0.0005" rgba="0.2 0.2 0.9 1"/>
        </body>
        <body name="B0" pos="0 0 0.5">
            <freejoint/>
            <geom type="capsule" size="0.005 0.025" mass="0.001" rgba="0.8 0.6 0.4 1"/>
            <body name="B1" pos="0 0 -0.055">
                <joint type="ball" damping="0.01" armature="0.001"/>
                <geom type="capsule" size="0.005 0.025" mass="0.001" rgba="0.8 0.6 0.4 1"/>
                <body name="B2" pos="0 0 -0.055">
                    <joint type="ball" damping="0.01" armature="0.001"/>
                    <geom type="capsule" size="0.005 0.025" mass="0.001" rgba="0.8 0.6 0.4 1"/>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>
"""


def run(label: str, mjcf_text: str):
    tmp = tempfile.NamedTemporaryFile(prefix=label + "_", suffix=".xml", delete=False, mode="w")
    tmp.write(mjcf_text)
    tmp.flush()
    tmp.close()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3, substeps=4, substeps_local=4, requires_grad=True
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    ent = scene.add_entity(gs.morphs.MJCF(file=tmp.name))
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

    print("=== two_free: 2 free Boxes in 1 MJCF entity (minimal reproducer for 2+ free joints) ===")
    run("two_free", TWO_FREE)

    print("\n=== one_free_plus_chain: 1 free sphere + 3-seg ball-joint chain w/ free root ===")
    run("one_free_plus_chain", ONE_FREE_PLUS_CHAIN)


if __name__ == "__main__":
    main()
