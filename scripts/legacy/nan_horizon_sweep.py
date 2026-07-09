"""
Follow-up to A-E: now that we've ruled out weld and ball-joint as
isolated culprits, test whether the NaN is driven by

  (i)  total simulated motion (long horizon makes any chain blow up)
  (ii) number of links (mass matrix conditioning at scale)

Vary horizon at 5/10/20/30/60/120 across:
  D: 5-segment cable WITH weld   (what we already know: ok at horizon=30)
  E: 26-segment cable NO weld    (what we already know: nan@29)
  F: 26-segment cable WITH weld  (== original cable.xml)
  G: 5-segment cable NO weld     (free-falling short chain — does it nan late?)

For each, report when (if ever) backward NaNs.
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


def make_scene(dt=2e-3, substeps=4):
    return gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt, substeps=substeps, substeps_local=substeps, requires_grad=True
        ),
        show_viewer=False,
    )


SCENE_FILES = {
    "D":  f"{SCENES_DIR}/cable_5seg.xml",
    "E":  f"{SCENES_DIR}/cable_no_weld.xml",
    "F":  "xml/cable.xml",
    "G":  None,  # built inline as a no-weld 5-seg variant
}


def write_g_scene_if_missing():
    path = f"{SCENES_DIR}/cable_5seg_no_weld.xml"
    if os.path.exists(path):
        return path
    body = """<mujoco model="cable_5seg_no_weld">
    <worldbody>
        <body name="B0" pos="0 0 1.0">
            <freejoint/>
            <geom type="capsule" size="0.005 0.025" mass="0.001" rgba="0.8 0.6 0.4 1"/>
            <body name="B1" pos="0 0 -0.055">
                <joint type="ball" damping="0.01" armature="0.001"/>
                <geom type="capsule" size="0.005 0.025" mass="0.001" rgba="0.8 0.6 0.4 1"/>
                <body name="B2" pos="0 0 -0.055">
                    <joint type="ball" damping="0.01" armature="0.001"/>
                    <geom type="capsule" size="0.005 0.025" mass="0.001" rgba="0.8 0.6 0.4 1"/>
                    <body name="B3" pos="0 0 -0.055">
                        <joint type="ball" damping="0.01" armature="0.001"/>
                        <geom type="capsule" size="0.005 0.025" mass="0.001" rgba="0.8 0.6 0.4 1"/>
                        <body name="B4" pos="0 0 -0.055">
                            <joint type="ball" damping="0.01" armature="0.001"/>
                            <geom type="capsule" size="0.005 0.025" mass="0.001" rgba="0.8 0.6 0.4 1"/>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
    with open(path, "w") as f:
        f.write(body)
    return path


def run_one(label: str, file: str, horizon: int):
    scene = make_scene()
    scene.add_entity(gs.morphs.Plane())
    ent = scene.add_entity(gs.morphs.MJCF(file=file))
    scene.build()
    scene.reset()

    t0 = time.time()
    try:
        for _ in range(horizon):
            scene.step()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[{label}|h={horizon}] FORWARD FAIL: {e!r}")
        return
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

    print(f"[{label}|h={horizon:>3}] n_dofs={ent.n_dofs:>3}  fwd={t_fwd:.2f}s  loss={loss.item():.5f}  bwd={t_bwd:.2f}s  status={status}  mem={gpu_mem_mb()}MB")


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    g_path = write_g_scene_if_missing()
    SCENE_FILES["G"] = g_path

    horizons = [5, 10, 20, 30, 60, 120]
    for label in ["D", "G", "E", "F"]:
        for h in horizons:
            run_one(label, SCENE_FILES[label], h)


if __name__ == "__main__":
    main()
