"""
NaN onset sweep: characterise where Genesis 1.1.1's rigid+autograd backward
breaks on the bundled `xml/cable.xml` cable (26-segment ball-jointed rope).

Sweep dims (small but informative):
  - n_segments      : built into cable.xml (26 fixed); we vary horizon instead
                      as a proxy for "depth of backward chain"
  - dt              : {2e-3, 1e-3, 5e-4}
  - horizon         : {5, 10, 20, 30}
  - contact         : {none, light, hard}  -- presence and depth of cube push

For each cell, reports:
  * forward_ok / forward_time
  * backward_status in {ok, nan_at_step_N, oom, error}
  * backward_time
  * grad_finite / grad_norm  (for the autograd-traced input)
  * peak_gpu_mem_mb (nvidia-smi process RSS)

Output: csv to /home/ubuntu/genisis/logs/nan_onset_sweep.csv

Run from genesis-world/ so xml/cable.xml resolves:
  cd /home/ubuntu/genisis/external/genesis-world
  python /home/ubuntu/genisis/scripts/nan_onset_sweep.py
"""

import csv
import os
import subprocess
import time
import traceback

import torch

import genesis as gs


CONTACT_LIGHT_X = -0.30   # cube starts beside the cable, just barely touching
CONTACT_HARD_X = -0.20    # cube driven hard into the cable
CABLE_X = -0.38           # cable.xml hangs vertically near (-0.38, -0.26)
CABLE_Y = -0.26


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


def build_scene(dt, substeps, contact_mode):
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

    cube = None
    if contact_mode != "none":
        if contact_mode == "light":
            cube_x = CONTACT_LIGHT_X
        elif contact_mode == "hard":
            cube_x = CONTACT_HARD_X
        else:
            raise ValueError(contact_mode)
        cube = scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(cube_x, CABLE_Y, 0.05),
            )
        )

    scene.build()
    return scene, cable, cube


def run_cell(dt, substeps, horizon, contact_mode):
    record = dict(
        dt=dt,
        substeps=substeps,
        horizon=horizon,
        contact=contact_mode,
        forward_ok=False,
        forward_time=None,
        backward_status="not_run",
        backward_time=None,
        grad_norm=None,
        grad_nan=None,
        peak_torch_mb=None,
        peak_nvsmi_mb=None,
        loss=None,
    )

    try:
        scene, cable, cube = build_scene(dt, substeps, contact_mode)
    except Exception as e:
        record["backward_status"] = f"build_error:{repr(e)[:120]}"
        return record

    scene.reset()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Always carry an autograd-traced input so backward is non-trivial.
    if cube is not None:
        v_in = gs.tensor([0.40, 0.0, 0.0], requires_grad=True)
        v_pad = torch.zeros(3, device=v_in.device, dtype=v_in.dtype)

        try:
            t0 = time.time()
            for _ in range(horizon):
                cube.set_dofs_velocity(torch.cat([v_in, v_pad]))
                scene.step()
            torch.cuda.synchronize()
            record["forward_time"] = time.time() - t0
            record["forward_ok"] = True
        except Exception as e:
            record["backward_status"] = f"forward_error:{repr(e)[:120]}"
            return record
    else:
        v_in = gs.tensor([0.05, 0.0, 0.0], requires_grad=True)
        # apply once to cable's free-root dofs as initial perturbation
        v_pad = torch.zeros(cable.n_dofs - 3, device=v_in.device, dtype=v_in.dtype)
        try:
            cable.set_dofs_velocity(torch.cat([v_in, v_pad]))
            t0 = time.time()
            for _ in range(horizon):
                scene.step()
            torch.cuda.synchronize()
            record["forward_time"] = time.time() - t0
            record["forward_ok"] = True
        except Exception as e:
            record["backward_status"] = f"forward_error:{repr(e)[:120]}"
            return record

    tail_state = cable.get_state()
    tail_pos = tail_state.pos[-1] if tail_state.pos.dim() == 2 else tail_state.pos[0, -1]
    goal = gs.tensor([0.0, 0.0, 0.0])
    loss = torch.pow(tail_pos - goal, 2).sum()
    record["loss"] = float(loss.item())

    try:
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        record["backward_time"] = time.time() - t0
        record["backward_status"] = "ok"
        if v_in.grad is not None:
            record["grad_norm"] = float(v_in.grad.norm().item())
            record["grad_nan"] = bool(torch.isnan(v_in.grad).any().item())
    except gs.GenesisException as e:
        msg = str(e)
        if "Nan grad" in msg:
            # Format: "Nan grad in qpos or dofs_vel found at step N"
            record["backward_status"] = msg.replace("Nan grad in qpos or dofs_vel found at step ", "nan@step")
        else:
            record["backward_status"] = f"genesis_error:{msg[:120]}"
    except Exception as e:
        record["backward_status"] = f"error:{repr(e)[:120]}"

    record["peak_torch_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    record["peak_nvsmi_mb"] = gpu_mem_mb()

    return record


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    out_csv = "/home/ubuntu/genisis/logs/nan_onset_sweep.csv"
    fieldnames = [
        "dt", "substeps", "horizon", "contact",
        "forward_ok", "forward_time",
        "backward_status", "backward_time",
        "grad_norm", "grad_nan",
        "peak_torch_mb", "peak_nvsmi_mb",
        "loss",
    ]

    cells = []
    # keep substeps <= 4 to stay within 4060 8GB substeps_local memory
    for dt, substeps in [(2e-3, 4), (1e-3, 4), (5e-4, 4)]:
        for horizon in [5, 10, 20, 30]:
            for contact in ["none", "light", "hard"]:
                cells.append((dt, substeps, horizon, contact))

    print(f"[sweep] {len(cells)} cells")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, (dt, substeps, horizon, contact) in enumerate(cells):
            print(f"[sweep] cell {i+1}/{len(cells)}: dt={dt} substeps={substeps} horizon={horizon} contact={contact}")
            t0 = time.time()
            try:
                rec = run_cell(dt, substeps, horizon, contact)
            except Exception as e:
                rec = dict(zip(fieldnames, [dt, substeps, horizon, contact, False, None,
                                            f"outer_error:{repr(e)[:120]}",
                                            None, None, None, None, None, None]))
                traceback.print_exc()
            print(f"[sweep]   -> {rec['backward_status']}  fwd={rec['forward_time']}s  bwd={rec['backward_time']}s  cell wall={time.time()-t0:.1f}s")
            w.writerow(rec)
            f.flush()


if __name__ == "__main__":
    main()
