"""Probe whether a free rigid box position loss is differentiable in Genesis."""
import argparse
import json
import math
import os
import tempfile
from pathlib import Path

RUNTIME_CACHE_ROOT = Path(tempfile.gettempdir()) / "genisis_runtime"


def _configure_runtime_dirs():
    defaults = {
        "NUMBA_CACHE_DIR": RUNTIME_CACHE_ROOT / "numba",
        "MPLCONFIGDIR": RUNTIME_CACHE_ROOT / "matplotlib",
        "XDG_CACHE_HOME": RUNTIME_CACHE_ROOT / "xdg",
        "GS_CACHE_FILE_PATH": RUNTIME_CACHE_ROOT / "genesis",
        "QD_OFFLINE_CACHE_FILE_PATH": RUNTIME_CACHE_ROOT / "qdcache",
    }
    for key, path in defaults.items():
        os.environ.setdefault(key, str(path))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


_configure_runtime_dirs()

import torch
import genesis as gs


def _flat_pos(pos):
    return pos.reshape(-1, 3)[0]


def run_case(mode, horizon):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=4, substeps_local=4, requires_grad=True),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -0.001)))

    if mode == "free":
        box_pos = (0.0, 0.0, 0.5)
        velocity = [0.20, 0.0, 0.0, 0.0, 0.0, 0.0]
        target = [0.08, 0.0, 0.5]
    elif mode == "plane_contact":
        box_pos = (0.0, 0.0, 0.025)
        velocity = [0.20, 0.0, 0.0, 0.0, 0.0, 0.0]
        target = [0.08, 0.0, 0.025]
    else:
        raise ValueError(mode)

    box = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=box_pos))
    scene.build()
    scene.reset()

    v_list = [gs.tensor(velocity, requires_grad=True) for _ in range(horizon)]
    for v in v_list:
        box.set_dofs_velocity(v)
        scene.step()

    pos = _flat_pos(box.get_state().pos)
    loss = (pos - gs.tensor(target)).pow(2).sum()

    status = "ok"
    try:
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as e:
        status = f"bwd_error:{repr(e)[:160]}"

    grad_norms = []
    grad_nan = 0
    for v in v_list:
        if v.grad is None:
            grad_norms.append(None)
        elif torch.isnan(v.grad).any():
            grad_nan += 1
            grad_norms.append(float("nan"))
        else:
            grad_norms.append(float(v.grad.norm().item()))

    return {
        "mode": mode,
        "horizon": horizon,
        "n_dofs": box.n_dofs,
        "pos": [float(x) for x in pos.detach().cpu().tolist()],
        "pos_requires_grad": bool(getattr(pos, "requires_grad", False)),
        "loss": float(loss.item()),
        "status": status,
        "grad_nan": grad_nan,
        "grad_none": sum(g is None for g in grad_norms),
        "grad_norm_first": grad_norms[0],
        "grad_norm_last": grad_norms[-1],
        "grad_norm_sum": float(sum(g for g in grad_norms if g is not None and math.isfinite(g))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--modes", default="free,plane_contact")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    results = [run_case(mode.strip(), args.horizon) for mode in args.modes.split(",") if mode.strip()]
    for result in results:
        print("__BOXPOS__" + json.dumps(result))


if __name__ == "__main__":
    raise SystemExit(main())
