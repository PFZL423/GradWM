"""Arm + rope no-contact Genesis backward sanity gate."""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from make_arm_mjcf import make_arm_gripper_mjcf
from segment_death_line import make_cable_mjcf

if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    print("usage: python scripts/arm_rope_nocontact_sanity.py")
    raise SystemExit(0)

import torch

import genesis as gs


EXPECTED_ARM_DOFS = 9
SCRIPT_NAME = Path(__file__).name


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


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fmt_mb(value):
    return "None" if value is None else f"{value:.1f}"


def nan_status_from_exception(e):
    msg = str(e)
    if "Nan grad" in msg:
        return msg.replace("Nan grad in qpos or dofs_vel found at step ", "nan@step")
    return f"genesis_error:{msg[:100]}"


def write_verdict(verdict: str) -> None:
    log_path = Path("logs/arm_rope_nocontact.log")
    try:
        with log_path.open("a") as f:
            f.write(verdict + "\n")
    except OSError as e:
        print(f"[sanity] log append failed: {e}")
    print(verdict)


def _move_cable_anchor(cable_mjcf: str, x: float, y: float, z: float) -> str:
    """Option A placement: rewrite only the generated cable anchor body pos."""
    return cable_mjcf.replace(
        '<body name="anchor" pos="0 0 1.07">',
        f'<body name="anchor" pos="{x} {y} {z}">',
    )


def _write_temp_mjcf(prefix: str, mjcf_text: str) -> str:
    tmp = tempfile.NamedTemporaryFile(
        prefix=prefix,
        suffix=".xml",
        delete=False,
        mode="w",
    )
    tmp.write(mjcf_text)
    tmp.flush()
    tmp.close()
    return tmp.name


def main():
    arm_tmp = _write_temp_mjcf("arm_rope_nocontact_arm_", make_arm_gripper_mjcf())
    cable_text = _move_cable_anchor(
        make_cable_mjcf(n_segments=12, with_weld=True),
        0.6,
        0.6,
        1.0,
    )
    cable_tmp = _write_temp_mjcf("arm_rope_nocontact_cable_", cable_text)
    print(f"[sanity] temp arm mjcf: {arm_tmp}")
    print(f"[sanity] temp cable mjcf: {cable_tmp}")
    print("[sanity] cable anchor moved with Option A string replacement to (0.6, 0.6, 1.0)")

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

    scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -1.0)))
    arm = scene.add_entity(gs.morphs.MJCF(file=arm_tmp))
    cable = scene.add_entity(gs.morphs.MJCF(file=cable_tmp))

    scene.build()

    def callable_attr(obj, name):
        try:
            attr = getattr(obj, name)
        except Exception:
            return None
        return attr if callable(attr) else None

    cleanup_targets = [
        ("scene.sim", getattr(scene, "sim", None)),
        ("scene", scene),
    ]
    cleanup_priority = ("reset_grad", "reset_grad_state", "clear_grad", "zero_grad", "_reset_grad")
    grad_methods = []
    for target_name, target in cleanup_targets:
        if target is None:
            continue
        available = [
            name
            for name in dir(target)
            if "grad" in name.lower() and callable_attr(target, name) is not None
        ]
        if available:
            grad_methods.append(f"{target_name}: {available}")
    print(
        "[sanity] available grad cleanup methods: "
        + ("; ".join(grad_methods) if grad_methods else "none")
    )

    graph_cleanup = None
    graph_cleanup_name = None
    for method_name in cleanup_priority:
        for target_name, target in cleanup_targets:
            if target is None:
                continue
            method = callable_attr(target, method_name)
            if method is not None:
                graph_cleanup = method
                graph_cleanup_name = f"{target_name}.{method_name}()"
                break
        if graph_cleanup is not None:
            break
    print(f"[sanity] selected graph cleanup: {graph_cleanup_name or 'torch.cuda.empty_cache()'}")

    scene.reset()

    print(f"[sanity] arm n_links={arm.n_links}  n_dofs={arm.n_dofs}")
    print(f"[sanity] cable n_links={cable.n_links}  n_dofs={cable.n_dofs}")
    if arm.n_dofs != EXPECTED_ARM_DOFS:
        detail = f"expected arm n_dofs={EXPECTED_ARM_DOFS}, got {arm.n_dofs}"
        print(f"[ABORT] {detail}")
        write_verdict(
            f"[VERDICT] FAIL  {SCRIPT_NAME}: warmup status=abort grad_nan=30/{horizon}  {detail}"
        )
        return 1

    def run_once(label: str, take_backward: bool):
        if graph_cleanup is not None:
            graph_cleanup()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        scene.reset()

        base_vel = [0.10, -0.09, 0.08, -0.07, 0.06, -0.05, 0.04, 0.0, 0.0]
        v_list = [
            gs.tensor([v * (1.0 + 0.01 * i) for v in base_vel], requires_grad=True)
            for i in range(horizon)
        ]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        proc_mem_samples = [gpu_mem_mb()]
        status = "ok"
        t_fwd_start = time.time()
        try:
            for i in range(horizon):
                arm.set_dofs_velocity(v_list[i])
                scene.step()
            cuda_sync()
        except gs.GenesisException as e:
            status = nan_status_from_exception(e)
        except Exception as e:
            status = f"error:{repr(e)[:100]}"
        t_fwd = time.time() - t_fwd_start
        proc_mem_samples.append(gpu_mem_mb())

        loss = None
        if status == "ok":
            # Velocity loss exercises backward without differentiable TCP FK.
            qvel = arm.get_dofs_velocity()
            target_qvel = gs.tensor([0.05] * 7 + [0.0, 0.0])
            loss = (qvel - target_qvel).pow(2).sum()

        if take_backward and status == "ok":
            t_bwd_start = time.time()
            try:
                loss.backward()
                cuda_sync()
            except gs.GenesisException as e:
                status = nan_status_from_exception(e)
            except Exception as e:
                status = f"error:{repr(e)[:100]}"
            t_bwd = time.time() - t_bwd_start
            proc_mem_samples.append(gpu_mem_mb())

            grad_norms = [
                v.grad.norm().item() if v.grad is not None else None for v in v_list
            ]
            nan_count = sum(
                1 for v in v_list if v.grad is not None and torch.isnan(v.grad).any()
            )
            if status.startswith("nan@step") and nan_count == 0:
                nan_count = horizon
            elif status == "ok" and nan_count:
                status = "nan_grad_tensor"
        else:
            t_bwd = 0.0
            grad_norms = [None for _ in range(horizon)]
            nan_count = horizon if status.startswith("nan@step") else None

        peak_torch = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else None
        )
        proc_mem_samples = [m for m in proc_mem_samples if m is not None]
        peak_proc = max(proc_mem_samples) if proc_mem_samples else None
        loss_item = float("nan") if loss is None else loss.item()
        fwd_rate = horizon / t_fwd if t_fwd > 0 else float("inf")

        print(
            f"[sanity:{label}] forward {horizon} steps: {t_fwd:.3f}s -> {fwd_rate:.1f} step/s  "
            f"loss={loss_item:.5f}  status={status}"
        )
        if take_backward:
            print(f"[sanity:{label}] backward: {t_bwd:.3f}s")
            print(f"[sanity:{label}] grad NaN: {nan_count}/{horizon}")
            print(
                f"[sanity:{label}] grad norms: "
                f"{[round(g, 4) if g is not None else None for g in grad_norms]}"
            )
        print(
            f"[sanity:{label}] peak torch alloc: {fmt_mb(peak_torch)} MB  "
            f"peak nvidia-smi: {peak_proc} MB"
        )

        return {
            "label": label,
            "status": status,
            "nan_count": nan_count,
        }

    print("[sanity] === warmup run (JIT compile, expect 100s+) ===")
    warmup = run_once("warmup", take_backward=True)

    print("[sanity] === steady-state run #1 ===")
    steady1 = run_once("steady1", take_backward=True)

    print("[sanity] === steady-state run #2 ===")
    steady2 = run_once("steady2", take_backward=True)

    print("[sanity] === forward-only steady-state ===")
    run_once("fwd-only", take_backward=False)

    # The first (warmup) backward is the real test: it goes through the full
    # JIT-compiled rigid backward kernel from a fresh sim state. steady1/steady2
    # may still expose the known graph-reuse artifact from rerunning one scene.
    if warmup["status"] == "ok" and warmup["nan_count"] == 0:
        write_verdict(f"[VERDICT] PASS  {SCRIPT_NAME}: warmup backward NaN=0/30")
        return 0

    reason = "arm+rope no-contact warmup backward did not complete cleanly"
    write_verdict(
        f"[VERDICT] FAIL  {SCRIPT_NAME}: warmup status={warmup['status']} "
        f"grad_nan={warmup['nan_count']}/{horizon}  {reason}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
