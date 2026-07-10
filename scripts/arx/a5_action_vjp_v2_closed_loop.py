"""Leak-free local closed-loop gate for the A5 action-side VJP v2 model."""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_QPOS = [0.0, 1.4, -0.4, 0.5, 0.0, 0.0]


def _parse_floats(value):
    return [float(item) for item in value.split(",") if item.strip()]


def _tail(value, lines=12):
    rows = [line for line in value.splitlines() if line.strip()]
    return "\n".join(rows[-lines:])


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _worker(request_path):
    import numpy as np
    import torch
    import genesis as gs

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from a5_action_vjp_v2_collect_worker import _query_replay
    from a5_action_vjp_v2_train_trusted import MatrixModel, _context, _load_records
    from a5_pusher_forward_sanity import _make_scene

    request = json.loads(request_path.read_text())
    records = _load_records(Path(request["matrices"]))
    by_id = {record.anchor_id: record for record in records}
    selected = [by_id[int(anchor_id)] for anchor_id in request["anchor_ids"]]
    if not selected:
        raise RuntimeError("closed-loop worker has no records")

    checkpoint = torch.load(request["checkpoint"], map_location="cpu", weights_only=False)
    model = MatrixModel(checkpoint["context_dim"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    x_mean = checkpoint["x_mean"].numpy()
    x_std = checkpoint["x_std"].numpy()
    row_scale = checkpoint["row_scale"].numpy().reshape(1, 3, 1)

    def predict(record):
        value = (_context(record, checkpoint["mode"])[None] - x_mean) / x_std
        with torch.no_grad():
            pred = model(torch.tensor(value, dtype=torch.float32, device="cpu")).numpy()
        return (pred * row_scale)[0]

    first = selected[0]
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene, arm, obj = _make_scene((0.306, first.obj_y, 0.120), requires_grad=True)
    base_state = scene.get_state()
    query_config = {"settle_steps": 20, "max_contacts": 16}
    global_matrix = np.asarray(request["global_matrix"], dtype=np.float64)
    scales = [float(value) for value in request["scales"]]
    target_dvy = float(request["target_dvy"])
    model_names = request["models"]
    rows = []
    for record in selected:
        job = {
            "obj_pos": [float(record.obj_x), float(record.obj_y), 0.120],
            "qpos": DEFAULT_QPOS,
            "qvel": [float(record.speed), 0.0, 0.0, 0.0, 0.0, 0.0],
            "anchor_step": int(record.anchor_step),
        }
        nominal = _query_replay(
            scene, arm, obj, base_state, job, job["qvel"], 1, query_config, "signature"
        )
        nominal_velocity = np.asarray(nominal["object"]["qvel"][:3], dtype=np.float64)
        target_velocity = nominal_velocity.copy()
        target_velocity[1] += target_dvy
        nominal_loss = float((nominal_velocity[1] - target_velocity[1]) ** 2)
        cotangent = np.zeros(3, dtype=np.float64)
        cotangent[1] = 2.0 * (nominal_velocity[1] - target_velocity[1])
        matrices = {
            "learned": predict(record),
            "oracle": np.asarray(record.matrix, dtype=np.float64),
            "global": global_matrix,
        }
        oracle_gradient = matrices["oracle"].T @ cotangent
        for model_name in model_names:
            matrix = matrices[model_name]
            gradient = matrix.T @ cotangent
            gradient_norm = float(np.linalg.norm(gradient))
            oracle_norm = float(np.linalg.norm(oracle_gradient))
            denom = gradient_norm * oracle_norm
            gradient_cosine = (
                None
                if denom < 1e-12
                else float(np.dot(gradient, oracle_gradient) / denom)
            )
            if gradient_norm < 1e-12:
                rows.append(
                    {
                        "anchor_id": record.anchor_id,
                        "split": record.split,
                        "model": model_name,
                        "scale": "",
                        "status": "zero_gradient",
                        "gradient_norm": gradient_norm,
                        "gradient_cosine_oracle": gradient_cosine,
                        "nominal_loss": nominal_loss,
                    }
                )
                continue
            descent = -gradient / gradient_norm
            for scale in scales:
                trial = {}
                for kind, direction in (("descent", descent), ("ascent", -descent)):
                    action = (np.asarray(job["qvel"], dtype=np.float64) + scale * direction).tolist()
                    result = _query_replay(
                        scene, arm, obj, base_state, job, action, 1, query_config, "signature"
                    )
                    velocity = np.asarray(result["object"]["qvel"][:3], dtype=np.float64)
                    loss = float((velocity[1] - target_velocity[1]) ** 2)
                    trial[kind] = {
                        "loss": loss,
                        "delta": loss - nominal_loss,
                        "velocity": velocity.tolist(),
                    }
                rows.append(
                    {
                        "anchor_id": record.anchor_id,
                        "split": record.split,
                        "model": model_name,
                        "scale": scale,
                        "status": "ok",
                        "gradient_norm": gradient_norm,
                        "gradient_cosine_oracle": gradient_cosine,
                        "nominal_loss": nominal_loss,
                        "nominal_velocity": json.dumps(nominal_velocity.tolist()),
                        "target_velocity": json.dumps(target_velocity.tolist()),
                        "descent_loss": trial["descent"]["loss"],
                        "descent_delta": trial["descent"]["delta"],
                        "ascent_loss": trial["ascent"]["loss"],
                        "ascent_delta": trial["ascent"]["delta"],
                        "descent_correct": trial["descent"]["delta"] < 0.0,
                        "ascent_correct": trial["ascent"]["delta"] > 0.0,
                    }
                )

    out_path = Path(request["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_path, rows)
    print(
        f"[a5-vjp-v2-closed-worker] anchors={len(selected)} rows={len(rows)} "
        f"out={out_path}"
    )


def _run_batch(args, phase, index, anchor_ids, scales, models, global_matrix):
    request_path = args.out_dir / "requests" / f"{phase}_{index:04d}.json"
    output_path = args.out_dir / "runs" / f"{phase}_{index:04d}.csv"
    if args.resume and output_path.exists() and output_path.stat().st_size:
        return {"status": "resumed", "path": output_path, "elapsed_seconds": 0.0}
    request_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        json.dumps(
            {
                "matrices": str(args.matrices),
                "checkpoint": str(args.checkpoint),
                "anchor_ids": anchor_ids,
                "scales": scales,
                "models": models,
                "target_dvy": args.target_dvy,
                "global_matrix": global_matrix,
                "out": str(output_path),
            },
            indent=2,
        )
        + "\n"
    )
    command = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "--no-capture-output",
        "python",
        str(SCRIPT_PATH),
        "--worker-request",
        str(request_path),
    ]
    start = time.perf_counter()
    env = None
    if args.gpu_ids:
        gpu_ids = [item for item in args.gpu_ids.split(",") if item.strip()]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[index % len(gpu_ids)]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=args.batch_timeout,
        env=env,
    )
    status = "ok" if process.returncode == 0 and output_path.exists() else f"error:{process.returncode}"
    return {
        "status": status,
        "path": output_path,
        "elapsed_seconds": time.perf_counter() - start,
        "stdout_tail": _tail(process.stdout),
        "stderr_tail": _tail(process.stderr),
    }


def _chunks(values, size):
    return [values[start:start + size] for start in range(0, len(values), size)]


def _run_phase(args, phase, anchor_ids, scales, models, global_matrix):
    batches = _chunks(anchor_ids, args.batch_size)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _run_batch, args, phase, index, batch, scales, models, global_matrix
            )
            for index, batch in enumerate(batches)
        ]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    failed = [record for record in records if record["status"] not in ("ok", "resumed")]
    if failed:
        raise RuntimeError(f"closed-loop {phase} worker failures: {failed}")
    rows = []
    for record in records:
        with record["path"].open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows, records


def _select_scale(rows, scales):
    candidates = []
    for scale in scales:
        values = [
            row
            for row in rows
            if row["model"] == "learned"
            and row["status"] == "ok"
            and math.isclose(float(row["scale"]), scale, abs_tol=1e-12)
        ]
        if not values:
            continue
        descent = [float(row["descent_delta"]) for row in values]
        ascent = [float(row["ascent_delta"]) for row in values]
        success = sum(value < 0.0 for value in descent) / len(descent)
        separated = sum(d < 0.0 and a > 0.0 for d, a in zip(descent, ascent)) / len(descent)
        median_delta = sorted(descent)[len(descent) // 2]
        candidates.append(
            {
                "scale": scale,
                "num": len(values),
                "descent_rate": success,
                "separated_rate": separated,
                "median_descent_delta": median_delta,
            }
        )
    if not candidates:
        raise RuntimeError("validation produced no scale candidates")
    best = max(
        candidates,
        key=lambda row: (
            row["descent_rate"],
            row["separated_rate"],
            -row["median_descent_delta"],
            -row["scale"],
        ),
    )
    return best, candidates


def _summary_by_model(rows):
    summary = {}
    for model in sorted({row["model"] for row in rows}):
        values = [row for row in rows if row["model"] == model]
        ok = [row for row in values if row["status"] == "ok"]
        cosines = [
            float(row["gradient_cosine_oracle"])
            for row in ok
            if row.get("gradient_cosine_oracle") not in (None, "")
        ]
        summary[model] = {
            "num": len(values),
            "num_ok": len(ok),
            "nonzero_rate": len(ok) / max(len(values), 1),
            "descent_rate": sum(float(row["descent_delta"]) < 0.0 for row in ok) / max(len(ok), 1),
            "ascent_rate": sum(float(row["ascent_delta"]) < 0.0 for row in ok) / max(len(ok), 1),
            "separated_rate": sum(
                float(row["descent_delta"]) < 0.0 and float(row["ascent_delta"]) > 0.0
                for row in ok
            )
            / max(len(ok), 1),
            "gradient_cosine_median": (
                None if not cosines else sorted(cosines)[len(cosines) // 2]
            ),
        }
    return summary


def _coordinator(args):
    import numpy as np

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from a5_action_vjp_v2_train_trusted import _load_records

    records = _load_records(args.matrices)
    split = {}
    for record in records:
        split.setdefault(record.split, []).append(record)
    val_ids = [record.anchor_id for record in split.get("val", [])]
    test_records = split.get("test_id", []) + split.get("test_ood", [])
    test_ids = [record.anchor_id for record in test_records]
    if len(val_ids) < args.min_val:
        raise RuntimeError(f"too few validation anchors: {len(val_ids)}")
    if len(test_ids) < args.min_heldout:
        raise RuntimeError(f"too few held-out anchors: {len(test_ids)}")
    global_matrix = np.mean(
        np.stack([record.matrix for record in split["train"]]), axis=0
    ).tolist()
    scales = _parse_floats(args.val_scales)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_rows, val_workers = _run_phase(
        args, "val", val_ids, scales, ["learned"], global_matrix
    )
    selected, scale_candidates = _select_scale(val_rows, scales)
    selected_scale = float(selected["scale"])
    test_rows, test_workers = _run_phase(
        args,
        "test",
        test_ids,
        [selected_scale],
        ["learned", "global", "oracle"],
        global_matrix,
    )
    _write_csv(args.out_dir / "a5_action_vjp_v2_closed_loop_val.csv", val_rows)
    _write_csv(args.out_dir / "a5_action_vjp_v2_closed_loop_test.csv", test_rows)
    payload = {
        "description": "A5 action-side VJP v2 local velocity closed-loop gate",
        "matrices": str(args.matrices),
        "checkpoint": str(args.checkpoint),
        "target_dvy": args.target_dvy,
        "split_counts": {key: len(value) for key, value in split.items()},
        "scale_candidates": scale_candidates,
        "selected_validation_scale": selected_scale,
        "validation_selection": selected,
        "test_metrics": _summary_by_model(test_rows),
        "test_metrics_by_split": {
            split_name: _summary_by_model(
                [row for row in test_rows if row["split"] == split_name]
            )
            for split_name in ("test_id", "test_ood")
        },
        "val_workers": val_workers,
        "test_workers": test_workers,
    }
    out = args.out_dir / "a5_action_vjp_v2_closed_loop_summary.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"[a5-vjp-v2-closed] selected_scale={selected_scale} val={selected}")
    print(f"[a5-vjp-v2-closed] test={payload['test_metrics']}")
    print(f"[a5-vjp-v2-closed] wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--conda-env", default="genesis")
    parser.add_argument("--matrices", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--target-dvy", type=float, default=0.05)
    parser.add_argument("--val-scales", default="0.001,0.003,0.01,0.03,0.1")
    parser.add_argument("--min-val", type=int, default=10)
    parser.add_argument("--min-heldout", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--batch-timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    if args.worker_request:
        _worker(args.worker_request)
        return 0
    for name in ("matrices", "checkpoint", "out_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    _coordinator(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
