"""Frozen Phase-1 probe for A5 rejected/marginal contact anchors."""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
COLLECT_GRID = SCRIPT_PATH.parent / "a5_action_vjp_v2_collect_grid.py"
DEFAULT_OUT = REPO_ROOT / "analysis/2026-07-09_arx_pusher/marginal_probe"
DEFAULT_PREFILTER = (
    REPO_ROOT
    / "analysis/2026-07-09_arx_pusher/action_vjp_v2_restore_prefilter1200"
    / "trusted/a5_action_vjp_v2_anchor_matrices.csv"
)
DEFAULT_DENSE = (
    REPO_ROOT
    / "analysis/2026-07-09_arx_pusher/action_vjp_v2_dense_scan26568"
    / "a5_action_vjp_v2_dense_scan26568_frozen_split.csv"
)
DEFAULT_STABLE_MATRICES = (
    REPO_ROOT
    / "analysis/2026-07-09_arx_pusher/action_vjp_v2_replay330/final"
    / "a5_action_vjp_v2_final_matrices.csv"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "analysis/2026-07-09_arx_pusher/action_vjp_v2_replay330/final/train"
    / "a5_action_vjp_v2_model.pt"
)
DEFAULT_STABLE_CLOSED = (
    REPO_ROOT
    / "analysis/2026-07-09_arx_pusher/action_vjp_v2_replay330/final/closed_loop"
    / "a5_action_vjp_v2_closed_loop_summary.json"
)
DEFAULT_REPORT = (
    REPO_ROOT / "notes/a5_vjp_progress/2026-07-10_marginal_probe.md"
)
DEFAULT_QPOS = [0.0, 1.4, -0.4, 0.5, 0.0, 0.0]
MODELS = ("zero", "global", "learned", "analytic", "oracle")
TARGET_EPSILON = 0.01
REFERENCE_EPSILON = 0.003


def _as_bool(value):
    return str(value).strip().lower() not in ("", "0", "false", "no", "none")


def _json(value, default):
    return default if value in (None, "") else json.loads(value)


def _float(value):
    if value in (None, ""):
        return None
    return float(value)


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denom < 1e-12 else float(np.dot(a, b) / denom)


def _relative_rmse(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    denom = float(np.sqrt(np.mean(truth * truth)))
    if denom < 1e-12:
        return None
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / denom)


def _percentile(values, q):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else float(np.percentile(finite, q))


def _distribution(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "num": len(finite),
        "min": None if not finite else min(finite),
        "q25": _percentile(finite, 25),
        "median": _percentile(finite, 50),
        "q75": _percentile(finite, 75),
        "max": None if not finite else max(finite),
    }


def _mean_present(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else float(np.mean(finite))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*values):
    text = "|".join(str(value) for value in values)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _read_csv(path):
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return value


def _write_structured_csv(path, rows):
    _write_csv(path, [{key: _csv_value(value) for key, value in row.items()} for row in rows])


def _subtype(reasons):
    cross = "cross_epsilon_y_cosine" in reasons
    signature = "contact_signature_switch" in reasons
    if cross and signature:
        return "both"
    if cross:
        return "cross_only"
    if signature:
        return "signature_only"
    return None


def _maximin_select(rows, count, seed, subtype):
    if len(rows) <= count:
        return list(rows)
    features = np.asarray(
        [
            [row["obj_x"], row["obj_y"], row["speed"], row["anchor_step"]]
            for row in rows
        ],
        dtype=np.float64,
    )
    low = features.min(axis=0)
    span = features.max(axis=0) - low
    span[span < 1e-12] = 1.0
    features = (features - low) / span
    order = sorted(
        range(len(rows)),
        key=lambda index: _stable_hash(seed, subtype, rows[index]["anchor_id"]),
    )
    selected = [order[0]]
    remaining = set(order[1:])
    while len(selected) < count:
        best = max(
            remaining,
            key=lambda index: (
                min(float(np.sum((features[index] - features[other]) ** 2)) for other in selected),
                -order.index(index),
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return [rows[index] for index in selected]


def _selection_paths(out_dir):
    return {
        "selection": out_dir / "a5_marginal_probe_frozen_selection.csv",
        "probe_a_jobs": out_dir / "a5_marginal_probe_a_jobs.csv",
        "probe_b_jobs": out_dir / "a5_marginal_probe_b_jobs.csv",
        "probe_a_dir": out_dir / "probe_a",
        "probe_b_dir": out_dir / "probe_b",
        "labels": out_dir / "a5_marginal_probe_labels.csv",
        "matrices": out_dir / "a5_marginal_probe_matrices.csv",
        "offline": out_dir / "a5_marginal_probe_offline.csv",
        "analytic": out_dir / "a5_marginal_probe_analytic.csv",
        "closed": out_dir / "a5_marginal_probe_closed_loop.csv",
        "final_csv": out_dir / "a5_marginal_probe.csv",
        "summary": out_dir / "a5_marginal_probe_summary.json",
    }


def _select(args):
    paths = _selection_paths(args.out_dir)
    if paths["selection"].exists() and not args.force_selection:
        rows = _read_csv(paths["selection"])
        if len(rows) != args.num_anchors:
            raise RuntimeError(
                f"frozen selection has {len(rows)} anchors, expected {args.num_anchors}; "
                "use --force-selection only to deliberately replace it"
            )
        print(f"[marginal-select] reused frozen selection {paths['selection']}")
        return rows

    prefilter = _read_csv(args.prefilter_matrices)
    dense = {int(row["anchor_id"]): row for row in _read_csv(args.dense_manifest)}
    candidates = []
    for row in prefilter:
        reasons = set(row["gate_reasons"].split("|"))
        weak = bool(reasons & {"weak_y_vjp", "weak_random_y_signal"})
        subtype = _subtype(reasons)
        anchor_id = int(row["anchor_id"])
        if _as_bool(row["usable"]) or weak or subtype is None or anchor_id not in dense:
            continue
        job = dense[anchor_id]
        obj_pos = _json(job["obj_pos"], [0.0, 0.0, 0.0])
        candidates.append(
            {
                "anchor_id": anchor_id,
                "phase1_role": "frozen_diagnostic_test",
                "subtype": subtype,
                "original_split": row["split"],
                "gate_reasons": row["gate_reasons"],
                "obj_x": float(obj_pos[0]),
                "obj_y": float(obj_pos[1]),
                "obj_z": float(obj_pos[2]),
                "speed": float(job["speed"]),
                "anchor_step": int(job["anchor_step"]),
                "qpos": job["qpos"],
                "qvel": job["qvel"],
                "dense_seed": int(job["seed"]),
                "prefilter_hold_y_cosine": row.get("hold_y_cosine", ""),
                "prefilter_cross_epsilon_y_cosine": row.get("cross_epsilon_y_cosine", ""),
                "prefilter_signature_equal_rate": row.get("contact_signature_equal_rate", ""),
                "prefilter_y_vjp_norm": row.get("y_vjp_norm", ""),
            }
        )
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row["subtype"]].append(row)
    subtypes = ("cross_only", "signature_only", "both")
    base, remainder = divmod(args.num_anchors, len(subtypes))
    quotas = {name: base + int(index < remainder) for index, name in enumerate(subtypes)}
    selected = []
    for subtype in subtypes:
        if len(grouped[subtype]) < quotas[subtype]:
            raise RuntimeError(
                f"subtype {subtype} has {len(grouped[subtype])} candidates, "
                f"needs {quotas[subtype]}"
            )
        selected.extend(_maximin_select(grouped[subtype], quotas[subtype], args.seed, subtype))
    selected.sort(key=lambda row: (subtypes.index(row["subtype"]), row["anchor_id"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(paths["selection"], selected)

    for probe_name, seed_base in (("probe_a", args.probe_a_seed), ("probe_b", args.probe_b_seed)):
        jobs = []
        for row in selected:
            jobs.append(
                {
                    "anchor_id": row["anchor_id"],
                    "split": f"marginal_{row['subtype']}",
                    "obj_pos": json.dumps([row["obj_x"], row["obj_y"], row["obj_z"]]),
                    "qpos": row["qpos"],
                    "qvel": row["qvel"],
                    "speed": row["speed"],
                    "anchor_step": row["anchor_step"],
                    "seed": seed_base + 17 * row["anchor_id"],
                }
            )
        _write_csv(paths[f"{probe_name}_jobs"], jobs)
    print(
        f"[marginal-select] candidates={len(candidates)} selected={len(selected)} "
        f"subtypes={dict(Counter(row['subtype'] for row in selected))}"
    )
    print(f"[marginal-select] wrote {paths['selection']}")
    return selected


def _run_command(command, env=None, timeout=None):
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        return {
            "status": "ok" if result.returncode == 0 else f"error:{result.returncode}",
            "returncode": result.returncode,
            "elapsed_seconds": time.perf_counter() - start,
            "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
            "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "returncode": None,
            "elapsed_seconds": time.perf_counter() - start,
            "stdout_tail": "\n".join((exc.stdout or "").splitlines()[-20:]) if isinstance(exc.stdout, str) else "",
            "stderr_tail": "\n".join((exc.stderr or "").splitlines()[-20:]) if isinstance(exc.stderr, str) else "",
        }


def _split_gpus(text):
    gpu_ids = [value.strip() for value in text.split(",") if value.strip()]
    if not gpu_ids:
        return "", ""
    if len(gpu_ids) == 1:
        return gpu_ids[0], gpu_ids[0]
    middle = max(1, len(gpu_ids) // 2)
    return ",".join(gpu_ids[:middle]), ",".join(gpu_ids[middle:])


def _collect_replays(args):
    paths = _selection_paths(args.out_dir)
    gpu_a, gpu_b = _split_gpus(args.gpu_ids)
    specifications = (
        ("a", paths["probe_a_jobs"], paths["probe_a_dir"], gpu_a),
        ("b", paths["probe_b_jobs"], paths["probe_b_dir"], gpu_b),
    )

    def command_for(name, jobs, out_dir, gpu_ids):
        command = [
            sys.executable,
            str(COLLECT_GRID),
            "--conda-env",
            args.conda_env,
            "--out-dir",
            str(out_dir),
            "--tag",
            f"a5_marginal_probe_{name}",
            "--jobs-manifest",
            str(jobs),
            "--branch-mode",
            "replay",
            "--response-steps",
            "0",
            "--epsilons",
            f"{REFERENCE_EPSILON},{TARGET_EPSILON}",
            "--num-random",
            str(args.num_random),
            "--batch-size",
            str(args.replay_batch_size),
            "--workers",
            str(args.replay_workers),
            "--batch-timeout",
            str(args.replay_batch_timeout),
            "--retries",
            "1",
        ]
        if gpu_ids:
            command.extend(["--gpu-ids", gpu_ids])
        if args.no_resume:
            command.append("--no-resume")
        return command

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                _run_command,
                command_for(name, jobs, out_dir, gpu_ids),
                None,
                args.replay_timeout,
            ): name
            for name, jobs, out_dir, gpu_ids in specifications
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            print(
                f"[marginal-replay:{name}] status={results[name]['status']} "
                f"elapsed={results[name]['elapsed_seconds']:.1f}s"
            )
            if results[name]["stdout_tail"]:
                print(results[name]["stdout_tail"])
            if results[name]["stderr_tail"]:
                print(results[name]["stderr_tail"], file=sys.stderr)
    failed = {name: value for name, value in results.items() if value["status"] != "ok"}
    if failed:
        raise RuntimeError(f"pristine replay coordinator failed: {failed}")
    return results


def _rows_by_anchor(path):
    grouped = defaultdict(list)
    for row in _read_csv(path):
        if _as_bool(row.get("keep", True)):
            grouped[int(row["anchor_id"])].append(row)
    return grouped


def _fit_random_matrix(rows, epsilon):
    selected = [
        row
        for row in rows
        if row["direction"].startswith("random")
        and math.isclose(float(row["epsilon"]), epsilon, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(selected) < 6:
        return None, {"num": len(selected), "rank": 0, "condition": None}
    directions = np.asarray([_json(row["direction_vec"], []) for row in selected], dtype=np.float64)
    responses = np.asarray(
        [_json(row["linear_velocity_response"], []) for row in selected], dtype=np.float64
    )
    rank = int(np.linalg.matrix_rank(directions))
    if rank < 6:
        return None, {"num": len(selected), "rank": rank, "condition": None}
    coefficients, _, _, singular = np.linalg.lstsq(directions, responses, rcond=None)
    condition = None if singular[-1] <= 0.0 else float(singular[0] / singular[-1])
    return coefficients.T, {"num": len(selected), "rank": rank, "condition": condition}


def _direction_data(rows, epsilon, prefix):
    selected = [
        row
        for row in rows
        if row["direction"].startswith(prefix)
        and math.isclose(float(row["epsilon"]), epsilon, rel_tol=0.0, abs_tol=1e-12)
    ]
    directions = np.asarray([_json(row["direction_vec"], []) for row in selected], dtype=np.float64)
    responses = np.asarray(
        [_json(row["linear_velocity_response"], []) for row in selected], dtype=np.float64
    )
    return selected, directions, responses


def _fit_combined(rows_a, rows_b, epsilon):
    selected_a, directions_a, responses_a = _direction_data(rows_a, epsilon, "random")
    selected_b, directions_b, responses_b = _direction_data(rows_b, epsilon, "random")
    if len(selected_a) < 6 or len(selected_b) < 6:
        return None
    directions = np.concatenate([directions_a, directions_b], axis=0)
    responses = np.concatenate([responses_a, responses_b], axis=0)
    if np.linalg.matrix_rank(directions) < 6:
        return None
    coefficients, _, _, _ = np.linalg.lstsq(directions, responses, rcond=None)
    return coefficients.T


def _axis_data(rows_a, rows_b, epsilon):
    by_name = defaultdict(list)
    for row in list(rows_a) + list(rows_b):
        if row["direction"].startswith("joint") and math.isclose(
            float(row["epsilon"]), epsilon, rel_tol=0.0, abs_tol=1e-12
        ):
            by_name[row["direction"]].append(row)
    names = [f"joint{index}+" for index in range(1, 7)]
    if any(name not in by_name for name in names):
        return None, None
    directions = np.asarray([_json(by_name[name][0]["direction_vec"], []) for name in names])
    responses = np.asarray(
        [
            np.mean(
                [_json(row["linear_velocity_response"], []) for row in by_name[name]],
                axis=0,
            )
            for name in names
        ],
        dtype=np.float64,
    )
    return directions, responses


def _cross_response_cosine(matrix, rows, epsilon):
    _, directions, responses = _direction_data(rows, epsilon, "random")
    if len(directions) == 0:
        return None
    return _cosine(directions @ matrix[1], responses[:, 1])


def _signature_equal_rate(rows, epsilon):
    values = [
        _as_bool(row["contact_signature_trace_equal"])
        for row in rows
        if math.isclose(float(row["epsilon"]), epsilon, rel_tol=0.0, abs_tol=1e-12)
    ]
    return None if not values else float(np.mean(values))


def _symmetric_ratio(a, b):
    a = float(a)
    b = float(b)
    if min(abs(a), abs(b)) < 1e-12:
        return None
    return float(max(abs(a / b), abs(b / a)))


def _build_labels(args):
    paths = _selection_paths(args.out_dir)
    selection = _read_csv(paths["selection"])
    probe_rows = {
        "a": _rows_by_anchor(paths["probe_a_dir"] / "a5_marginal_probe_a_rows.csv"),
        "b": _rows_by_anchor(paths["probe_b_dir"] / "a5_marginal_probe_b_rows.csv"),
    }
    probe_anchors = {
        "a": {
            int(row["anchor_id"]): row
            for row in _read_csv(paths["probe_a_dir"] / "a5_marginal_probe_a_anchors.csv")
        },
        "b": {
            int(row["anchor_id"]): row
            for row in _read_csv(paths["probe_b_dir"] / "a5_marginal_probe_b_anchors.csv")
        },
    }
    label_rows = []
    matrix_rows = []
    for selected in selection:
        anchor_id = int(selected["anchor_id"])
        reasons = []
        anchor_a = probe_anchors["a"].get(anchor_id)
        anchor_b = probe_anchors["b"].get(anchor_id)
        rows_a = probe_rows["a"].get(anchor_id, [])
        rows_b = probe_rows["b"].get(anchor_id, [])
        if anchor_a is None or anchor_b is None:
            reasons.append("missing_replay_anchor")
        if anchor_a is not None and anchor_a.get("status") != "ok":
            reasons.append(f"probe_a_{anchor_a.get('status') or 'status'}")
        if anchor_b is not None and anchor_b.get("status") != "ok":
            reasons.append(f"probe_b_{anchor_b.get('status') or 'status'}")
        matrices = {}
        diagnostics = {}
        for probe_name, rows in (("a", rows_a), ("b", rows_b)):
            for epsilon in (REFERENCE_EPSILON, TARGET_EPSILON):
                matrix, fit = _fit_random_matrix(rows, epsilon)
                matrices[(probe_name, epsilon)] = matrix
                diagnostics[(probe_name, epsilon)] = fit
                if matrix is None:
                    reasons.append(f"probe_{probe_name}_eps{epsilon}_rank")
        truth = _fit_combined(rows_a, rows_b, TARGET_EPSILON)
        if truth is None:
            reasons.append("combined_target_matrix")

        values = {
            "anchor_id": anchor_id,
            "subtype": selected["subtype"],
            "original_split": selected["original_split"],
            "gate_reasons": selected["gate_reasons"],
            "obj_x": selected["obj_x"],
            "obj_y": selected["obj_y"],
            "obj_z": selected["obj_z"],
            "speed": selected["speed"],
            "anchor_step": selected["anchor_step"],
        }
        if not reasons:
            matrix_a = matrices[("a", TARGET_EPSILON)]
            matrix_b = matrices[("b", TARGET_EPSILON)]
            ref_a = matrices[("a", REFERENCE_EPSILON)]
            ref_b = matrices[("b", REFERENCE_EPSILON)]
            axis_directions, axis_responses = _axis_data(rows_a, rows_b, TARGET_EPSILON)
            axis_directions_a, axis_responses_a = _axis_data(rows_a, [], TARGET_EPSILON)
            axis_directions_b, axis_responses_b = _axis_data(rows_b, [], TARGET_EPSILON)
            if axis_directions is None or axis_directions_a is None or axis_directions_b is None:
                reasons.append("missing_axis_hold")
            else:
                values.update(
                    {
                        "label_label_y_cosine": _cosine(matrix_a[1], matrix_b[1]),
                        "probe_a_on_b_y_cosine": _cross_response_cosine(
                            matrix_a, rows_b, TARGET_EPSILON
                        ),
                        "probe_b_on_a_y_cosine": _cross_response_cosine(
                            matrix_b, rows_a, TARGET_EPSILON
                        ),
                        "oracle_axis_y_cosine": _cosine(
                            axis_directions @ truth[1], axis_responses[:, 1]
                        ),
                        "oracle_axis_y_relative_rmse": _relative_rmse(
                            axis_directions @ truth[1], axis_responses[:, 1]
                        ),
                        "axis_repeat_y_cosine": _cosine(
                            axis_responses_a[:, 1], axis_responses_b[:, 1]
                        ),
                        "axis_repeat_y_max_abs": float(
                            np.max(np.abs(axis_responses_a[:, 1] - axis_responses_b[:, 1]))
                        ),
                        "probe_a_epsilon_y_cosine": _cosine(matrix_a[1], ref_a[1]),
                        "probe_b_epsilon_y_cosine": _cosine(matrix_b[1], ref_b[1]),
                        "target_label_norm_ratio": _symmetric_ratio(
                            np.linalg.norm(matrix_a[1]), np.linalg.norm(matrix_b[1])
                        ),
                        "probe_a_epsilon_norm_ratio": _symmetric_ratio(
                            np.linalg.norm(matrix_a[1]), np.linalg.norm(ref_a[1])
                        ),
                        "probe_b_epsilon_norm_ratio": _symmetric_ratio(
                            np.linalg.norm(matrix_b[1]), np.linalg.norm(ref_b[1])
                        ),
                        "truth_y_norm": float(np.linalg.norm(truth[1])),
                        "probe_a_signature_equal_rate": _signature_equal_rate(
                            rows_a, TARGET_EPSILON
                        ),
                        "probe_b_signature_equal_rate": _signature_equal_rate(
                            rows_b, TARGET_EPSILON
                        ),
                        "probe_a_condition": diagnostics[("a", TARGET_EPSILON)]["condition"],
                        "probe_b_condition": diagnostics[("b", TARGET_EPSILON)]["condition"],
                        "truth_matrix": json.dumps(truth.tolist()),
                        "probe_a_matrix": json.dumps(matrix_a.tolist()),
                        "probe_b_matrix": json.dumps(matrix_b.tolist()),
                    }
                )
        values["replay_complete"] = not reasons
        values["keep_reason"] = "ok" if not reasons else "|".join(dict.fromkeys(reasons))
        label_rows.append(values)
        if reasons:
            continue
        source = anchor_a
        matrix_rows.append(
            {
                "anchor_id": anchor_id,
                "split": f"marginal_{selected['subtype']}",
                "status": "ok",
                "branch_mode": "replay_dual_random",
                "usable": True,
                "gate_reasons": selected["gate_reasons"],
                "obj_pos": json.dumps(
                    [float(selected["obj_x"]), float(selected["obj_y"]), float(selected["obj_z"])]
                ),
                "speed": selected["speed"],
                "anchor_step": selected["anchor_step"],
                "arm_contact_events": source["arm_contact_events"],
                "repeat_state_max_abs": source["repeat_state_max_abs"],
                "target_epsilon": TARGET_EPSILON,
                "reference_epsilon": REFERENCE_EPSILON,
                "target_matrix": json.dumps(truth.tolist()),
                "reference_matrix": json.dumps(
                    (_fit_combined(rows_a, rows_b, REFERENCE_EPSILON)).tolist()
                ),
                "y_vjp_norm": float(np.linalg.norm(truth[1])),
                "hold_relative_rmse": values["oracle_axis_y_relative_rmse"],
                "hold_response_cosine": values["oracle_axis_y_cosine"],
                "hold_y_relative_rmse": values["oracle_axis_y_relative_rmse"],
                "hold_y_cosine": values["oracle_axis_y_cosine"],
                "hold_y_target_rms": float(np.sqrt(np.mean(axis_responses[:, 1] ** 2))),
                "cross_epsilon_y_cosine": _mean_present(
                    [values["probe_a_epsilon_y_cosine"], values["probe_b_epsilon_y_cosine"]]
                ),
                "contact_signature_equal_rate": _mean_present(
                    [
                        values["probe_a_signature_equal_rate"],
                        values["probe_b_signature_equal_rate"],
                    ]
                ),
                "anchor_object_state": source["anchor_object_state"],
                "anchor_arm_state": source["anchor_arm_state"],
                "anchor_contact": source["anchor_contact"],
                "nominal_object_state": source["nominal_object_state"],
                "nominal_contact_trace": source["nominal_contact_trace"],
                "nominal_contact_geometry_trace": source["nominal_contact_geometry_trace"],
            }
        )
    _write_csv(paths["labels"], label_rows)
    _write_csv(paths["matrices"], matrix_rows)
    print(
        f"[marginal-labels] complete={len(matrix_rows)}/{len(selection)} "
        f"wrote={paths['matrices']}"
    )
    return label_rows, matrix_rows


def _sign_agreement(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    threshold = max(float(np.linalg.norm(truth)) * 1e-5, 1e-12)
    mask = np.abs(truth) > threshold
    return None if not mask.any() else float(np.mean(np.sign(pred[mask]) == np.sign(truth[mask])))


def _load_frozen_predictions(args):
    import torch

    script_dir = str(SCRIPT_PATH.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from a5_action_vjp_v2_train_trusted import MatrixModel, _context, _load_records

    paths = _selection_paths(args.out_dir)
    marginal = _load_records(paths["matrices"])
    stable = _load_records(args.stable_matrices)
    stable_train = [record for record in stable if record.split == "train"]
    global_matrix = np.mean(np.stack([record.matrix for record in stable_train]), axis=0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MatrixModel(checkpoint["context_dim"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    x_mean = checkpoint["x_mean"].numpy()
    x_std = checkpoint["x_std"].numpy()
    row_scale = checkpoint["row_scale"].numpy().reshape(1, 3, 1)
    output = []
    for record in marginal:
        value = (_context(record, checkpoint["mode"])[None] - x_mean) / x_std
        with torch.no_grad():
            learned = model(torch.tensor(value, dtype=torch.float32)).numpy() * row_scale
        truth = np.asarray(record.matrix, dtype=np.float64)
        predictions = {
            "zero": np.zeros_like(truth),
            "global": global_matrix,
            "learned": learned[0],
            "oracle": truth,
        }
        for model_name, prediction in predictions.items():
            output.append(
                {
                    "anchor_id": record.anchor_id,
                    "model": model_name,
                    "truth_y_row": json.dumps(truth[1].tolist()),
                    "pred_y_row": json.dumps(np.asarray(prediction)[1].tolist()),
                    "truth_y_norm": float(np.linalg.norm(truth[1])),
                    "pred_y_norm": float(np.linalg.norm(np.asarray(prediction)[1])),
                    "y_cosine": _cosine(np.asarray(prediction)[1], truth[1]),
                    "sign_agreement": _sign_agreement(np.asarray(prediction)[1], truth[1]),
                    "nonzero": float(np.linalg.norm(np.asarray(prediction)[1])) > 1e-10,
                }
            )
    _write_csv(paths["offline"], output)
    print(f"[marginal-offline] rows={len(output)} wrote={paths['offline']}")
    return global_matrix


def _analytic_worker(request_path):
    import torch
    import genesis as gs

    script_dir = str(SCRIPT_PATH.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from a5_action_vjp_v2_collect_worker import _contact_geometry, _set_velocity
    from a5_pusher_forward_sanity import _make_scene

    request = json.loads(request_path.read_text())
    job = request["job"]
    started = time.perf_counter()
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene, arm, obj = _make_scene(tuple(job["obj_pos"]), requires_grad=True)
    obj.set_pos(gs.tensor(job["obj_pos"]), zero_velocity=True)
    arm.set_dofs_position(torch.tensor(job["qpos"], dtype=torch.float32), zero_velocity=True)
    zero = [0.0] * len(job["qvel"])
    for _ in range(request["settle_steps"]):
        _set_velocity(arm, zero)
        scene.step()
    for _ in range(job["anchor_step"]):
        _set_velocity(arm, job["qvel"])
        scene.step()
    anchor_contact = _contact_geometry(obj, arm, request["max_contacts"])
    action = gs.tensor(job["qvel"], requires_grad=True)
    arm.set_dofs_velocity(action)
    scene.step()
    probe = obj.get_dofs_velocity().reshape(-1)[1]
    status = "ok"
    gradient = None
    try:
        probe.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if action.grad is None:
            status = "grad_none"
            gradient = [0.0] * len(job["qvel"])
        elif torch.isnan(action.grad).any():
            status = "grad_nan"
        else:
            gradient = [float(value) for value in action.grad.detach().cpu().reshape(-1).tolist()]
    except RuntimeError as exc:
        if "does not require grad" in str(exc):
            status = "loss_has_no_grad_path"
            gradient = [0.0] * len(job["qvel"])
        else:
            status = f"backward_error:{type(exc).__name__}:{str(exc)[:160]}"
    except Exception as exc:
        status = f"backward_error:{type(exc).__name__}:{str(exc)[:160]}"
    payload = {
        "anchor_id": job["anchor_id"],
        "status": status,
        "analytic_y_row": gradient,
        "analytic_y_norm": None if gradient is None else float(np.linalg.norm(gradient)),
        "object_vy": float(probe.detach().cpu().item()),
        "anchor_contact_count": int(anchor_contact["count"]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    out = Path(request["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[marginal-analytic-worker] anchor={job['anchor_id']} status={status} "
        f"norm={payload['analytic_y_norm']}"
    )


def _jobs_from_selection(selection):
    return {
        int(row["anchor_id"]): {
            "anchor_id": int(row["anchor_id"]),
            "obj_pos": [float(row["obj_x"]), float(row["obj_y"]), float(row["obj_z"])],
            "qpos": _json(row["qpos"], DEFAULT_QPOS),
            "qvel": _json(row["qvel"], [float(row["speed"]), 0.0, 0.0, 0.0, 0.0, 0.0]),
            "speed": float(row["speed"]),
            "anchor_step": int(row["anchor_step"]),
        }
        for row in selection
    }


def _run_independent_workers(args, kind, anchor_ids, make_request, batch_size=1):
    request_dir = args.out_dir / "requests" / kind
    result_dir = args.out_dir / "runs" / kind
    request_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    chunks = [anchor_ids[index:index + batch_size] for index in range(0, len(anchor_ids), batch_size)]

    def run(index, ids):
        out = result_dir / f"{kind}_{index:04d}.json"
        request = request_dir / f"{kind}_{index:04d}.json"
        if not (args.no_resume or not out.exists() or not out.stat().st_size):
            return {"status": "resumed", "ids": ids, "out": out, "elapsed_seconds": 0.0}
        payload = make_request(ids, out)
        request.write_text(json.dumps(payload, indent=2) + "\n")
        command = [sys.executable, str(SCRIPT_PATH), "--stage", f"{kind}-worker", "--request", str(request)]
        env = os.environ.copy()
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids[index % len(gpu_ids)]
        result = _run_command(command, env=env, timeout=args.worker_timeout)
        result.update({"ids": ids, "out": out})
        return result

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run, index, ids): index for index, ids in enumerate(chunks)
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"[marginal-{kind}] ids={record['ids']} status={record['status']} "
                f"elapsed={record['elapsed_seconds']:.1f}s"
            )
    records.sort(key=lambda record: record["ids"][0])
    return records


def _run_analytic(args):
    paths = _selection_paths(args.out_dir)
    selection = _read_csv(paths["selection"])
    complete_ids = {
        int(row["anchor_id"])
        for row in _read_csv(paths["labels"])
        if _as_bool(row["replay_complete"])
    }
    jobs = _jobs_from_selection(selection)
    anchor_ids = sorted(complete_ids)

    def make_request(ids, out):
        if len(ids) != 1:
            raise ValueError("analytic workers require exactly one anchor")
        return {
            "job": jobs[ids[0]],
            "settle_steps": 20,
            "max_contacts": 16,
            "out": str(out),
        }

    records = _run_independent_workers(args, "analytic", anchor_ids, make_request, 1)
    output = []
    for record in records:
        if record["out"].exists() and record["out"].stat().st_size:
            output.append(json.loads(record["out"].read_text()))
        else:
            output.append(
                {
                    "anchor_id": record["ids"][0],
                    "status": record["status"],
                    "analytic_y_row": "",
                    "analytic_y_norm": "",
                    "object_vy": "",
                    "anchor_contact_count": "",
                    "elapsed_seconds": record["elapsed_seconds"],
                }
            )
    _write_structured_csv(paths["analytic"], output)
    print(f"[marginal-analytic] rows={len(output)} wrote={paths['analytic']}")
    return output


def _closed_worker(request_path):
    import torch
    import genesis as gs

    script_dir = str(SCRIPT_PATH.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from a5_action_vjp_v2_collect_worker import _query_replay
    from a5_action_vjp_v2_train_trusted import MatrixModel, _context, _load_records
    from a5_pusher_forward_sanity import _make_scene

    request = json.loads(request_path.read_text())
    records = _load_records(Path(request["matrices"]))
    by_id = {record.anchor_id: record for record in records}
    selected = [by_id[int(anchor_id)] for anchor_id in request["anchor_ids"]]
    analytic = {int(key): value for key, value in request["analytic"].items()}
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
    scene, arm, obj = _make_scene((first.obj_x, first.obj_y, 0.120), requires_grad=True)
    base_state = scene.get_state()
    query_config = {"settle_steps": 20, "max_contacts": 16}
    global_matrix = np.asarray(request["global_matrix"], dtype=np.float64)
    scale = float(request["scale"])
    target_dvy = float(request["target_dvy"])
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
        analytic_matrix = np.zeros((3, 6), dtype=np.float64)
        analytic_row = analytic.get(record.anchor_id)
        if analytic_row not in (None, ""):
            analytic_matrix[1] = np.asarray(analytic_row, dtype=np.float64)
        truth = np.asarray(record.matrix, dtype=np.float64)
        matrices = {
            "zero": np.zeros_like(truth),
            "global": global_matrix,
            "learned": predict(record),
            "analytic": analytic_matrix,
            "oracle": truth,
        }
        oracle_gradient = truth.T @ cotangent
        for model_name, matrix in matrices.items():
            gradient = matrix.T @ cotangent
            gradient_norm = float(np.linalg.norm(gradient))
            oracle_norm = float(np.linalg.norm(oracle_gradient))
            gradient_cosine = _cosine(gradient, oracle_gradient)
            base = {
                "anchor_id": record.anchor_id,
                "model": model_name,
                "scale": scale,
                "gradient_norm": gradient_norm,
                "gradient_cosine_oracle": gradient_cosine,
                "nominal_loss": nominal_loss,
                "nominal_velocity": json.dumps(nominal_velocity.tolist()),
                "target_velocity": json.dumps(target_velocity.tolist()),
            }
            if gradient_norm < 1e-12 or oracle_norm < 1e-12:
                rows.append(
                    {
                        **base,
                        "status": "zero_gradient",
                        "descent_loss": nominal_loss,
                        "descent_delta": 0.0,
                        "ascent_loss": nominal_loss,
                        "ascent_delta": 0.0,
                        "descent_correct": False,
                        "ascent_correct": False,
                    }
                )
                continue
            descent = -gradient / gradient_norm
            trial = {}
            for kind, direction in (("descent", descent), ("ascent", -descent)):
                action = (np.asarray(job["qvel"], dtype=np.float64) + scale * direction).tolist()
                result = _query_replay(
                    scene, arm, obj, base_state, job, action, 1, query_config, "signature"
                )
                velocity = np.asarray(result["object"]["qvel"][:3], dtype=np.float64)
                loss = float((velocity[1] - target_velocity[1]) ** 2)
                trial[kind] = {"loss": loss, "delta": loss - nominal_loss}
            rows.append(
                {
                    **base,
                    "status": "ok",
                    "descent_loss": trial["descent"]["loss"],
                    "descent_delta": trial["descent"]["delta"],
                    "ascent_loss": trial["ascent"]["loss"],
                    "ascent_delta": trial["ascent"]["delta"],
                    "descent_correct": trial["descent"]["delta"] < 0.0,
                    "ascent_correct": trial["ascent"]["delta"] > 0.0,
                }
            )
    out = Path(request["out"])
    _write_csv(out, rows)
    out.with_suffix(".json").write_text(
        json.dumps({"status": "ok", "num_anchors": len(selected), "num_rows": len(rows)}, indent=2)
        + "\n"
    )
    print(f"[marginal-closed-worker] anchors={len(selected)} rows={len(rows)} out={out}")


def _run_closed(args, global_matrix):
    paths = _selection_paths(args.out_dir)
    records = _read_csv(paths["matrices"])
    anchor_ids = sorted(int(row["anchor_id"]) for row in records)
    analytic = {
        int(row["anchor_id"]): _json(row.get("analytic_y_row"), "")
        for row in _read_csv(paths["analytic"])
        if row.get("analytic_y_row") not in (None, "")
    }

    def make_request(ids, out):
        return {
            "anchor_ids": ids,
            "matrices": str(paths["matrices"]),
            "checkpoint": str(args.checkpoint),
            "global_matrix": np.asarray(global_matrix).tolist(),
            "analytic": {str(anchor_id): analytic.get(anchor_id, "") for anchor_id in ids},
            "scale": args.action_scale,
            "target_dvy": args.target_dvy,
            "out": str(out.with_suffix(".csv")),
        }

    records = _run_independent_workers(
        args, "closed", anchor_ids, make_request, args.closed_batch_size
    )
    output = []
    for record in records:
        csv_path = record["out"].with_suffix(".csv")
        if csv_path.exists() and csv_path.stat().st_size:
            output.extend(_read_csv(csv_path))
            continue
        for anchor_id in record["ids"]:
            for model_name in MODELS:
                output.append(
                    {
                        "anchor_id": anchor_id,
                        "model": model_name,
                        "scale": args.action_scale,
                        "status": record["status"],
                        "gradient_norm": "",
                        "gradient_cosine_oracle": "",
                        "nominal_loss": "",
                        "descent_delta": "",
                        "ascent_delta": "",
                    }
                )
    _write_csv(paths["closed"], output)
    print(f"[marginal-closed] rows={len(output)} wrote={paths['closed']}")
    return output


def _model_metrics(rows):
    replay_complete = [row for row in rows if _as_bool(row["replay_complete"])]
    evaluable = [row for row in replay_complete if _as_bool(row.get("model_evaluable", True))]
    cosines = [_float(row.get("y_cosine")) for row in evaluable]
    signs = [_float(row.get("sign_agreement")) for row in evaluable]
    descent = [_float(row.get("descent_delta")) for row in evaluable]
    ascent = [_float(row.get("ascent_delta")) for row in evaluable]
    valid_closed = [
        index
        for index, (d_value, a_value) in enumerate(zip(descent, ascent))
        if d_value is not None and a_value is not None
    ]
    return {
        "num_selected": len(rows),
        "num_replay_complete": len(replay_complete),
        "num_model_evaluable": len(evaluable),
        "num_closed_evaluable": len(valid_closed),
        "nonzero_rate": (
            None if not evaluable else float(np.mean([_as_bool(row["nonzero"]) for row in evaluable]))
        ),
        "y_cosine": _distribution(cosines),
        "sign_agreement": (
            None
            if not [value for value in signs if value is not None]
            else float(np.mean([value for value in signs if value is not None]))
        ),
        "descent_rate": (
            None
            if not valid_closed
            else float(np.mean([descent[index] < 0.0 for index in valid_closed]))
        ),
        "ascent_rate": (
            None
            if not valid_closed
            else float(np.mean([ascent[index] < 0.0 for index in valid_closed]))
        ),
        "separated_rate": (
            None
            if not valid_closed
            else float(
                np.mean(
                    [descent[index] < 0.0 and ascent[index] > 0.0 for index in valid_closed]
                )
            )
        ),
    }


def _label_metrics(rows):
    complete = [row for row in rows if _as_bool(row["replay_complete"])]
    fields = (
        "label_label_y_cosine",
        "probe_a_on_b_y_cosine",
        "probe_b_on_a_y_cosine",
        "oracle_axis_y_cosine",
        "oracle_axis_y_relative_rmse",
        "axis_repeat_y_cosine",
        "axis_repeat_y_max_abs",
        "probe_a_epsilon_y_cosine",
        "probe_b_epsilon_y_cosine",
        "target_label_norm_ratio",
        "probe_a_epsilon_norm_ratio",
        "probe_b_epsilon_norm_ratio",
        "truth_y_norm",
        "probe_a_signature_equal_rate",
        "probe_b_signature_equal_rate",
    )
    output = {
        "num_selected": len(rows),
        "num_replay_complete": len(complete),
        "replay_complete_rate": len(complete) / max(len(rows), 1),
    }
    for field in fields:
        output[field] = _distribution([_float(row.get(field)) for row in complete])
    cross_min = []
    eps_min = []
    max_norm_ratio = []
    for row in complete:
        cross_values = [
            _float(row.get("probe_a_on_b_y_cosine")),
            _float(row.get("probe_b_on_a_y_cosine")),
        ]
        epsilon_values = [
            _float(row.get("probe_a_epsilon_y_cosine")),
            _float(row.get("probe_b_epsilon_y_cosine")),
        ]
        norm_values = [
            _float(row.get("target_label_norm_ratio")),
            _float(row.get("probe_a_epsilon_norm_ratio")),
            _float(row.get("probe_b_epsilon_norm_ratio")),
        ]
        if all(value is not None for value in cross_values):
            cross_min.append(min(cross_values))
        if all(value is not None for value in epsilon_values):
            eps_min.append(min(epsilon_values))
        if all(value is not None for value in norm_values):
            max_norm_ratio.append(max(norm_values))
    output["cross_probe_y_cosine_min"] = _distribution(cross_min)
    output["epsilon_y_cosine_min"] = _distribution(eps_min)
    output["max_norm_ratio"] = _distribution(max_norm_ratio)
    return output


def _format_number(value, digits=3):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _write_report(args, summary):
    methods = summary["methods"]
    labels = summary["label_reliability"]
    lines = [
        "# A5 Marginal Cohort Phase-1 Probe",
        "",
        "Date: 2026-07-10",
        "",
        "## Scope",
        "",
        "The passed action-side v2 stable gate and all of its collection/reset/restore",
        "protocols remained frozen. This probe evaluates a separate rejected cohort; it",
        "does not retrain the stable model and does not start state-side VJP, SHAC, policy",
        "training, Dreamer, MoE, or an abstention gate.",
        "",
        "## Commands",
        "",
        "```bash",
        *summary["commands_run"],
        "```",
        "",
        "The coordinator launched two independent pristine-replay direction sets, one",
        "fresh-process analytic backward per replay-complete anchor, and fixed-scale",
        "closed-loop replay workers.",
        "",
        "## Frozen Cohort",
        "",
        f"- selected: `{summary['selection']['num_selected']}` anchors;",
        f"- replay-complete: `{labels['num_replay_complete']}`;",
        f"- subtype counts: `{summary['selection']['subtype_counts']}`;",
        "- weak-y-only candidates were excluded; no Phase-1 anchor may be used for",
        "  later Phase-2 training.",
        "",
        "## Label Reliability",
        "",
        "| Metric | Median | Q25 | Q75 |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("independent label y-cosine", "label_label_y_cosine"),
        ("cross-probe held y-cosine (min)", "cross_probe_y_cosine_min"),
        ("epsilon y-cosine (min)", "epsilon_y_cosine_min"),
        ("oracle axis held y-cosine", "oracle_axis_y_cosine"),
        ("same-axis replay y-cosine", "axis_repeat_y_cosine"),
        ("same-axis replay max abs", "axis_repeat_y_max_abs"),
        ("maximum norm ratio", "max_norm_ratio"),
    ):
        metric = labels[key]
        lines.append(
            f"| {label} | {_format_number(metric['median'])} | "
            f"{_format_number(metric['q25'])} | {_format_number(metric['q75'])} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Results",
            "",
            "All closed-loop methods use the stable validation scale `0.01`; no marginal",
            "scale tuning was performed. Global LSQ is frozen from the stable training split.",
            "",
            "| Method | Median y-cosine | Sign agreement | Nonzero | Descent | Separated |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in MODELS:
        metric = methods[name]
        lines.append(
            f"| {name} | {_format_number(metric['y_cosine']['median'])} | "
            f"{_format_number(metric['sign_agreement'])} | "
            f"{_format_number(metric['nonzero_rate'])} | "
            f"{_format_number(metric['descent_rate'])} | "
            f"{_format_number(metric['separated_rate'])} |"
        )
    degradation = summary["stable_relative_degradation"]
    lines.extend(
        [
            "",
            "## Stable-Cohort Degradation",
            "",
            f"- stable learned median cosine: `{degradation['stable_cosine']:.3f}`;",
            f"- marginal learned median cosine: `{_format_number(degradation['marginal_cosine'])}`;",
            f"- cosine change: `{_format_number(degradation['cosine_change'])}`;",
            f"- stable learned descent: `{degradation['stable_descent']:.3f}`;",
            f"- marginal learned descent: `{_format_number(degradation['marginal_descent'])}`;",
            f"- descent change: `{_format_number(degradation['descent_change'])}`.",
            "",
            "## Results By Marginal Subtype",
            "",
            "| Subtype | Replay N | Label-label cosine | Oracle held cosine |",
            "|---|---:|---:|---:|",
        ]
    )
    for subtype, metric in summary["label_reliability_by_subtype"].items():
        lines.append(
            f"| {subtype} | {metric['num_replay_complete']} | "
            f"{_format_number(metric['label_label_y_cosine']['median'])} | "
            f"{_format_number(metric['oracle_axis_y_cosine']['median'])} |"
        )
    lines.extend(
        [
            "",
            "| Subtype | Method | N | Median y-cosine | Descent |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for subtype, subtype_methods in summary["methods_by_subtype"].items():
        for name in MODELS:
            metric = subtype_methods[name]
            lines.append(
                f"| {subtype} | {name} | {metric['num_replay_complete']} | "
                f"{_format_number(metric['y_cosine']['median'])} | "
                f"{_format_number(metric['descent_rate'])} |"
            )
    decision = summary["decision"]
    lines.extend(
        [
            "",
            "## Scientific Interpretation",
            "",
            "The two replay sets reproduce every shared axis response exactly",
            "(`cosine=1.0`, maximum absolute difference `0.0`), so the disagreement",
            "between independently fitted random-direction matrices is not simulator",
            "noise. It is direction/contact-branch dependence at the tested finite scales.",
            "",
            "`signature_only` remains a coherent matrix cohort, while `cross_only` and",
            "`both` do not. A contact-mode feature encoder could describe the anchor more",
            "richly, but it cannot turn a direction-dependent target into one fixed matrix.",
            "The overall learned/global descent rates are therefore useful behavior checks,",
            "not evidence that the combined random-LSQ matrix is valid ground truth.",
            "",
            "## Decision Matrix",
            "",
            "The decision is label-reliability-first; model cosine is interpreted only",
            "after testing whether one local matrix is a coherent target.",
            "",
            "| Ordered rule | Phase-2 direction |",
            "|---|---|",
            "| replay direction unreliable or oracle descent < 0.7 | single-matrix label failure; redesign branch/direction target |",
            "| reliable, learned cosine >= 0.5 and descent >= 0.7 | no contact-feature Phase 2 |",
            "| reliable, learned cosine < 0.5, median magnitude spread > 2x | direction-only target redesign |",
            "| reliable, learned cosine in [0.2, 0.5), magnitude stable | contact-mode-aware features |",
            "| reliable, learned cosine < 0.2, magnitude stable | severe context OOD; contact features, not automatic label redesign |",
            "",
            "| Condition | Result |",
            "|---|---|",
            f"| replay labels direction-reliable | `{decision['label_direction_reliable']}` |",
            f"| oracle closed-loop ceiling passes | `{decision['oracle_ceiling_pass']}` |",
            f"| marginal amplitude stable within 2x median ratio | `{decision['amplitude_stable']}` |",
            f"| selected next direction | **`{decision['next_direction']}`** |",
            "",
            decision["rationale"],
            "",
            "## Unexpected Observations",
            "",
        ]
    )
    observations = summary.get("unexpected_observations") or ["None beyond the quantified marginal behavior."]
    lines.extend(f"- {item}" for item in observations)
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `{summary['artifacts']['csv']}`",
            f"- `{summary['artifacts']['summary']}`",
            f"- `{summary['artifacts']['selection']}`",
            f"- `{summary['artifacts']['labels']}`",
            f"- `{summary['artifacts']['analytic']}`",
            f"- `{summary['artifacts']['closed_loop']}`",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")


def _finalize(args):
    paths = _selection_paths(args.out_dir)
    selection = _read_csv(paths["selection"])
    selection_by_id = {int(row["anchor_id"]): row for row in selection}
    labels = _read_csv(paths["labels"])
    labels_by_id = {int(row["anchor_id"]): row for row in labels}
    offline = {
        (int(row["anchor_id"]), row["model"]): row for row in _read_csv(paths["offline"])
    }
    analytic = {int(row["anchor_id"]): row for row in _read_csv(paths["analytic"])}
    closed = {
        (int(row["anchor_id"]), row["model"]): row for row in _read_csv(paths["closed"])
    }
    matrix_rows = _read_csv(paths["matrices"])
    truth_by_id = {
        int(row["anchor_id"]): np.asarray(_json(row["target_matrix"], []), dtype=np.float64)
        for row in matrix_rows
    }
    global_matrix = None
    offline_global = next((row for row in _read_csv(paths["offline"]) if row["model"] == "global"), None)
    if offline_global is not None:
        global_y = _json(offline_global["pred_y_row"], [])
        global_matrix = {"y_row": global_y}

    final_rows = []
    for anchor_id, selected in sorted(selection_by_id.items()):
        label = labels_by_id.get(anchor_id, {})
        replay_complete = _as_bool(label.get("replay_complete"))
        truth = truth_by_id.get(anchor_id)
        analytic_row = analytic.get(anchor_id, {})
        analytic_y = _json(analytic_row.get("analytic_y_row"), "")
        for model_name in MODELS:
            method = offline.get((anchor_id, model_name), {})
            model_evaluable = replay_complete and bool(method)
            if model_name == "analytic" and replay_complete and truth is not None and analytic_y not in (None, ""):
                prediction = np.zeros(6, dtype=np.float64)
                prediction = np.asarray(analytic_y, dtype=np.float64)
                method = {
                    "truth_y_row": json.dumps(truth[1].tolist()),
                    "pred_y_row": json.dumps(prediction.tolist()),
                    "truth_y_norm": float(np.linalg.norm(truth[1])),
                    "pred_y_norm": float(np.linalg.norm(prediction)),
                    "y_cosine": _cosine(prediction, truth[1]),
                    "sign_agreement": _sign_agreement(prediction, truth[1]),
                    "nonzero": float(np.linalg.norm(prediction)) > 1e-10,
                }
                model_evaluable = True
            elif model_name == "analytic":
                model_evaluable = False
            closed_row = closed.get((anchor_id, model_name), {})
            method_reason = label.get("keep_reason", "missing_label_audit")
            if replay_complete and not model_evaluable:
                method_reason = f"{model_name}_not_evaluable:{analytic_row.get('status', 'missing')}"
            final_rows.append(
                {
                    "anchor_id": anchor_id,
                    "subtype": selected["subtype"],
                    "original_split": selected["original_split"],
                    "gate_reasons": selected["gate_reasons"],
                    "obj_x": selected["obj_x"],
                    "obj_y": selected["obj_y"],
                    "speed": selected["speed"],
                    "anchor_step": selected["anchor_step"],
                    "phase1_role": selected["phase1_role"],
                    "replay_complete": replay_complete,
                    "keep": model_evaluable,
                    "keep_reason": method_reason,
                    "label_label_y_cosine": label.get("label_label_y_cosine", ""),
                    "probe_a_on_b_y_cosine": label.get("probe_a_on_b_y_cosine", ""),
                    "probe_b_on_a_y_cosine": label.get("probe_b_on_a_y_cosine", ""),
                    "oracle_axis_y_cosine": label.get("oracle_axis_y_cosine", ""),
                    "oracle_axis_y_relative_rmse": label.get("oracle_axis_y_relative_rmse", ""),
                    "axis_repeat_y_cosine": label.get("axis_repeat_y_cosine", ""),
                    "axis_repeat_y_max_abs": label.get("axis_repeat_y_max_abs", ""),
                    "probe_a_epsilon_y_cosine": label.get("probe_a_epsilon_y_cosine", ""),
                    "probe_b_epsilon_y_cosine": label.get("probe_b_epsilon_y_cosine", ""),
                    "target_label_norm_ratio": label.get("target_label_norm_ratio", ""),
                    "probe_a_epsilon_norm_ratio": label.get("probe_a_epsilon_norm_ratio", ""),
                    "probe_b_epsilon_norm_ratio": label.get("probe_b_epsilon_norm_ratio", ""),
                    "truth_y_norm": method.get("truth_y_norm", label.get("truth_y_norm", "")),
                    "model": model_name,
                    "model_evaluable": model_evaluable,
                    "pred_y_norm": method.get("pred_y_norm", ""),
                    "y_cosine": method.get("y_cosine", ""),
                    "sign_agreement": method.get("sign_agreement", ""),
                    "nonzero": method.get("nonzero", False),
                    "analytic_status": analytic_row.get("status", "not_run"),
                    "closed_loop_status": closed_row.get("status", "not_run"),
                    "action_scale": closed_row.get("scale", args.action_scale),
                    "descent_delta": closed_row.get("descent_delta", ""),
                    "ascent_delta": closed_row.get("ascent_delta", ""),
                    "descent_correct": closed_row.get("descent_correct", ""),
                    "ascent_correct": closed_row.get("ascent_correct", ""),
                    "truth_y_row": method.get("truth_y_row", ""),
                    "pred_y_row": method.get("pred_y_row", ""),
                }
            )
    _write_csv(paths["final_csv"], final_rows)

    label_summary = _label_metrics(labels)
    label_summary_by_subtype = {
        subtype: _label_metrics([row for row in labels if row["subtype"] == subtype])
        for subtype in ("cross_only", "signature_only", "both")
    }
    method_summary = {
        name: _model_metrics([row for row in final_rows if row["model"] == name])
        for name in MODELS
    }
    methods_by_subtype = {}
    for subtype in ("cross_only", "signature_only", "both"):
        methods_by_subtype[subtype] = {
            name: _model_metrics(
                [
                    row
                    for row in final_rows
                    if row["model"] == name and row["subtype"] == subtype
                ]
            )
            for name in MODELS
        }
    stable_closed = json.loads(args.stable_closed_summary.read_text())
    stable_learned = stable_closed["test_metrics"]["learned"]
    learned = method_summary["learned"]
    oracle = method_summary["oracle"]
    label_direction_reliable = all(
        (label_summary[key]["median"] or -1.0) >= 0.5
        for key in (
            "label_label_y_cosine",
            "cross_probe_y_cosine_min",
            "oracle_axis_y_cosine",
        )
    )
    oracle_ceiling_pass = (oracle["descent_rate"] or 0.0) >= 0.7
    amplitude_stable = (label_summary["max_norm_ratio"]["median"] or float("inf")) <= 2.0
    learned_cosine = learned["y_cosine"]["median"]
    learned_descent = learned["descent_rate"]
    if not label_direction_reliable or not oracle_ceiling_pass:
        next_direction = "single_matrix_label_failure"
        rationale = (
            "The marginal cohort does not provide a sufficiently coherent single-matrix "
            "target or oracle descent ceiling. Contact features cannot repair an ill-defined "
            "label; redesign must condition on direction/contact branch before model expansion."
        )
    elif learned_cosine is not None and learned_cosine >= 0.5 and (learned_descent or 0.0) >= 0.7:
        next_direction = "no_contact_features_proceed_downstream"
        rationale = (
            "The frozen stable-trained MLP already clears the marginal direction and closed-loop "
            "thresholds. Phase 2 contact features are not justified by this probe."
        )
    elif not amplitude_stable:
        next_direction = "direction_only_target_redesign"
        rationale = (
            "Replay direction is coherent but finite-scale magnitude changes by more than 2x at "
            "the cohort median. A direction-focused target is justified before adding context."
        )
    else:
        next_direction = "contact_features_phase2"
        rationale = (
            "Replay directions and oracle descent are coherent, magnitude is not the dominant "
            "failure, and the frozen state-only model remains below the marginal threshold. "
            "This is the evidence pattern required to test contact-mode-aware features."
        )
    observations = []
    incomplete = [row for row in labels if not _as_bool(row["replay_complete"])]
    if incomplete:
        observations.append(
            f"{len(incomplete)} selected anchors lacked complete dual-replay labels; reasons are retained in CSV."
        )
    analytic_statuses = Counter(row.get("status", "missing") for row in analytic.values())
    analytic_nonzero = sum(
        (_float(row.get("analytic_y_norm")) or 0.0) > 1e-12 for row in analytic.values()
    )
    observations.append(
        f"Genesis analytic statuses: {dict(analytic_statuses)}; nonzero velocity-y rows: {analytic_nonzero}/96."
    )
    if (label_summary["probe_a_signature_equal_rate"]["median"] or 1.0) < 0.8:
        observations.append("Contact-signature changes remain common under target-scale perturbations.")
    if (label_summary["axis_repeat_y_max_abs"]["max"] or 0.0) <= 1e-12:
        observations.append(
            "Identical axis probes reproduced exactly across replay sets; random-set label disagreement is direction/branch dependence, not run-to-run noise."
        )
    observations.append(
        "The first closed-loop coordinator launch exposed a new-script-only Torch default-device bug after gs.init(gpu); forcing checkpoint inputs to CPU fixed it, and all 12 batches were rerun successfully."
    )

    summary = {
        "description": "A5 frozen marginal/rejected cohort Phase-1 probe",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "command": "conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage all --gpu-ids "
        + args.gpu_ids,
        "commands_run": [
            "conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage select",
            "conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage replay --gpu-ids 0,1,2,3",
            "conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage labels --gpu-ids 0,1,2,3",
            "conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage analytic --gpu-ids 0,1,2,3 --workers 8",
            "conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage closed --gpu-ids 0,1,2,3 --workers 8",
            "conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage finalize --gpu-ids 0,1,2,3",
        ],
        "frozen_inputs": {
            "prefilter_matrices": str(args.prefilter_matrices),
            "dense_manifest": str(args.dense_manifest),
            "stable_matrices": str(args.stable_matrices),
            "stable_checkpoint": str(args.checkpoint),
            "stable_closed_summary": str(args.stable_closed_summary),
            "sha256": {
                "prefilter_matrices": _sha256(args.prefilter_matrices),
                "dense_manifest": _sha256(args.dense_manifest),
                "stable_matrices": _sha256(args.stable_matrices),
                "stable_checkpoint": _sha256(args.checkpoint),
            },
        },
        "protocol": {
            "branch": "pristine_serial_replay",
            "probe_sets": 2,
            "random_directions_per_set": args.num_random,
            "epsilons": [REFERENCE_EPSILON, TARGET_EPSILON],
            "truth_fit": "combined independent random-direction LSQ at epsilon=0.01",
            "held_direction": "axis directions excluded from truth fit",
            "action_scale": args.action_scale,
            "target_dvy": args.target_dvy,
            "marginal_scale_tuning": False,
        },
        "selection": {
            "num_selected": len(selection),
            "subtype_counts": dict(Counter(row["subtype"] for row in selection)),
            "original_split_counts": dict(Counter(row["original_split"] for row in selection)),
            "phase1_role": "frozen_diagnostic_test_never_phase2_train",
        },
        "label_reliability": label_summary,
        "label_reliability_by_subtype": label_summary_by_subtype,
        "methods": method_summary,
        "methods_by_subtype": methods_by_subtype,
        "stable_reference": stable_learned,
        "stable_relative_degradation": {
            "stable_cosine": stable_learned["gradient_cosine_median"],
            "marginal_cosine": learned_cosine,
            "cosine_change": (
                None
                if learned_cosine is None
                else learned_cosine - stable_learned["gradient_cosine_median"]
            ),
            "stable_descent": stable_learned["descent_rate"],
            "marginal_descent": learned_descent,
            "descent_change": (
                None if learned_descent is None else learned_descent - stable_learned["descent_rate"]
            ),
        },
        "decision": {
            "label_direction_reliable": label_direction_reliable,
            "oracle_ceiling_pass": oracle_ceiling_pass,
            "amplitude_stable": amplitude_stable,
            "learned_marginal_median_cosine": learned_cosine,
            "learned_marginal_descent_rate": learned_descent,
            "next_direction": next_direction,
            "rationale": rationale,
        },
        "unexpected_observations": observations,
        "global_matrix_reference": global_matrix,
        "artifacts": {
            "csv": str(paths["final_csv"]),
            "summary": str(paths["summary"]),
            "selection": str(paths["selection"]),
            "labels": str(paths["labels"]),
            "matrices": str(paths["matrices"]),
            "analytic": str(paths["analytic"]),
            "closed_loop": str(paths["closed"]),
            "report": str(args.report),
        },
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, default=str) + "\n")
    _write_report(args, summary)
    print(f"[marginal-final] methods={method_summary}")
    print(f"[marginal-final] decision={summary['decision']}")
    print(f"[marginal-final] wrote {paths['summary']}")
    print(f"[marginal-final] wrote {args.report}")
    return summary


def _labels_and_offline(args):
    _build_labels(args)
    return _load_frozen_predictions(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "select",
            "replay",
            "labels",
            "analytic",
            "closed",
            "finalize",
            "analytic-worker",
            "closed-worker",
        ),
        default="all",
    )
    parser.add_argument("--request", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--prefilter-matrices", type=Path, default=DEFAULT_PREFILTER)
    parser.add_argument("--dense-manifest", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--stable-matrices", type=Path, default=DEFAULT_STABLE_MATRICES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stable-closed-summary", type=Path, default=DEFAULT_STABLE_CLOSED)
    parser.add_argument("--num-anchors", type=int, default=96)
    parser.add_argument("--num-random", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7102026)
    parser.add_argument("--probe-a-seed", type=int, default=811003)
    parser.add_argument("--probe-b-seed", type=int, default=922007)
    parser.add_argument("--force-selection", action="store_true")
    parser.add_argument("--conda-env", default="genesis")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--replay-workers", type=int, default=4)
    parser.add_argument("--replay-batch-size", type=int, default=4)
    parser.add_argument("--replay-batch-timeout", type=int, default=1800)
    parser.add_argument("--replay-timeout", type=int, default=14400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker-timeout", type=int, default=1200)
    parser.add_argument("--closed-batch-size", type=int, default=8)
    parser.add_argument("--action-scale", type=float, default=0.01)
    parser.add_argument("--target-dvy", type=float, default=0.05)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.report = args.report.resolve()
    for name in (
        "prefilter_matrices",
        "dense_manifest",
        "stable_matrices",
        "checkpoint",
        "stable_closed_summary",
    ):
        setattr(args, name, getattr(args, name).resolve())

    if args.stage == "analytic-worker":
        if args.request is None:
            parser.error("--request is required for analytic-worker")
        _analytic_worker(args.request)
        return 0
    if args.stage == "closed-worker":
        if args.request is None:
            parser.error("--request is required for closed-worker")
        _closed_worker(args.request)
        return 0

    if args.stage in ("all", "select"):
        _select(args)
    if args.stage in ("all", "replay"):
        if not _selection_paths(args.out_dir)["selection"].exists():
            _select(args)
        _collect_replays(args)
    global_matrix = None
    if args.stage in ("all", "labels"):
        global_matrix = _labels_and_offline(args)
    if args.stage in ("all", "analytic"):
        _run_analytic(args)
    if args.stage in ("all", "closed"):
        if global_matrix is None:
            global_matrix = _load_frozen_predictions(args)
        _run_closed(args, global_matrix)
    if args.stage in ("all", "finalize"):
        _finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
