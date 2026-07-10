"""Phase-A pilot for the A5 action-side linear-velocity VJP.

This script reuses the existing restored-anchor CSV.  It first fits one
action-to-object-linear-velocity matrix per anchor, then learns the mapping
from an anchor context to that matrix with anchor-level train/val/test splits.
Random FD directions not used by the per-anchor fit remain a genuine
directional holdout.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


DEFAULT_DATA = Path(
    "analysis/2026-07-09_arx_pusher/stage2_phase_grid_full132/"
    "a5_stage2_phase_grid_full132.csv"
)
DEFAULT_OUT_DIR = Path(
    "analysis/2026-07-09_arx_pusher/stage2_phase_grid_full132/action_vjp_v2_phase_a"
)
ACTION_DIM = 6
TARGET_DIM = 3


def _loads_vec(value, length=None):
    values = np.asarray(json.loads(value), dtype=np.float64)
    if length is not None:
        if values.size < length:
            values = np.pad(values, (0, length - values.size))
        values = values[:length]
    return values


def _float(row, key, default=0.0):
    value = row.get(key, "")
    return float(default if value in (None, "") else value)


def _as_bool(value):
    return str(value).strip().lower() not in ("", "0", "false", "no", "none")


def _rms(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= eps:
        return None
    return float(np.dot(a, b) / denom)


def _random_index(name):
    if not name.startswith("random"):
        return None
    try:
        return int(name[len("random"):])
    except ValueError:
        return None


def _contact_history(row, length=6):
    trace = _loads_vec(row.get("nominal_contact_trace", "[]"))
    if trace.size == 0:
        return np.zeros(length, dtype=np.float64)
    center = trace.size // 2
    history = trace[: center + 1]
    if history.size < length:
        history = np.pad(history, (length - history.size, 0), mode="edge")
    return history[-length:]


def _context(row):
    """Causal-ish Phase-A context using only current and pre-anchor fields."""
    scalar = np.asarray(
        [
            _float(row, "obj_x"),
            _float(row, "obj_y"),
            _float(row, "obj_z"),
            _float(row, "speed"),
            _float(row, "anchor_step"),
            _float(row, "response_steps"),
        ],
        dtype=np.float64,
    )
    vectors = [
        _loads_vec(row["qpos"], ACTION_DIM),
        _loads_vec(row["qvel"], ACTION_DIM),
        _loads_vec(row["nominal_anchor_pos"], 3),
        _loads_vec(row["nominal_anchor_qvel"], 6),
        _loads_vec(row["nominal_pre_disp"], 3),
        _loads_vec(row["nominal_pre_qvel_delta"], 6),
        _contact_history(row),
    ]
    return np.concatenate([scalar, *vectors]).astype(np.float32)


@dataclass
class AnchorRecord:
    anchor_id: int
    obj_y: float
    speed: float
    anchor_step: int
    context: np.ndarray
    matrix: np.ndarray
    fit_rank: int
    fit_condition: float
    fit_rel_rmse: float
    hold_rel_rmse: float
    hold_response_cosine: float | None
    signal_rms: np.ndarray
    hold_v: np.ndarray
    hold_y: np.ndarray


def _load_grouped_rows(path, require_keep):
    grouped = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if require_keep and not _as_bool(row.get("keep", True)):
                continue
            grouped[int(row["anchor_id"])].append(row)
    return grouped


def _fit_anchor(anchor_id, rows, fit_random_max, min_hold_rows):
    fit_rows = []
    hold_rows = []
    for row in rows:
        random_idx = _random_index(row["direction"])
        if random_idx is not None and random_idx > fit_random_max:
            hold_rows.append(row)
        else:
            fit_rows.append(row)

    if len(hold_rows) < min_hold_rows:
        random_rows = sorted(
            [row for row in fit_rows if _random_index(row["direction"]) is not None],
            key=lambda row: _random_index(row["direction"]),
        )
        move = random_rows[-min(min_hold_rows - len(hold_rows), len(random_rows)):]
        move_ids = {id(row) for row in move}
        fit_rows = [row for row in fit_rows if id(row) not in move_ids]
        hold_rows.extend(move)

    fit_v = np.stack([_loads_vec(row["direction_vec"], ACTION_DIM) for row in fit_rows])
    fit_y = np.stack([_loads_vec(row["local_vel_response"], 6)[:TARGET_DIM] for row in fit_rows])
    hold_v = np.stack([_loads_vec(row["direction_vec"], ACTION_DIM) for row in hold_rows])
    hold_y = np.stack([_loads_vec(row["local_vel_response"], 6)[:TARGET_DIM] for row in hold_rows])

    coef, _, rank, singular = np.linalg.lstsq(fit_v, fit_y, rcond=None)
    matrix = coef.T
    fit_pred = fit_v @ coef
    hold_pred = hold_v @ coef
    fit_rel = _rms(fit_pred - fit_y) / (_rms(fit_y) + 1e-12)
    hold_rel = _rms(hold_pred - hold_y) / (_rms(hold_y) + 1e-12)
    condition = float(singular[0] / max(singular[-1], 1e-12)) if singular.size else float("inf")
    all_y = np.stack([_loads_vec(row["local_vel_response"], 6)[:TARGET_DIM] for row in rows])
    signal_rms = np.sqrt(np.mean(all_y * all_y, axis=0))
    first = rows[0]
    return AnchorRecord(
        anchor_id=anchor_id,
        obj_y=_float(first, "obj_y"),
        speed=_float(first, "speed"),
        anchor_step=int(_float(first, "anchor_step")),
        context=_context(first),
        matrix=matrix.astype(np.float32),
        fit_rank=int(rank),
        fit_condition=condition,
        fit_rel_rmse=fit_rel,
        hold_rel_rmse=hold_rel,
        hold_response_cosine=_cosine(hold_pred, hold_y),
        signal_rms=signal_rms.astype(np.float32),
        hold_v=hold_v.astype(np.float32),
        hold_y=hold_y.astype(np.float32),
    )


def _stable_seed(seed, label):
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def _split_records(records, seed, ood_y_min, val_frac, test_frac):
    ood = sorted([record for record in records if record.obj_y >= ood_y_min], key=lambda r: r.anchor_id)
    candidates = [record for record in records if record.obj_y < ood_y_min]
    by_y = defaultdict(list)
    for record in candidates:
        by_y[round(record.obj_y, 6)].append(record)

    train = []
    val = []
    test_id = []
    for obj_y, group in sorted(by_y.items()):
        group = sorted(group, key=lambda record: record.anchor_id)
        rng = np.random.default_rng(_stable_seed(seed, f"y={obj_y}"))
        order = rng.permutation(len(group)).tolist()
        shuffled = [group[idx] for idx in order]
        n_test = max(1, int(round(len(group) * test_frac)))
        n_val = max(1, int(round(len(group) * val_frac)))
        if n_test + n_val >= len(group):
            n_val = max(0, len(group) - n_test - 1)
        test_id.extend(shuffled[:n_test])
        val.extend(shuffled[n_test:n_test + n_val])
        train.extend(shuffled[n_test + n_val:])

    return {
        "train": sorted(train, key=lambda r: r.anchor_id),
        "val": sorted(val, key=lambda r: r.anchor_id),
        "test_id": sorted(test_id, key=lambda r: r.anchor_id),
        "test_ood": ood,
    }


class MatrixMLP(torch.nn.Module):
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(context_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, TARGET_DIM * ACTION_DIM),
        )

    def forward(self, context):
        return self.net(context).reshape(-1, TARGET_DIM, ACTION_DIM)


def _stack(records):
    return (
        np.stack([record.context for record in records]),
        np.stack([record.matrix for record in records]),
    )


def _scaled_matrix(matrix, row_scale):
    return matrix / row_scale.reshape(1, TARGET_DIM, 1)


def _torch_objective(pred, target, cosine_weight):
    huber = torch.nn.functional.smooth_l1_loss(pred, target, beta=0.1)
    pred_flat = pred.reshape(-1, TARGET_DIM, ACTION_DIM)
    target_flat = target.reshape(-1, TARGET_DIM, ACTION_DIM)
    target_norm = torch.linalg.vector_norm(target_flat, dim=-1)
    cosine = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=-1, eps=1e-8)
    active = target_norm > 1e-7
    cosine_loss = (1.0 - cosine[active]).mean() if active.any() else pred.sum() * 0.0
    return huber + cosine_weight * cosine_loss


def _train_one(split, args, run_seed, x_mean, x_std, row_scale):
    train_x, train_y = _stack(split["train"])
    val_x, val_y = _stack(split["val"])
    train_x = (train_x - x_mean) / x_std
    val_x = (val_x - x_mean) / x_std
    train_y = _scaled_matrix(train_y, row_scale)
    val_y = _scaled_matrix(val_y, row_scale)

    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MatrixMLP(train_x.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tx = torch.tensor(train_x, dtype=torch.float32, device=device)
    ty = torch.tensor(train_y, dtype=torch.float32, device=device)
    vx = torch.tensor(val_x, dtype=torch.float32, device=device)
    vy = torch.tensor(val_y, dtype=torch.float32, device=device)

    best = None
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _torch_objective(model(tx), ty, args.cosine_weight)
        loss.backward()
        optimizer.step()
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = float(_torch_objective(model(vx), vy, args.cosine_weight).item())
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_loss": val_loss})
        if best is None or val_loss < best["val_loss"] - args.min_improvement:
            best = history[-1]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.cpu().eval()
    return model, {"seed": run_seed, "best": best, "history": history, "device": str(device)}


def _predict_mlp(model, records, x_mean, x_std, row_scale):
    if not records:
        return np.zeros((0, TARGET_DIM, ACTION_DIM), dtype=np.float32)
    x = np.stack([record.context for record in records])
    x = (x - x_mean) / x_std
    with torch.no_grad():
        scaled = model(torch.tensor(x, dtype=torch.float32)).numpy()
    return scaled * row_scale.reshape(1, TARGET_DIM, 1)


def _predict_nearest(train_records, records, x_mean, x_std):
    train_x = (np.stack([record.context for record in train_records]) - x_mean) / x_std
    test_x = (np.stack([record.context for record in records]) - x_mean) / x_std
    output = []
    for context in test_x:
        idx = int(np.argmin(np.sum((train_x - context) ** 2, axis=1)))
        output.append(train_records[idx].matrix)
    return np.stack(output)


def _vjp_lambdas(seed, num_random=8):
    basis = np.eye(TARGET_DIM, dtype=np.float64)
    rng = np.random.default_rng(seed)
    random_lambdas = rng.normal(size=(num_random, TARGET_DIM))
    random_lambdas /= np.linalg.norm(random_lambdas, axis=1, keepdims=True)
    return np.concatenate([basis, random_lambdas], axis=0)


def _sign_agreement(pred, truth):
    threshold = max(float(np.linalg.norm(truth)) * 1e-5, 1e-12)
    mask = np.abs(truth) > threshold
    if not mask.any():
        return None
    return float(np.mean(np.sign(pred[mask]) == np.sign(truth[mask])))


def _evaluate_model(name, records, predictions, lambdas, signal_threshold):
    anchor_rows = []
    all_matrix_error = []
    all_matrix_truth = []
    response_errors = []
    response_truth = []
    vjp_cosines = []
    sign_scores = []
    y_cosines = []
    nonzero_y = 0
    active_y = 0

    for record, pred in zip(records, predictions):
        truth = record.matrix.astype(np.float64)
        pred = pred.astype(np.float64)
        all_matrix_error.append((pred - truth).reshape(-1))
        all_matrix_truth.append(truth.reshape(-1))
        hold_pred = record.hold_v @ pred.T
        response_errors.append((hold_pred - record.hold_y).reshape(-1))
        response_truth.append(record.hold_y.reshape(-1))

        local_cosines = []
        local_signs = []
        for lam in lambdas:
            truth_grad = truth.T @ lam
            pred_grad = pred.T @ lam
            cosine = _cosine(pred_grad, truth_grad)
            if cosine is not None:
                local_cosines.append(cosine)
                vjp_cosines.append(cosine)
            sign = _sign_agreement(pred_grad, truth_grad)
            if sign is not None:
                local_signs.append(sign)
                sign_scores.append(sign)

        y_truth = truth[1]
        y_pred = pred[1]
        y_cosine = _cosine(y_pred, y_truth)
        if y_cosine is not None:
            y_cosines.append(y_cosine)
        is_active_y = float(np.linalg.norm(y_truth)) > signal_threshold
        active_y += int(is_active_y)
        nonzero_y += int(is_active_y and float(np.linalg.norm(y_pred)) > signal_threshold)
        anchor_rows.append(
            {
                "anchor_id": record.anchor_id,
                "model": name,
                "obj_y": record.obj_y,
                "speed": record.speed,
                "anchor_step": record.anchor_step,
                "signal_rms_x": float(record.signal_rms[0]),
                "signal_rms_y": float(record.signal_rms[1]),
                "signal_rms_z": float(record.signal_rms[2]),
                "matrix": json.dumps(truth.tolist()),
                "pred_matrix": json.dumps(pred.tolist()),
                "matrix_rel_rmse": _rms(pred - truth) / (_rms(truth) + 1e-12),
                "hold_rel_rmse": _rms(hold_pred - record.hold_y) / (_rms(record.hold_y) + 1e-12),
                "vjp_cosine_mean": float(np.mean(local_cosines)) if local_cosines else None,
                "vjp_sign_agreement": float(np.mean(local_signs)) if local_signs else None,
                "y_vjp_cosine": y_cosine,
                "y_vjp_truth_norm": float(np.linalg.norm(y_truth)),
                "y_vjp_pred_norm": float(np.linalg.norm(y_pred)),
            }
        )

    matrix_error = np.concatenate(all_matrix_error) if all_matrix_error else np.zeros(0)
    matrix_truth = np.concatenate(all_matrix_truth) if all_matrix_truth else np.zeros(0)
    response_error = np.concatenate(response_errors) if response_errors else np.zeros(0)
    response_target = np.concatenate(response_truth) if response_truth else np.zeros(0)
    metrics = {
        "num_anchors": len(records),
        "matrix_relative_rmse": _rms(matrix_error) / (_rms(matrix_truth) + 1e-12),
        "hold_direction_relative_rmse": _rms(response_error) / (_rms(response_target) + 1e-12),
        "vjp_cosine_mean": float(np.mean(vjp_cosines)) if vjp_cosines else None,
        "vjp_cosine_median": float(np.median(vjp_cosines)) if vjp_cosines else None,
        "vjp_sign_agreement": float(np.mean(sign_scores)) if sign_scores else None,
        "y_vjp_cosine_mean": float(np.mean(y_cosines)) if y_cosines else None,
        "y_vjp_cosine_median": float(np.median(y_cosines)) if y_cosines else None,
        "active_y_anchors": active_y,
        "nonzero_predicted_y_on_active": nonzero_y,
        "nonzero_predicted_y_rate": nonzero_y / max(active_y, 1),
    }
    return metrics, anchor_rows


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fit-random-max", type=int, default=16)
    parser.add_argument("--min-hold-rows", type=int, default=8)
    parser.add_argument("--require-keep", action="store_true", default=True)
    parser.add_argument("--allow-filtered", dest="require_keep", action="store_false")
    parser.add_argument("--ood-y-min", type=float, default=0.095)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=4701)
    parser.add_argument("--num-train-seeds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-improvement", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cosine-weight", type=float, default=0.2)
    parser.add_argument("--signal-threshold", type=float, default=1e-10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    grouped = _load_grouped_rows(args.data, args.require_keep)
    records = []
    dropped = []
    for anchor_id, rows in sorted(grouped.items()):
        try:
            record = _fit_anchor(anchor_id, rows, args.fit_random_max, args.min_hold_rows)
        except Exception as exc:
            dropped.append({"anchor_id": anchor_id, "reason": f"{type(exc).__name__}:{exc}"})
            continue
        if record.fit_rank < ACTION_DIM:
            dropped.append({"anchor_id": anchor_id, "reason": f"rank={record.fit_rank}"})
            continue
        records.append(record)
    if len(records) < 12:
        raise RuntimeError(f"too few full-rank anchors: {len(records)}")

    split = _split_records(records, args.seed, args.ood_y_min, args.val_frac, args.test_frac)
    if not split["train"] or not split["val"] or not split["test_id"] or not split["test_ood"]:
        raise RuntimeError({key: len(value) for key, value in split.items()})

    train_x, train_matrices = _stack(split["train"])
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    train_row_rms = np.sqrt(np.mean(train_matrices * train_matrices, axis=2))
    row_scale = np.quantile(train_row_rms, 0.75, axis=0)
    row_scale = np.maximum(row_scale, 1e-6).astype(np.float32)

    runs = []
    trained = []
    for idx in range(args.num_train_seeds):
        run_seed = args.seed + 1009 * idx
        model, info = _train_one(split, args, run_seed, x_mean, x_std, row_scale)
        val_pred = _predict_mlp(model, split["val"], x_mean, x_std, row_scale)
        val_metrics, _ = _evaluate_model(
            "mlp", split["val"], val_pred, _vjp_lambdas(args.seed + 17), args.signal_threshold
        )
        info["val_metrics"] = val_metrics
        runs.append(info)
        trained.append(model)
    best_idx = min(range(len(runs)), key=lambda idx: runs[idx]["best"]["val_loss"])
    model = trained[best_idx]

    global_matrix = np.mean(train_matrices, axis=0)
    lambdas = _vjp_lambdas(args.seed + 29)
    summary_metrics = {}
    eval_rows = []
    for split_name in ("val", "test_id", "test_ood"):
        split_records = split[split_name]
        predictions = {
            "zero": np.zeros((len(split_records), TARGET_DIM, ACTION_DIM), dtype=np.float32),
            "global": np.repeat(global_matrix[None], len(split_records), axis=0),
            "nearest": _predict_nearest(split["train"], split_records, x_mean, x_std),
            "mlp": _predict_mlp(model, split_records, x_mean, x_std, row_scale),
        }
        summary_metrics[split_name] = {}
        for model_name, pred in predictions.items():
            metrics, rows = _evaluate_model(
                model_name, split_records, pred, lambdas, args.signal_threshold
            )
            summary_metrics[split_name][model_name] = metrics
            for row in rows:
                row["split"] = split_name
            eval_rows.extend(rows)

    anchor_rows = []
    split_by_id = {
        record.anchor_id: split_name
        for split_name, split_records in split.items()
        for record in split_records
    }
    for record in records:
        anchor_rows.append(
            {
                "anchor_id": record.anchor_id,
                "split": split_by_id[record.anchor_id],
                "obj_y": record.obj_y,
                "speed": record.speed,
                "anchor_step": record.anchor_step,
                "fit_rank": record.fit_rank,
                "fit_condition": record.fit_condition,
                "fit_relative_rmse": record.fit_rel_rmse,
                "hold_direction_relative_rmse": record.hold_rel_rmse,
                "hold_response_cosine": record.hold_response_cosine,
                "signal_rms_x": float(record.signal_rms[0]),
                "signal_rms_y": float(record.signal_rms[1]),
                "signal_rms_z": float(record.signal_rms[2]),
                "matrix": json.dumps(record.matrix.tolist()),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "a5_action_vjp_v2_phase_a_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "context_dim": int(train_x.shape[1]),
            "hidden_dim": args.hidden_dim,
            "x_mean": torch.tensor(x_mean, dtype=torch.float32),
            "x_std": torch.tensor(x_std, dtype=torch.float32),
            "row_scale": torch.tensor(row_scale, dtype=torch.float32),
            "split_anchor_ids": {
                key: [record.anchor_id for record in value] for key, value in split.items()
            },
            "args": vars(args),
        },
        checkpoint_path,
    )
    _write_csv(args.out_dir / "a5_action_vjp_v2_phase_a_anchors.csv", anchor_rows)
    _write_csv(args.out_dir / "a5_action_vjp_v2_phase_a_eval.csv", eval_rows)
    payload = {
        "description": "A5 action-side VJP v2 Phase-A linear-velocity pilot",
        "data": str(args.data),
        "target": "local_vel_response[:3]",
        "num_input_rows": sum(len(rows) for rows in grouped.values()),
        "num_full_rank_anchors": len(records),
        "dropped_anchors": dropped,
        "context_dim": int(train_x.shape[1]),
        "fit_random_max": args.fit_random_max,
        "min_hold_rows": args.min_hold_rows,
        "split_anchor_ids": {
            key: [record.anchor_id for record in value] for key, value in split.items()
        },
        "split_counts": {key: len(value) for key, value in split.items()},
        "row_scale": row_scale.tolist(),
        "training_runs": runs,
        "selected_training_run": best_idx,
        "metrics": summary_metrics,
        "checkpoint": str(checkpoint_path),
    }
    summary_path = args.out_dir / "a5_action_vjp_v2_phase_a_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    print(f"[a5-vjp-v2-a] anchors={len(records)} split={payload['split_counts']}")
    print(f"[a5-vjp-v2-a] context_dim={train_x.shape[1]} row_scale={row_scale.tolist()}")
    for split_name in ("test_id", "test_ood"):
        print(f"[a5-vjp-v2-a] {split_name}")
        for model_name in ("zero", "global", "nearest", "mlp"):
            metrics = summary_metrics[split_name][model_name]
            print(
                f"  {model_name:7s} hold_rel={metrics['hold_direction_relative_rmse']:.4f} "
                f"vjp_cos={metrics['vjp_cosine_median']} y_cos={metrics['y_vjp_cosine_median']} "
                f"nonzero={metrics['nonzero_predicted_y_rate']:.3f}"
            )
    print(f"[a5-vjp-v2-a] wrote {summary_path}")
    print(f"[a5-vjp-v2-a] wrote {checkpoint_path}")


if __name__ == "__main__":
    raise SystemExit(main())
