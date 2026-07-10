"""Train A5 action-side VJP models from trusted v2 anchor matrices."""

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


TARGET_DIM = 3
ACTION_DIM = 6


def _json(value, default):
    return default if value in (None, "") else json.loads(value)


def _vec(value, length):
    output = np.asarray(value, dtype=np.float64).reshape(-1)
    if output.size < length:
        output = np.pad(output, (0, length - output.size))
    return output[:length]


def _as_bool(value):
    return str(value).strip().lower() not in ("", "0", "false", "no", "none")


def _rms(value):
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value))) if value.size else 0.0


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denom < 1e-12 else float(np.dot(a, b) / denom)


@dataclass
class Record:
    anchor_id: int
    split: str
    matrix: np.ndarray
    state_context: np.ndarray
    anchor_pooled_context: np.ndarray
    anchor_topk_context: np.ndarray
    transition_pooled_context: np.ndarray
    transition_topk_context: np.ndarray
    obj_x: float
    obj_y: float
    speed: float
    anchor_step: int


def _state_context(row):
    arm = _json(row["anchor_arm_state"], {})
    obj = _json(row["anchor_object_state"], {})
    arm_qpos = _vec(arm.get("qpos", []), 6)
    arm_qvel = _vec(arm.get("qvel", []), 6)
    tip_pos = _vec(arm.get("tip_pos", []), 3)
    tip_quat = _vec(arm.get("tip_quat", []), 4)
    tip_vel = _vec(arm.get("tip_vel", []), 3)
    tip_ang = _vec(arm.get("tip_ang", []), 3)
    obj_pos = _vec(obj.get("pos", []), 3)
    obj_quat = _vec(obj.get("quat", []), 4)
    obj_qvel = _vec(obj.get("qvel", []), 6)
    scalar = np.asarray(
        [float(row["speed"]), float(row["anchor_step"])], dtype=np.float64
    )
    relative = np.concatenate(
        [tip_pos - obj_pos, tip_vel - obj_qvel[:3], tip_ang - obj_qvel[3:6]]
    )
    return np.concatenate(
        [
            scalar,
            arm_qpos,
            arm_qvel,
            tip_pos,
            tip_quat,
            tip_vel,
            tip_ang,
            obj_pos,
            obj_quat,
            obj_qvel,
            relative,
        ]
    ).astype(np.float32)


def _pooled_contact_value(contact):
    pooled = contact.get("pooled", {})
    return np.concatenate(
        [
            np.asarray(
                [
                    float(contact.get("count", 0)),
                    float(pooled.get("penetration_sum", 0.0)),
                    float(pooled.get("penetration_mean", 0.0)),
                    float(pooled.get("penetration_max", 0.0)),
                ],
                dtype=np.float64,
            ),
            _vec(pooled.get("position_object_mean", []), 3),
            _vec(pooled.get("normal_object_mean", []), 3),
            _vec(pooled.get("force_object_sum", []), 3),
        ]
    ).astype(np.float32)


def _topk_contact_value(contact, topk=4):
    contacts = list(contact.get("contacts", []))
    contacts.sort(key=lambda item: -abs(float(item.get("penetration", 0.0))))
    output = []
    for index in range(topk):
        if index >= len(contacts):
            output.extend([0.0] * 12)
            continue
        item = contacts[index]
        object_side = item.get("object_side")
        arm_link = item.get("link_b", -1) if object_side == "a" else item.get("link_a", -1)
        output.extend(
            [
                1.0,
                float(arm_link) / 16.0,
                float(item.get("penetration", 0.0)),
                *_vec(item.get("position_object", []), 3).tolist(),
                *_vec(item.get("normal_object", []), 3).tolist(),
                *_vec(item.get("force_object", []), 3).tolist(),
            ]
        )
    return np.asarray(output, dtype=np.float32)


def _transition_contact(row, encoder, steps=2):
    trace = _json(row.get("nominal_contact_geometry_trace", "[]"), [])
    empty = {"count": 0, "contacts": [], "pooled": {}}
    return np.concatenate(
        [encoder(trace[index] if index < len(trace) else empty) for index in range(steps)]
    ).astype(np.float32)


def _load_records(path):
    records = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not _as_bool(row["usable"]):
                continue
            state = _state_context(row)
            anchor_contact = _json(row["anchor_contact"], {})
            anchor_pooled = np.concatenate(
                [state, _pooled_contact_value(anchor_contact)]
            ).astype(np.float32)
            anchor_topk = np.concatenate(
                [anchor_pooled, _topk_contact_value(anchor_contact)]
            ).astype(np.float32)
            transition_pooled = np.concatenate(
                [state, _transition_contact(row, _pooled_contact_value)]
            ).astype(np.float32)
            transition_topk = np.concatenate(
                [transition_pooled, _transition_contact(row, _topk_contact_value)]
            ).astype(np.float32)
            obj_pos = _json(row["obj_pos"], [0.0, 0.0, 0.0])
            records.append(
                Record(
                    anchor_id=int(row["anchor_id"]),
                    split=row["split"],
                    matrix=np.asarray(json.loads(row["target_matrix"]), dtype=np.float32),
                    state_context=state,
                    anchor_pooled_context=anchor_pooled,
                    anchor_topk_context=anchor_topk,
                    transition_pooled_context=transition_pooled,
                    transition_topk_context=transition_topk,
                    obj_x=float(obj_pos[0]),
                    obj_y=float(obj_pos[1]),
                    speed=float(row["speed"]),
                    anchor_step=int(row["anchor_step"]),
                )
            )
    return records


def _context(record, mode):
    if mode == "state":
        return record.state_context
    if mode == "anchor_pooled":
        return record.anchor_pooled_context
    if mode == "anchor_topk":
        return record.anchor_topk_context
    if mode == "transition_pooled":
        return record.transition_pooled_context
    if mode == "transition_topk":
        return record.transition_topk_context
    raise ValueError(mode)


class MatrixModel(torch.nn.Module):
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


def _objective(pred, target, cosine_weight, y_cosine_weight):
    huber = torch.nn.functional.smooth_l1_loss(pred, target, beta=0.1)
    row_cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1, eps=1e-8)
    row_norm = torch.linalg.vector_norm(target, dim=-1)
    active = row_norm > 1e-7
    all_cos = (1.0 - row_cos[active]).mean() if active.any() else pred.sum() * 0.0
    y_active = active[:, 1]
    y_cos = (
        (1.0 - row_cos[:, 1][y_active]).mean()
        if y_active.any()
        else pred.sum() * 0.0
    )
    return huber + cosine_weight * all_cos + y_cosine_weight * y_cos


def _arrays(records, mode):
    return (
        np.stack([_context(record, mode) for record in records]),
        np.stack([record.matrix for record in records]),
    )


def _train(records_by_split, mode, args, run_seed):
    train_x, train_y = _arrays(records_by_split["train"], mode)
    val_x, val_y = _arrays(records_by_split["val"], mode)
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    row_rms = np.sqrt(np.mean(train_y * train_y, axis=2))
    row_scale = np.maximum(np.quantile(row_rms, 0.75, axis=0), 1e-6).astype(np.float32)
    train_x = (train_x - x_mean) / x_std
    val_x = (val_x - x_mean) / x_std
    train_y = train_y / row_scale.reshape(1, TARGET_DIM, 1)
    val_y = val_y / row_scale.reshape(1, TARGET_DIM, 1)

    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MatrixModel(train_x.shape[1], args.hidden_dim).to(device)
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
        loss = _objective(model(tx), ty, args.cosine_weight, args.y_cosine_weight)
        loss.backward()
        optimizer.step()
        if epoch % args.eval_every and epoch != args.epochs:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = float(
                _objective(model(vx), vy, args.cosine_weight, args.y_cosine_weight).item()
            )
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
        raise RuntimeError("no checkpoint")
    model.load_state_dict(best_state)
    model.cpu().eval()
    return {
        "model": model,
        "mode": mode,
        "seed": run_seed,
        "best": best,
        "history": history,
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "row_scale": row_scale,
        "context_dim": train_x.shape[1],
        "device": str(device),
    }


def _predict(run, records):
    if not records:
        return np.zeros((0, TARGET_DIM, ACTION_DIM), dtype=np.float32)
    x = np.stack([_context(record, run["mode"]) for record in records])
    x = (x - run["x_mean"]) / run["x_std"]
    with torch.no_grad():
        pred = run["model"](torch.tensor(x, dtype=torch.float32)).numpy()
    return pred * run["row_scale"].reshape(1, TARGET_DIM, 1)


def _nearest(train, test, mode):
    train_x = np.stack([_context(record, mode) for record in train])
    test_x = np.stack([_context(record, mode) for record in test])
    mean = train_x.mean(axis=0, keepdims=True)
    std = np.where(train_x.std(axis=0, keepdims=True) < 1e-6, 1.0, train_x.std(axis=0, keepdims=True))
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    output = []
    for value in test_x:
        index = int(np.argmin(np.sum((train_x - value) ** 2, axis=1)))
        output.append(train[index].matrix)
    return np.stack(output)


def _metrics(records, pred, nonzero_threshold):
    truth = np.stack([record.matrix for record in records])
    matrix_rel = _rms(pred - truth) / (_rms(truth) + 1e-12)
    y_cosines = []
    all_cosines = []
    sign_scores = []
    nonzero = 0
    per_anchor = []
    for record, p, t in zip(records, pred, truth):
        y_cos = _cosine(p[1], t[1])
        if y_cos is not None:
            y_cosines.append(y_cos)
        local = []
        for index in range(TARGET_DIM):
            value = _cosine(p[index], t[index])
            if value is not None:
                local.append(value)
                all_cosines.append(value)
        threshold = max(float(np.linalg.norm(t[1])) * 1e-5, 1e-12)
        mask = np.abs(t[1]) > threshold
        sign = None if not mask.any() else float(np.mean(np.sign(p[1][mask]) == np.sign(t[1][mask])))
        if sign is not None:
            sign_scores.append(sign)
        nonzero += int(float(np.linalg.norm(p[1])) > nonzero_threshold)
        per_anchor.append(
            {
                "anchor_id": record.anchor_id,
                "split": record.split,
                "obj_y": record.obj_y,
                "speed": record.speed,
                "anchor_step": record.anchor_step,
                "y_vjp_cosine": y_cos,
                "row_cosine_mean": None if not local else float(np.mean(local)),
                "y_sign_agreement": sign,
                "y_truth_norm": float(np.linalg.norm(t[1])),
                "y_pred_norm": float(np.linalg.norm(p[1])),
                "truth_matrix": json.dumps(t.tolist()),
                "pred_matrix": json.dumps(p.tolist()),
            }
        )
    return {
        "num_anchors": len(records),
        "matrix_relative_rmse": matrix_rel,
        "vjp_cosine_mean": None if not all_cosines else float(np.mean(all_cosines)),
        "vjp_cosine_median": None if not all_cosines else float(np.median(all_cosines)),
        "y_vjp_cosine_mean": None if not y_cosines else float(np.mean(y_cosines)),
        "y_vjp_cosine_median": None if not y_cosines else float(np.median(y_cosines)),
        "y_sign_agreement": None if not sign_scores else float(np.mean(sign_scores)),
        "nonzero_predicted_y_rate": nonzero / max(len(records), 1),
    }, per_anchor


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--modes",
        default="state,anchor_pooled,anchor_topk,transition_pooled,transition_topk",
    )
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-improvement", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--y-cosine-weight", type=float, default=0.5)
    parser.add_argument("--nonzero-threshold", type=float, default=1e-10)
    parser.add_argument("--min-train", type=int, default=40)
    parser.add_argument("--min-val", type=int, default=10)
    parser.add_argument("--min-heldout", type=int, default=50)
    parser.add_argument("--allow-small-pilot", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    records = _load_records(args.matrices)
    records_by_split = defaultdict(list)
    for record in records:
        records_by_split[record.split].append(record)
    heldout_count = len(records_by_split["test_id"]) + len(records_by_split["test_ood"])
    if not args.allow_small_pilot:
        if len(records_by_split["train"]) < args.min_train:
            raise RuntimeError(f"too few train anchors: {len(records_by_split['train'])}")
        if len(records_by_split["val"]) < args.min_val:
            raise RuntimeError(f"too few val anchors: {len(records_by_split['val'])}")
        if heldout_count < args.min_heldout:
            raise RuntimeError(f"too few frozen heldout anchors: {heldout_count}")

    modes = [mode for mode in args.modes.split(",") if mode]
    runs = []
    for mode in modes:
        for index in range(args.num_seeds):
            run = _train(records_by_split, mode, args, args.seed + 1009 * index + 10007 * modes.index(mode))
            val_pred = _predict(run, records_by_split["val"])
            val_metrics, _ = _metrics(records_by_split["val"], val_pred, args.nonzero_threshold)
            run["val_metrics"] = val_metrics
            runs.append(run)
    selected_index = min(range(len(runs)), key=lambda index: runs[index]["best"]["val_loss"])
    selected = runs[selected_index]

    train_truth = np.stack([record.matrix for record in records_by_split["train"]])
    global_matrix = train_truth.mean(axis=0)
    summary_metrics = {}
    eval_rows = []
    for split_name in ("val", "test_id", "test_ood"):
        split_records = records_by_split[split_name]
        if not split_records:
            summary_metrics[split_name] = {}
            continue
        predictions = {
            "zero": np.zeros((len(split_records), TARGET_DIM, ACTION_DIM), dtype=np.float32),
            "global": np.repeat(global_matrix[None], len(split_records), axis=0),
            "nearest_state": _nearest(records_by_split["train"], split_records, "state"),
            "nearest_transition_topk": _nearest(
                records_by_split["train"], split_records, "transition_topk"
            ),
            "selected_mlp": _predict(selected, split_records),
        }
        for run_index, run in enumerate(runs):
            predictions[f"mlp_{run['mode']}_seed{run_index}"] = _predict(run, split_records)
        summary_metrics[split_name] = {}
        for name, pred in predictions.items():
            metrics, rows = _metrics(split_records, pred, args.nonzero_threshold)
            summary_metrics[split_name][name] = metrics
            for row in rows:
                row["model"] = name
            eval_rows.extend(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "a5_action_vjp_v2_model.pt"
    torch.save(
        {
            "model_state_dict": selected["model"].state_dict(),
            "mode": selected["mode"],
            "context_dim": selected["context_dim"],
            "hidden_dim": args.hidden_dim,
            "x_mean": torch.tensor(selected["x_mean"]),
            "x_std": torch.tensor(selected["x_std"]),
            "row_scale": torch.tensor(selected["row_scale"]),
            "split_anchor_ids": {
                key: [record.anchor_id for record in value]
                for key, value in records_by_split.items()
            },
            "args": vars(args),
        },
        checkpoint_path,
    )
    serial_runs = []
    for run in runs:
        serial_runs.append(
            {
                "mode": run["mode"],
                "seed": run["seed"],
                "best": run["best"],
                "history": run["history"],
                "context_dim": run["context_dim"],
                "row_scale": run["row_scale"].tolist(),
                "device": run["device"],
                "val_metrics": run["val_metrics"],
            }
        )
    payload = {
        "description": "A5 action-side VJP v2 trusted-matrix trainer",
        "matrices": str(args.matrices),
        "num_records": len(records),
        "split_counts": {key: len(value) for key, value in records_by_split.items()},
        "selected_run": selected_index,
        "selected_mode": selected["mode"],
        "training_runs": serial_runs,
        "metrics": summary_metrics,
        "checkpoint": str(checkpoint_path),
    }
    summary_path = args.out_dir / "a5_action_vjp_v2_train_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    _write_csv(args.out_dir / "a5_action_vjp_v2_train_eval.csv", eval_rows)
    print(f"[a5-vjp-v2-train] records={len(records)} split={payload['split_counts']}")
    print(
        f"[a5-vjp-v2-train] selected={selected_index} mode={selected['mode']} "
        f"val_loss={selected['best']['val_loss']:.4f}"
    )
    for split_name in ("test_id", "test_ood"):
        if split_name not in summary_metrics or not summary_metrics[split_name]:
            continue
        print(f"[a5-vjp-v2-train] {split_name}")
        for name in (
            "zero",
            "global",
            "nearest_state",
            "nearest_transition_topk",
            "selected_mlp",
        ):
            metrics = summary_metrics[split_name][name]
            print(
                f"  {name:13s} y_cos={metrics['y_vjp_cosine_median']} "
                f"sign={metrics['y_sign_agreement']} nonzero={metrics['nonzero_predicted_y_rate']:.3f}"
            )
    print(f"[a5-vjp-v2-train] wrote {summary_path}")
    print(f"[a5-vjp-v2-train] wrote {checkpoint_path}")


if __name__ == "__main__":
    raise SystemExit(main())
