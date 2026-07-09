"""Train/evaluate a tiny matrix-head MLP on A5 FD response labels.

The model follows the intended action-side VJP structure in forward form:

    d_hat = A_theta(z_anchor) v

where z_anchor is a small context vector and v is an action perturbation
direction. This script is a potential check, not the final VJP trainer.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


DEFAULT_DATA = Path("analysis/2026-07-09_arx_pusher/a5_multi_anchor_fd_dataset.csv")
DEFAULT_OUT = Path("analysis/2026-07-09_arx_pusher/a5_response_mlp_eval.json")


def _loads_vec(text, default=None, length=None):
    if text is None or text == "":
        if default is None:
            raise ValueError("missing vector value")
        values = list(default)
    else:
        values = list(json.loads(text))
    if length is not None:
        if len(values) < length:
            values = values + [0.0] * (length - len(values))
        values = values[:length]
    return np.asarray(values, dtype=np.float32)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "none", "")


def _float(row, name, default=0.0):
    value = row.get(name, "")
    if value == "":
        return float(default)
    return float(value)


def _context_from_row(row, args):
    if args.context_mode == "basic":
        values = [
            _float(row, "obj_y"),
            _float(row, "speed"),
            _float(row, "anchor_step"),
            _float(row, "horizontal_disp"),
            _float(row, "vertical_disp"),
            _float(row, "epsilon"),
        ]
        return np.asarray(values, dtype=np.float32)

    values = [
        _float(row, "obj_x"),
        _float(row, "obj_y"),
        _float(row, "obj_z"),
        _float(row, "speed"),
        _float(row, "anchor_step"),
        _float(row, "response_steps"),
        _float(row, "horizontal_disp"),
        _float(row, "vertical_disp"),
        _float(row, "epsilon"),
        _float(row, "nominal_contact_mean"),
        _float(row, "nominal_contact_max"),
        _float(row, "nominal_contact_min"),
    ]
    robot_dof = args.robot_dof
    object_qvel_dim = args.object_qvel_dim
    for key, length in (
        ("qpos", robot_dof),
        ("qvel", robot_dof),
        ("nominal_anchor_pos", 3),
        ("nominal_pre_pos", 3),
        ("nominal_post_pos", 3),
        ("nominal_initial_pos", 3),
        ("nominal_final_pos", 3),
        ("nominal_anchor_qvel", object_qvel_dim),
        ("nominal_pre_qvel", object_qvel_dim),
        ("nominal_post_qvel", object_qvel_dim),
        ("nominal_initial_qvel", object_qvel_dim),
        ("nominal_final_qvel", object_qvel_dim),
        ("nominal_pre_disp", 3),
        ("nominal_post_disp", 3),
        ("nominal_total_disp", 3),
        ("nominal_pre_qvel_delta", object_qvel_dim),
        ("nominal_post_qvel_delta", object_qvel_dim),
        ("nominal_total_qvel_delta", object_qvel_dim),
        ("nominal_contact_trace", args.contact_trace_len),
    ):
        values.extend(_loads_vec(row.get(key, ""), default=[0.0] * length, length=length).tolist())

    if args.context_mode == "oracle":
        values.extend(
            [
                _float(row, "contact_mean_delta"),
                _float(row, "contact_max_delta"),
                _float(row, "plus_contact_mean"),
                _float(row, "minus_contact_mean"),
                _float(row, "plus_contact_max"),
                _float(row, "minus_contact_max"),
            ]
        )
        for key in ("plus_contact_trace", "minus_contact_trace"):
            values.extend(
                _loads_vec(row.get(key, ""), default=[0.0] * args.contact_trace_len, length=args.contact_trace_len).tolist()
            )
    return np.asarray(values, dtype=np.float32)


def _load_rows(path, target_name, args):
    rows = []
    skipped_keep = 0
    skipped_target_norm = 0
    with path.open() as f:
        for row in csv.DictReader(f):
            if args.require_keep and not _as_bool(row.get("keep", True)):
                skipped_keep += 1
                continue
            target = _loads_vec(row[target_name])
            target_norm = float(np.linalg.norm(target))
            if target_norm < args.min_target_norm or target_norm > args.max_target_norm:
                skipped_target_norm += 1
                continue
            rows.append(
                {
                    "anchor_id": int(row["anchor_id"]),
                    "direction": row["direction"],
                    "context": _context_from_row(row, args),
                    "v": _loads_vec(row["direction_vec"]),
                    "target": target,
                }
            )
    return rows, skipped_keep, skipped_target_norm


def _stack(rows):
    return (
        np.stack([r["context"] for r in rows], axis=0),
        np.stack([r["v"] for r in rows], axis=0),
        np.stack([r["target"] for r in rows], axis=0),
    )


def _standardize(train_x, x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean, std


def _metrics(pred, target):
    err = pred - target
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    target_rms = float(np.sqrt(np.mean(target ** 2)))
    rel_rmse = rmse / (target_rms + 1e-12)
    return {
        "rmse": rmse,
        "mae": mae,
        "target_rms": target_rms,
        "relative_rmse": rel_rmse,
    }


def _fit_global_matrix(train_rows, test_rows):
    train_x = np.stack([r["v"] for r in train_rows], axis=0)
    train_y = np.stack([r["target"] for r in train_rows], axis=0)
    coef, _, _, _ = np.linalg.lstsq(train_x, train_y, rcond=None)
    test_x = np.stack([r["v"] for r in test_rows], axis=0)
    test_y = np.stack([r["target"] for r in test_rows], axis=0)
    pred = test_x @ coef
    return _metrics(pred, test_y)


class MatrixHead(torch.nn.Module):
    def __init__(self, context_dim, action_dim, target_dim, hidden_dim):
        super().__init__()
        self.action_dim = action_dim
        self.target_dim = target_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(context_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, target_dim * action_dim),
        )

    def forward(self, context, direction):
        mat = self.net(context).reshape(-1, self.target_dim, self.action_dim)
        return torch.einsum("bij,bj->bi", mat, direction)


def _train_mlp(train_rows, test_rows, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_context, train_v, train_y = _stack(train_rows)
    test_context, test_v, test_y = _stack(test_rows)

    train_context_z, mean, std = _standardize(train_context, train_context)
    test_context_z = (test_context - mean) / std
    if args.standardize_target:
        y_mean = train_y.mean(axis=0, keepdims=True)
        y_std = train_y.std(axis=0, keepdims=True)
        y_std = np.where(y_std < 1e-6, 1.0, y_std)
        fit_train_y = (train_y - y_mean) / y_std
    elif args.scale_target:
        y_mean = np.zeros((1, train_y.shape[1]), dtype=np.float32)
        y_std = train_y.std(axis=0, keepdims=True)
        y_std = np.where(y_std < 1e-6, 1.0, y_std)
        fit_train_y = train_y / y_std
    else:
        y_mean = np.zeros((1, train_y.shape[1]), dtype=np.float32)
        y_std = np.ones((1, train_y.shape[1]), dtype=np.float32)
        fit_train_y = train_y

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MatrixHead(train_context.shape[1], train_v.shape[1], train_y.shape[1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.SmoothL1Loss(beta=args.huber_beta)

    x = torch.tensor(train_context_z, dtype=torch.float32, device=device)
    v = torch.tensor(train_v, dtype=torch.float32, device=device)
    y = torch.tensor(fit_train_y, dtype=torch.float32, device=device)
    tx = torch.tensor(test_context_z, dtype=torch.float32, device=device)
    tv = torch.tensor(test_v, dtype=torch.float32, device=device)

    best = None
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x, v)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                test_pred = model(tx, tv).detach().cpu().numpy()
            test_pred = test_pred * y_std + y_mean
            metric = _metrics(test_pred, test_y)
            if best is None or metric["relative_rmse"] < best["relative_rmse"]:
                best = metric
                best_state = {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_pred = model(x, v).detach().cpu().numpy()
        test_pred = model(tx, tv).detach().cpu().numpy()
    train_pred = train_pred * y_std + y_mean
    test_pred = test_pred * y_std + y_mean
    return {
        "device": str(device),
        "train": _metrics(train_pred, train_y),
        "test": _metrics(test_pred, test_y),
    }


def _leave_one_anchor(rows, args):
    anchor_ids = sorted({r["anchor_id"] for r in rows})
    folds = []
    for fold_idx, anchor_id in enumerate(anchor_ids):
        train_rows = [r for r in rows if r["anchor_id"] != anchor_id]
        test_rows = [r for r in rows if r["anchor_id"] == anchor_id]
        baseline = _fit_global_matrix(train_rows, test_rows)
        mlp = _train_mlp(train_rows, test_rows, args, seed=args.seed + fold_idx)
        folds.append(
            {
                "heldout_anchor_id": anchor_id,
                "num_train": len(train_rows),
                "num_test": len(test_rows),
                "global_matrix_baseline": baseline,
                "matrix_head_mlp": mlp,
                "relative_rmse_improvement": baseline["relative_rmse"] / (mlp["test"]["relative_rmse"] + 1e-12),
            }
        )
    return folds


def _row_split(rows, args):
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(rows) * args.test_frac)))
    test_ids = set(idx[:n_test].tolist())
    train_rows = [r for i, r in enumerate(rows) if i not in test_ids]
    test_rows = [r for i, r in enumerate(rows) if i in test_ids]
    baseline = _fit_global_matrix(train_rows, test_rows)
    mlp = _train_mlp(train_rows, test_rows, args, seed=args.seed + 999)
    return {
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "global_matrix_baseline": baseline,
        "matrix_head_mlp": mlp,
        "relative_rmse_improvement": baseline["relative_rmse"] / (mlp["test"]["relative_rmse"] + 1e-12),
    }


def _mean(values):
    return float(sum(values) / len(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--target",
        choices=(
            "local_response",
            "final_response",
            "local_vel_response",
            "final_vel_response",
            "local_state_response",
            "final_state_response",
        ),
        default="local_response",
    )
    parser.add_argument("--context-mode", choices=("basic", "rich", "oracle"), default="rich")
    parser.add_argument("--robot-dof", type=int, default=6)
    parser.add_argument("--object-qvel-dim", type=int, default=6)
    parser.add_argument("--contact-trace-len", type=int, default=11)
    parser.add_argument("--require-keep", action="store_true")
    parser.add_argument("--min-target-norm", type=float, default=0.0)
    parser.add_argument("--max-target-norm", type=float, default=float("inf"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=1e-3)
    parser.add_argument("--standardize-target", action="store_true")
    parser.add_argument(
        "--scale-target",
        action="store_true",
        help="Scale target dimensions without centering, preserving A(z) * 0 = 0.",
    )
    parser.add_argument("--eval-mode", choices=("all", "loo", "row-split"), default="all")
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.standardize_target and args.scale_target:
        raise ValueError("--standardize-target and --scale-target are mutually exclusive")

    rows, skipped_keep, skipped_target_norm = _load_rows(args.data, args.target, args)
    if len(rows) < 8:
        raise RuntimeError(f"not enough rows: {len(rows)}")

    loo = [] if args.eval_mode == "row-split" else _leave_one_anchor(rows, args)
    row_split = None if args.eval_mode == "loo" else _row_split(rows, args)
    baseline_rel = [f["global_matrix_baseline"]["relative_rmse"] for f in loo]
    mlp_rel = [f["matrix_head_mlp"]["test"]["relative_rmse"] for f in loo]

    payload = {
        "description": "A5 action-side response matrix-head MLP potential check",
        "data": str(args.data),
        "target": args.target,
        "context_mode": args.context_mode,
        "context_dim": int(rows[0]["context"].shape[0]),
        "require_keep": args.require_keep,
        "min_target_norm": args.min_target_norm,
        "max_target_norm": args.max_target_norm,
        "num_rows": len(rows),
        "num_skipped_keep_rows": skipped_keep,
        "num_skipped_target_norm_rows": skipped_target_norm,
        "num_anchors": len({r["anchor_id"] for r in rows}),
        "model": {
            "structure": "A_theta(z) matrix head, prediction = A_theta(z) v",
            "target_dim": int(rows[0]["target"].shape[0]),
            "action_dim": int(rows[0]["v"].shape[0]),
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "loss": "SmoothL1",
            "huber_beta": args.huber_beta,
            "standardize_target": args.standardize_target,
            "scale_target": args.scale_target,
        },
        "leave_one_anchor_out": loo,
        "leave_one_anchor_summary": {
            "baseline_relative_rmse_mean": _mean(baseline_rel),
            "mlp_relative_rmse_mean": _mean(mlp_rel),
            "mean_improvement_ratio": _mean([b / (m + 1e-12) for b, m in zip(baseline_rel, mlp_rel)]),
        },
        "row_split": row_split,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[a5-mlp] target={args.target} rows={len(rows)} anchors={payload['num_anchors']}")
    print(
        f"[a5-mlp] context={args.context_mode} dim={payload['context_dim']} "
        f"require_keep={args.require_keep} skipped_keep={skipped_keep} "
        f"skipped_target_norm={skipped_target_norm}"
    )
    if loo:
        print(
            "[a5-mlp] LOO rel_rmse baseline="
            f"{payload['leave_one_anchor_summary']['baseline_relative_rmse_mean']:.4f} "
            f"mlp={payload['leave_one_anchor_summary']['mlp_relative_rmse_mean']:.4f} "
            f"improve={payload['leave_one_anchor_summary']['mean_improvement_ratio']:.3f}x"
        )
    if row_split is not None:
        print(
            "[a5-mlp] row split rel_rmse baseline="
            f"{row_split['global_matrix_baseline']['relative_rmse']:.4f} "
            f"mlp={row_split['matrix_head_mlp']['test']['relative_rmse']:.4f} "
            f"improve={row_split['relative_rmse_improvement']:.3f}x"
        )
    print(f"[a5-mlp] wrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
