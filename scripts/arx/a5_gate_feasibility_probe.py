"""Diagnose a deployable stable-like versus both-like A5 gate."""

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "analysis/2026-07-09_arx_pusher/gate_feasibility_probe"
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
DEFAULT_FINAL = (
    REPO_ROOT
    / "analysis/2026-07-09_arx_pusher/action_vjp_v2_replay330/final"
    / "a5_action_vjp_v2_final_matrices.csv"
)
DEFAULT_MARGINAL = (
    REPO_ROOT
    / "analysis/2026-07-09_arx_pusher/marginal_probe"
    / "a5_marginal_probe_frozen_selection.csv"
)
DEFAULT_BRANCH_SELECTION = (
    REPO_ROOT
    / "analysis/2026-07-09_arx_pusher/both_branch_probe"
    / "a5_both_branch_probe_frozen_selection.csv"
)
DEFAULT_BRANCH_SUMMARY = (
    REPO_ROOT
    / "analysis/2026-07-09_arx_pusher/both_branch_probe"
    / "a5_both_branch_probe_anchor_summary.csv"
)
DEFAULT_REPORT = REPO_ROOT / "notes/a5_vjp_progress/2026-07-10_gate_feasibility_probe.md"

FEATURE_VARIANTS = ("X1", "X2", "X3")
MODEL_NAMES = ("logreg", "mlp")
WEAK_REASONS = {"weak_y_vjp", "weak_random_y_signal"}

X1_NAMES = [
    *[f"arm_qpos_{index}" for index in range(1, 7)],
    *[f"arm_qvel_{index}" for index in range(1, 7)],
    "obj_pos_x",
    "obj_pos_y",
    "obj_pos_z",
    "obj_quat_w",
    "obj_quat_x",
    "obj_quat_y",
    "obj_quat_z",
    "obj_linvel_x",
    "obj_linvel_y",
    "obj_linvel_z",
    "obj_angvel_x",
    "obj_angvel_y",
    "obj_angvel_z",
]
X2_EXTRA_NAMES = [
    "contact_point_obj_x",
    "contact_point_obj_y",
    "contact_point_obj_z",
    "contact_normal_obj_x",
    "contact_normal_obj_y",
    "contact_normal_obj_z",
    "contact_penetration_max",
]
X3_EXTRA_NAMES = [f"nominal_action_{index}" for index in range(1, 7)]
FEATURE_NAMES = {
    "X1": X1_NAMES,
    "X2": X1_NAMES + X2_EXTRA_NAMES,
    "X3": X1_NAMES + X2_EXTRA_NAMES + X3_EXTRA_NAMES,
}


def _as_bool(value):
    return str(value).strip().lower() not in ("", "0", "false", "no", "none")


def _json(value, default):
    return default if value in (None, "") else json.loads(value)


def _read_csv(path):
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return value


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
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*values):
    text = "|".join(str(value) for value in values)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _percentile(values, q):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else float(np.percentile(finite, q))


def _distribution(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "num": len(finite),
        "mean": None if not finite else float(np.mean(finite)),
        "q25": _percentile(finite, 25),
        "median": _percentile(finite, 50),
        "q75": _percentile(finite, 75),
        "min": None if not finite else min(finite),
        "max": None if not finite else max(finite),
    }


def _paths(out_dir):
    return {
        "dataset": out_dir / "a5_gate_feasibility_frozen_dataset.csv",
        "data_summary": out_dir / "a5_gate_feasibility_data_summary.json",
        "results": out_dir / "a5_gate_feasibility_results.csv",
        "predictions": out_dir / "a5_gate_feasibility_predictions.csv",
        "coefficients": out_dir / "a5_gate_feasibility_logreg_coefficients.csv",
        "training": out_dir / "a5_gate_feasibility_training.csv",
        "summary": out_dir / "a5_gate_feasibility_summary.json",
    }


def _source_label(row):
    reasons = set(row["gate_reasons"].split("|"))
    if _as_bool(row["usable"]):
        return 0, "stable"
    if reasons & WEAK_REASONS:
        return None, None
    cross = "cross_epsilon_y_cosine" in reasons
    signature = "contact_signature_switch" in reasons
    if signature and not cross:
        return 0, "signature_only"
    if signature and cross:
        return 1, "both"
    return None, None


def _canonical_quaternion(values):
    quaternion = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm > 1e-12:
        quaternion /= norm
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion.tolist()


def _contact_features(contact):
    pooled = contact.get("pooled", {})
    point = np.asarray(pooled.get("position_object_mean", [0.0, 0.0, 0.0]), dtype=np.float64)
    normal = np.asarray(pooled.get("normal_object_mean", [0.0, 0.0, 0.0]), dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm > 1e-12:
        normal /= normal_norm
    penetration = float(pooled.get("penetration_max", 0.0))
    return [*point.tolist(), *normal.tolist(), penetration]


def _trajectory_key(source_row, dense_row):
    payload = {
        "obj_pos": [round(float(value), 9) for value in _json(source_row["obj_pos"], [])],
        "command_qpos": [round(float(value), 9) for value in _json(dense_row["qpos"], [])],
        "command_qvel": [round(float(value), 9) for value in _json(dense_row["qvel"], [])],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _feature_record(source_row, dense_row):
    arm = _json(source_row["anchor_arm_state"], {})
    obj = _json(source_row["anchor_object_state"], {})
    contact = _json(source_row["anchor_contact"], {})
    obj_qvel = [float(value) for value in obj.get("qvel", [0.0] * 6)]
    x1 = [
        *[float(value) for value in arm["qpos"][:6]],
        *[float(value) for value in arm["qvel"][:6]],
        *[float(value) for value in obj["pos"][:3]],
        *_canonical_quaternion(obj["quat"][:4]),
        *obj_qvel[:3],
        *obj_qvel[3:6],
    ]
    x2 = x1 + _contact_features(contact)
    action = [float(value) for value in _json(dense_row["qvel"], [])[:6]]
    x3 = x2 + action
    if len(x1) != len(FEATURE_NAMES["X1"]):
        raise ValueError(f"X1 dimension mismatch: {len(x1)}")
    if len(x2) != len(FEATURE_NAMES["X2"]):
        raise ValueError(f"X2 dimension mismatch: {len(x2)}")
    if len(x3) != len(FEATURE_NAMES["X3"]):
        raise ValueError(f"X3 dimension mismatch: {len(x3)}")
    if not np.all(np.isfinite(np.asarray(x3, dtype=np.float64))):
        raise ValueError("non-finite gate feature")
    return {
        "X1": x1,
        "X2": x2,
        "X3": x3,
        "contact_count": int(contact.get("count", 0)),
    }


def _split_groups(rows, fractions, seed, attempts=256):
    split_names = tuple(fractions)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["trajectory_key"]].append(row)
    class_totals = Counter(int(row["label"]) for row in rows)
    targets = {
        split: {label: fractions[split] * class_totals[label] for label in (0, 1)}
        for split in split_names
    }
    group_items = []
    for key, members in grouped.items():
        counts = Counter(int(row["label"]) for row in members)
        group_items.append((key, members, counts))

    best = None
    for attempt in range(attempts):
        order = sorted(
            group_items,
            key=lambda item: (
                -len(item[1]),
                _stable_hash(seed, attempt, item[0]),
            ),
        )
        counts = {split: Counter() for split in split_names}
        group_counts = Counter()
        assignment = {}
        for key, members, group_class_counts in order:
            choices = []
            for split in split_names:
                trial_counts = {name: Counter(value) for name, value in counts.items()}
                trial_counts[split].update(group_class_counts)
                trial_group_counts = Counter(group_counts)
                trial_group_counts[split] += 1
                cost = 0.0
                for name in split_names:
                    for label in (0, 1):
                        scale = max(targets[name][label], 1.0)
                        cost += ((trial_counts[name][label] - targets[name][label]) / scale) ** 2
                    group_target = fractions[name] * len(group_items)
                    cost += 0.1 * (
                        (trial_group_counts[name] - group_target) / max(group_target, 1.0)
                    ) ** 2
                choices.append((cost, _stable_hash(seed, attempt, key, split), split))
            _, _, selected = min(choices)
            assignment[key] = selected
            counts[selected].update(group_class_counts)
            group_counts[selected] += 1
        final_cost = sum(
            ((counts[split][label] - targets[split][label]) / max(targets[split][label], 1.0)) ** 2
            for split in split_names
            for label in (0, 1)
        )
        candidate = (final_cost, attempt, assignment, counts, group_counts)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    _, attempt, assignment, counts, group_counts = best
    return assignment, counts, group_counts, attempt


def _build_data(args):
    paths = _paths(args.out_dir)
    if paths["dataset"].exists() and paths["data_summary"].exists() and not args.force_data:
        print(f"[gate-data] reused {paths['dataset']}")
        return _read_csv(paths["dataset"]), json.loads(paths["data_summary"].read_text())

    prefilter = _read_csv(args.prefilter)
    dense = {int(row["anchor_id"]): row for row in _read_csv(args.dense_manifest)}
    final = _read_csv(args.final_matrices)
    marginal = _read_csv(args.marginal_selection)
    branch_selection = _read_csv(args.branch_selection)
    branch_summary = _read_csv(args.branch_summary)
    source = {int(row["anchor_id"]): row for row in prefilter}

    labels = {}
    subtypes = {}
    features = {}
    trajectory_keys = {}
    for anchor_id, row in source.items():
        label, subtype = _source_label(row)
        if label is None:
            continue
        if anchor_id not in dense:
            raise RuntimeError(f"anchor {anchor_id} missing dense manifest row")
        labels[anchor_id] = label
        subtypes[anchor_id] = subtype
        features[anchor_id] = _feature_record(row, dense[anchor_id])
        trajectory_keys[anchor_id] = _trajectory_key(row, dense[anchor_id])

    final_frozen = {
        int(row["anchor_id"])
        for row in final
        if row["split"] in ("test_id", "test_ood")
    }
    final_stable = {
        int(row["anchor_id"])
        for row in final
        if row["split"] in ("test_id", "test_ood") and _as_bool(row["usable"])
    }
    marginal_frozen = {int(row["anchor_id"]) for row in marginal}
    marginal_signature = {
        int(row["anchor_id"]) for row in marginal if row["subtype"] == "signature_only"
    }
    marginal_both = {
        int(row["anchor_id"]) for row in marginal if row["subtype"] == "both"
    }
    branch_frozen = {int(row["anchor_id"]) for row in branch_selection}
    branch_confirmed = {
        int(row["anchor_id"])
        for row in branch_summary
        if _as_bool(row["both_confirmed_replay"])
    }
    frozen_ids = final_frozen | marginal_frozen | branch_frozen
    audit_provenance = {}
    for anchor_id in final_stable:
        audit_provenance[anchor_id] = (0, "v2_heldout_stable")
    for anchor_id in marginal_signature:
        audit_provenance[anchor_id] = (0, "marginal_signature_only")
    for anchor_id in marginal_both:
        audit_provenance[anchor_id] = (1, "marginal_both")
    for anchor_id in branch_confirmed:
        audit_provenance[anchor_id] = (1, "branch_replay_confirmed_both")

    missing_audit = sorted(set(audit_provenance) - set(source))
    mismatched_audit = sorted(
        (anchor_id, expected, labels.get(anchor_id))
        for anchor_id, (expected, _) in audit_provenance.items()
        if labels.get(anchor_id) != expected
    )
    if missing_audit or mismatched_audit:
        raise RuntimeError(
            f"audited labels unavailable/mismatched: missing={missing_audit} mismatch={mismatched_audit}"
        )

    frozen_groups = {
        _trajectory_key(source[anchor_id], dense[anchor_id])
        for anchor_id in frozen_ids
        if anchor_id in source and anchor_id in dense
    }
    source_rows = []
    for anchor_id, label in labels.items():
        if anchor_id in frozen_ids or trajectory_keys[anchor_id] in frozen_groups:
            continue
        source_rows.append(
            {
                "anchor_id": anchor_id,
                "label": label,
                "label_name": "stable_like" if label == 0 else "both_like",
                "source_subtype": subtypes[anchor_id],
                "evaluation_role": "source",
                "audit_provenance": "",
                "trajectory_key": trajectory_keys[anchor_id],
                "obj_pos": source[anchor_id]["obj_pos"],
                "speed": float(source[anchor_id]["speed"]),
                "anchor_step": int(source[anchor_id]["anchor_step"]),
                "contact_count": features[anchor_id]["contact_count"],
                "X1": features[anchor_id]["X1"],
                "X2": features[anchor_id]["X2"],
                "X3": features[anchor_id]["X3"],
            }
        )
    fractions = {"train": 0.60, "val": 0.20, "source_test": 0.20}
    assignment, split_counts, group_counts, split_attempt = _split_groups(
        source_rows, fractions, args.split_seed
    )
    for row in source_rows:
        row["split"] = assignment[row["trajectory_key"]]

    audit_rows = []
    for anchor_id, (label, provenance) in sorted(audit_provenance.items()):
        audit_rows.append(
            {
                "anchor_id": anchor_id,
                "label": label,
                "label_name": "stable_like" if label == 0 else "both_like",
                "source_subtype": subtypes[anchor_id],
                "evaluation_role": "replay_audited",
                "audit_provenance": provenance,
                "trajectory_key": trajectory_keys[anchor_id],
                "obj_pos": source[anchor_id]["obj_pos"],
                "speed": float(source[anchor_id]["speed"]),
                "anchor_step": int(source[anchor_id]["anchor_step"]),
                "contact_count": features[anchor_id]["contact_count"],
                "X1": features[anchor_id]["X1"],
                "X2": features[anchor_id]["X2"],
                "X3": features[anchor_id]["X3"],
                "split": "audited_test",
            }
        )

    dataset = source_rows + audit_rows
    dataset.sort(key=lambda row: (row["split"], int(row["anchor_id"])))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(paths["dataset"], dataset)
    source_group_sets = defaultdict(set)
    for row in source_rows:
        source_group_sets[row["split"]].add(row["trajectory_key"])
    split_group_overlap = {
        f"{left}:{right}": len(source_group_sets[left] & source_group_sets[right])
        for index, left in enumerate(fractions)
        for right in list(fractions)[index + 1:]
    }
    training_groups = source_group_sets["train"] | source_group_sets["val"]
    audited_groups = {row["trajectory_key"] for row in audit_rows}
    summary = {
        "description": "A5 stable-like versus both-like gate frozen dataset",
        "source_hashes": {
            "prefilter": _sha256(args.prefilter),
            "dense_manifest": _sha256(args.dense_manifest),
            "final_matrices": _sha256(args.final_matrices),
            "marginal_selection": _sha256(args.marginal_selection),
            "branch_selection": _sha256(args.branch_selection),
            "branch_summary": _sha256(args.branch_summary),
        },
        "feature_source": str(args.prefilter),
        "feature_timing": "pre-action anchor state and anchor_contact only",
        "feature_names": FEATURE_NAMES,
        "source_label_counts": {
            str(label): count for label, count in Counter(labels.values()).items()
        },
        "frozen_ids": len(frozen_ids),
        "frozen_trajectory_groups": len(frozen_groups),
        "source_eligible_after_freeze": {
            str(label): count
            for label, count in Counter(int(row["label"]) for row in source_rows).items()
        },
        "source_split_counts": {
            split: {str(label): int(split_counts[split][label]) for label in (0, 1)}
            for split in fractions
        },
        "source_split_group_counts": dict(group_counts),
        "split_search_attempt": split_attempt,
        "split_group_overlap": split_group_overlap,
        "audited_counts": {
            str(label): count
            for label, count in Counter(int(row["label"]) for row in audit_rows).items()
        },
        "audited_provenance": dict(Counter(row["audit_provenance"] for row in audit_rows)),
        "train_val_to_audited_group_overlap": len(training_groups & audited_groups),
        "dataset_rows": len(dataset),
        "dataset_sha256": _sha256(paths["dataset"]),
    }
    paths["data_summary"].write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[gate-data] source={summary['source_eligible_after_freeze']} "
        f"splits={summary['source_split_counts']} audited={summary['audited_counts']}"
    )
    print(f"[gate-data] wrote {paths['dataset']}")
    return _read_csv(paths["dataset"]), summary


def _load_xy(rows, split, variant):
    selected = [row for row in rows if row["split"] == split]
    x = np.asarray([_json(row[variant], []) for row in selected], dtype=np.float64)
    y = np.asarray([int(row["label"]) for row in selected], dtype=np.int64)
    groups = [row["trajectory_key"] for row in selected]
    anchor_ids = [int(row["anchor_id"]) for row in selected]
    return selected, x, y, groups, anchor_ids


def _standardize(train_x, *others):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return mean, scale, [(value - mean) / scale for value in (train_x, *others)]


def _weighted_binary_loss(logits, labels, weights):
    return float(np.mean(weights * (np.logaddexp(0.0, logits) - labels * logits)))


def _fit_logreg(train_x, train_y, val_x, val_y, l2):
    dimension = train_x.shape[1]
    count0 = max(int(np.sum(train_y == 0)), 1)
    count1 = max(int(np.sum(train_y == 1)), 1)
    class_weights = np.asarray(
        [len(train_y) / (2.0 * count0), len(train_y) / (2.0 * count1)],
        dtype=np.float64,
    )
    sample_weights = class_weights[train_y]

    def objective(parameters):
        coefficient = parameters[:dimension]
        intercept = parameters[-1]
        logits = train_x @ coefficient + intercept
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        loss = _weighted_binary_loss(logits, train_y, sample_weights)
        loss += 0.5 * l2 * float(np.dot(coefficient, coefficient))
        residual = sample_weights * (probabilities - train_y)
        gradient = np.concatenate(
            [train_x.T @ residual / len(train_y) + l2 * coefficient, [float(np.mean(residual))]]
        )
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(dimension + 1, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    coefficient = result.x[:dimension]
    intercept = float(result.x[-1])
    val_logits = val_x @ coefficient + intercept
    val_weights = class_weights[val_y]
    return {
        "coefficient": coefficient,
        "intercept": intercept,
        "status": "ok" if result.success else f"optimizer:{result.message}",
        "iterations": int(result.nit),
        "train_loss": float(result.fun),
        "val_loss": _weighted_binary_loss(val_logits, val_y, val_weights),
    }


class _TinyGate(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, value):
        return self.network(value).squeeze(-1)


def _fit_mlp(train_x, train_y, val_x, val_y, args, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = _TinyGate(train_x.shape[1], args.hidden_dim).cpu()
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32)
    x_val = torch.tensor(val_x, dtype=torch.float32)
    y_val = torch.tensor(val_y, dtype=torch.float32)
    count0 = max(int(np.sum(train_y == 0)), 1)
    count1 = max(int(np.sum(train_y == 1)), 1)
    pos_weight = torch.tensor([count0 / count1], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience = 0
    final_train = None
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(x_train)
        train_loss = criterion(train_logits, y_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), y_val).item())
        final_train = float(train_loss.item())
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("MLP failed to record a finite validation state")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "status": "ok",
        "epochs": epoch,
        "best_epoch": best_epoch,
        "train_loss": final_train,
        "val_loss": best_val,
    }


def _predict_logreg(model, x):
    logits = x @ model["coefficient"] + model["intercept"]
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    return probabilities


def _predict_mlp(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    prediction = (np.asarray(probabilities, dtype=np.float64) >= 0.5).astype(np.int64)
    tn = int(np.sum((labels == 0) & (prediction == 0)))
    fp = int(np.sum((labels == 0) & (prediction == 1)))
    fn = int(np.sum((labels == 1) & (prediction == 0)))
    tp = int(np.sum((labels == 1) & (prediction == 1)))
    recall0 = tn / (tn + fp) if tn + fp else 0.0
    recall1 = tp / (tp + fn) if tp + fn else 0.0
    precision1 = tp / (tp + fp) if tp + fp else 0.0
    return {
        "balanced_accuracy": 0.5 * (recall0 + recall1),
        "accuracy": float(np.mean(prediction == labels)),
        "precision_both": precision1,
        "recall_both": recall1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _group_bootstrap_ci(labels, probabilities, groups, seed, samples):
    grouped = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        indices = [index for key in chosen for index in grouped[key]]
        bootstrap_labels = np.asarray(labels)[indices]
        if len(set(bootstrap_labels.tolist())) < 2:
            continue
        bootstrap_probabilities = np.asarray(probabilities)[indices]
        values.append(_metrics(bootstrap_labels, bootstrap_probabilities)["balanced_accuracy"])
    return {
        "num": len(values),
        "low": _percentile(values, 2.5),
        "high": _percentile(values, 97.5),
    }


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _write_report(args, summary):
    data = summary["data"]
    decision = summary["decision"]
    result_lookup = {
        (row["variant"], row["model"], row["evaluation_set"]): row
        for row in summary["results"]
    }
    lines = [
        "# A5 Stable-vs-Both Gate Feasibility Probe",
        "",
        "## Scope",
        "",
        "This is a diagnostic-only binary gate test. It reuses frozen data and does not train experts, MoE, policies, smoothing, or delta models.",
        "",
        "All state/action/contact features are extracted from the same restore-prefilter table. Contact features use only pre-action `anchor_contact`: penetration-weighted object-local point/normal pooling and maximum penetration. No gate reason, finite-difference response, perturbed contact, contact force, or source-protocol field enters the model.",
        "",
        "## Data",
        "",
        f"- source labels before freezing: stable-like `{data['source_label_counts']['0']}`, both-like `{data['source_label_counts']['1']}`;",
        f"- after excluding all frozen diagnostic trajectories: `{data['source_eligible_after_freeze']['0']}/{data['source_eligible_after_freeze']['1']}`;",
        f"- audited test: `{data['audited_counts']['0']}` stable-like and `{data['audited_counts']['1']}` both-like;",
        f"- train/validation to audited trajectory overlap: `{data['train_val_to_audited_group_overlap']}`; source split overlap: `{data['split_group_overlap']}`.",
        "",
        "| Source split | Stable-like | Both-like | Trajectory groups |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "val", "source_test"):
        lines.append(
            f"| {split} | {data['source_split_counts'][split]['0']} | {data['source_split_counts'][split]['1']} | {data['source_split_group_counts'][split]} |"
        )
    for evaluation_set, title in (
        ("source_test", "Source Grouped Test"),
        ("audited_test", "Replay-Audited Frozen Test"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Variant | Model | Balanced accuracy | 95% group CI | Precision both | Recall both |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for variant in FEATURE_VARIANTS:
            for model in MODEL_NAMES:
                row = result_lookup[(variant, model, evaluation_set)]
                lines.append(
                    f"| {variant} | {model} | {_fmt(row['balanced_accuracy'])} | [{_fmt(row['balanced_accuracy_ci_low'])}, {_fmt(row['balanced_accuracy_ci_high'])}] | {_fmt(row['precision_both'])} | {_fmt(row['recall_both'])} |"
                )

    x3_mlp = result_lookup[("X3", "mlp", "audited_test")]
    lines.extend(
        [
            "",
            "## X3 Audited Diagnostics",
            "",
            "X3 MLP confusion matrix (`true` rows, `predicted` columns):",
            "",
            "| | Pred stable | Pred both |",
            "|---|---:|---:|",
            f"| True stable | {x3_mlp['tn']} | {x3_mlp['fp']} |",
            f"| True both | {x3_mlp['fn']} | {x3_mlp['tp']} |",
            "",
            "Audited provenance breakdown:",
            "",
            "| Provenance | Truth | N | Predicted-both rate | Mean p(both) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["audited_provenance_breakdown"]:
        lines.append(
            f"| {row['provenance']} | {row['truth']} | {row['num_rows']} | {_fmt(row['predicted_both_rate'])} | {_fmt(row['mean_probability_both'])} |"
        )
    lines.extend(
        [
            "",
            "Top standardized X3 LogReg coefficients by absolute magnitude:",
            "",
            "| Rank | Feature | Coefficient | Absolute |",
            "|---:|---|---:|---:|",
        ]
    )
    for rank, row in enumerate(summary["x3_top_coefficients"], start=1):
        lines.append(
            f"| {rank} | {row['feature']} | {_fmt(row['coefficient'])} | {_fmt(row['absolute_coefficient'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision Matrix",
            "",
            "The decision uses only X3 MLP balanced accuracy on the replay-audited frozen test.",
            "",
            "| Criterion | Result | Pass |",
            "|---|---:|---|",
            f"| trajectory leakage absent | overlap={data['train_val_to_audited_group_overlap']} | **{decision['trajectory_leakage_pass']}** |",
            f"| audited X3 MLP balanced accuracy >= 0.85 | {_fmt(decision['balanced_accuracy'])} | **{decision['deploy_pass']}** |",
            f"| audited X3 MLP balanced accuracy >= 0.70 | {_fmt(decision['balanced_accuracy'])} | **{decision['soft_gate_floor_pass']}** |",
            "",
            f"**Verdict: `{decision['verdict']}`.**",
            "",
            "No downstream action was started.",
        ]
    )
    if summary["unexpected"]:
        lines.extend(["", "## Unexpected", ""])
        lines.extend(f"- {value}" for value in summary["unexpected"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")


def _train(args):
    paths = _paths(args.out_dir)
    dataset, data_summary = _build_data(args)
    results = []
    predictions = []
    coefficients = []
    training_rows = []
    start = time.perf_counter()
    for variant_index, variant in enumerate(FEATURE_VARIANTS):
        train_rows, x_train, y_train, _, _ = _load_xy(dataset, "train", variant)
        val_rows, x_val, y_val, _, _ = _load_xy(dataset, "val", variant)
        source_rows, x_source, y_source, source_groups, source_ids = _load_xy(
            dataset, "source_test", variant
        )
        audit_rows, x_audit, y_audit, audit_groups, audit_ids = _load_xy(
            dataset, "audited_test", variant
        )
        mean, scale, standardized = _standardize(
            x_train, x_val, x_source, x_audit
        )
        x_train_s, x_val_s, x_source_s, x_audit_s = standardized

        logreg = _fit_logreg(x_train_s, y_train, x_val_s, y_val, args.logreg_l2)
        mlp = _fit_mlp(
            x_train_s,
            y_train,
            x_val_s,
            y_val,
            args,
            args.model_seed + 1009 * variant_index,
        )
        model_records = {
            "logreg": logreg,
            "mlp": mlp,
        }
        for model_name, model in model_records.items():
            training_rows.append(
                {
                    "variant": variant,
                    "model": model_name,
                    "status": model["status"],
                    "iterations_or_epochs": model.get("iterations", model.get("epochs")),
                    "best_epoch": model.get("best_epoch"),
                    "train_loss": model["train_loss"],
                    "val_loss": model["val_loss"],
                }
            )
            for evaluation_set, x_eval, y_eval, groups, anchor_ids, rows_eval in (
                ("source_test", x_source_s, y_source, source_groups, source_ids, source_rows),
                ("audited_test", x_audit_s, y_audit, audit_groups, audit_ids, audit_rows),
            ):
                probabilities = (
                    _predict_logreg(model, x_eval)
                    if model_name == "logreg"
                    else _predict_mlp(model["model"], x_eval)
                )
                metrics = _metrics(y_eval, probabilities)
                ci = _group_bootstrap_ci(
                    y_eval,
                    probabilities,
                    groups,
                    args.bootstrap_seed + 1009 * variant_index + (0 if model_name == "logreg" else 97),
                    args.bootstrap_samples,
                )
                results.append(
                    {
                        "variant": variant,
                        "model": model_name,
                        "evaluation_set": evaluation_set,
                        "num_rows": len(y_eval),
                        "num_stable": int(np.sum(y_eval == 0)),
                        "num_both": int(np.sum(y_eval == 1)),
                        **metrics,
                        "balanced_accuracy_ci_low": ci["low"],
                        "balanced_accuracy_ci_high": ci["high"],
                        "bootstrap_samples": ci["num"],
                    }
                )
                for row, anchor_id, truth, probability in zip(
                    rows_eval, anchor_ids, y_eval.tolist(), probabilities.tolist()
                ):
                    predictions.append(
                        {
                            "variant": variant,
                            "model": model_name,
                            "evaluation_set": evaluation_set,
                            "anchor_id": anchor_id,
                            "trajectory_key": row["trajectory_key"],
                            "audit_provenance": row["audit_provenance"],
                            "truth": truth,
                            "probability_both": probability,
                            "prediction": int(probability >= 0.5),
                        }
                    )
        for index, feature_name in enumerate(FEATURE_NAMES[variant]):
            coefficients.append(
                {
                    "variant": variant,
                    "feature": feature_name,
                    "coefficient": float(logreg["coefficient"][index]),
                    "absolute_coefficient": abs(float(logreg["coefficient"][index])),
                    "train_mean": float(mean[index]),
                    "train_scale": float(scale[index]),
                }
            )
        print(
            f"[gate-train] {variant} "
            f"source={results[-2]['balanced_accuracy']:.3f}/{results[-1]['balanced_accuracy']:.3f} "
            f"logreg_status={logreg['status']} mlp_epoch={mlp['best_epoch']}"
        )

    _write_csv(paths["results"], results)
    _write_csv(paths["predictions"], predictions)
    _write_csv(paths["coefficients"], coefficients)
    _write_csv(paths["training"], training_rows)
    result_lookup = {
        (row["variant"], row["model"], row["evaluation_set"]): row for row in results
    }
    x3_mlp = result_lookup[("X3", "mlp", "audited_test")]
    provenance_rows = defaultdict(list)
    for row in predictions:
        if (
            row["variant"] == "X3"
            and row["model"] == "mlp"
            and row["evaluation_set"] == "audited_test"
        ):
            provenance_rows[row["audit_provenance"]].append(row)
    provenance_breakdown = []
    for provenance, rows in sorted(provenance_rows.items()):
        truths = {int(row["truth"]) for row in rows}
        if len(truths) != 1:
            raise RuntimeError(f"mixed truths in audited provenance {provenance}")
        provenance_breakdown.append(
            {
                "provenance": provenance,
                "truth": next(iter(truths)),
                "num_rows": len(rows),
                "predicted_both_rate": float(
                    np.mean([int(row["prediction"]) for row in rows])
                ),
                "mean_probability_both": float(
                    np.mean([float(row["probability_both"]) for row in rows])
                ),
            }
        )
    balanced_accuracy = float(x3_mlp["balanced_accuracy"])
    trajectory_leakage_pass = data_summary["train_val_to_audited_group_overlap"] == 0
    deploy_pass = balanced_accuracy >= 0.85
    soft_gate_floor_pass = balanced_accuracy >= 0.70
    if not trajectory_leakage_pass:
        verdict = "invalid_trajectory_leakage"
    elif deploy_pass:
        verdict = "gate_deployable"
    elif soft_gate_floor_pass:
        verdict = "marginal_soft_gate"
    else:
        verdict = "gate_not_deployable"
    unexpected = []
    if balanced_accuracy > 0.95:
        unexpected.append(
            "X3 MLP audited balanced accuracy exceeds 0.95; source/feature/split leakage checks are required before interpreting deployment."
        )
    for variant in FEATURE_VARIANTS:
        logreg_score = float(result_lookup[(variant, "logreg", "audited_test")]["balanced_accuracy"])
        mlp_score = float(result_lookup[(variant, "mlp", "audited_test")]["balanced_accuracy"])
        if abs(logreg_score - mlp_score) < 0.01:
            unexpected.append(
                f"{variant} audited LogReg/MLP differ by <0.01 ({logreg_score:.3f}/{mlp_score:.3f}); convergence and linear separability were checked."
            )
    source_x3_mlp = float(
        result_lookup[("X3", "mlp", "source_test")]["balanced_accuracy"]
    )
    audited_gap = balanced_accuracy - source_x3_mlp
    if audited_gap < -0.10:
        unexpected.append(
            f"X3 MLP drops from source grouped test {source_x3_mlp:.3f} to replay-audited test {balanced_accuracy:.3f} (delta {audited_gap:.3f})."
        )
    x3_top = sorted(
        [row for row in coefficients if row["variant"] == "X3"],
        key=lambda row: (-float(row["absolute_coefficient"]), row["feature"]),
    )[:12]
    summary = {
        "description": "A5 stable-like versus both-like deployable gate diagnostic",
        "data": data_summary,
        "training_protocol": {
            "models": "L2 binary logistic regression and Linear-32-ReLU-Linear MLP",
            "train_only_standardization": True,
            "logreg_l2": args.logreg_l2,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "model_seed": args.model_seed,
            "bootstrap_samples": args.bootstrap_samples,
            "elapsed_seconds": time.perf_counter() - start,
        },
        "results": results,
        "audited_provenance_breakdown": provenance_breakdown,
        "x3_top_coefficients": x3_top,
        "decision": {
            "metric": "X3 MLP replay-audited test balanced accuracy",
            "balanced_accuracy": balanced_accuracy,
            "balanced_accuracy_ci_low": x3_mlp["balanced_accuracy_ci_low"],
            "balanced_accuracy_ci_high": x3_mlp["balanced_accuracy_ci_high"],
            "trajectory_leakage_pass": trajectory_leakage_pass,
            "deploy_pass": deploy_pass,
            "soft_gate_floor_pass": soft_gate_floor_pass,
            "verdict": verdict,
        },
        "unexpected": unexpected,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(args, summary)
    print(
        f"[gate-train] audited X3 MLP={balanced_accuracy:.3f} verdict={verdict}"
    )
    print(f"[gate-train] wrote {paths['summary']}")
    print(f"[gate-train] wrote {args.report}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "data", "train"), default="all")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--prefilter", type=Path, default=DEFAULT_PREFILTER)
    parser.add_argument("--dense-manifest", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--final-matrices", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--marginal-selection", type=Path, default=DEFAULT_MARGINAL)
    parser.add_argument("--branch-selection", type=Path, default=DEFAULT_BRANCH_SELECTION)
    parser.add_argument("--branch-summary", type=Path, default=DEFAULT_BRANCH_SUMMARY)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--split-seed", type=int, default=7403)
    parser.add_argument("--model-seed", type=int, default=9301)
    parser.add_argument("--bootstrap-seed", type=int, default=12011)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--logreg-l2", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    args = parser.parse_args()

    if args.stage in ("all", "data"):
        _build_data(args)
    if args.stage in ("all", "train"):
        _train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
