# A5 Stable-vs-Both Gate Feasibility Probe

## Scope

This is a diagnostic-only binary gate test. It reuses frozen data and does not train experts, MoE, policies, smoothing, or delta models.

All state/action/contact features are extracted from the same restore-prefilter table. Contact features use only pre-action `anchor_contact`: penetration-weighted object-local point/normal pooling and maximum penetration. No gate reason, finite-difference response, perturbed contact, contact force, or source-protocol field enters the model.

## Data

- source labels before freezing: stable-like `512`, both-like `481`;
- after excluding all frozen diagnostic trajectories: `232/230`;
- audited test: `145` stable-like and `61` both-like;
- train/validation to audited trajectory overlap: `0`; source split overlap: `{'train:val': 0, 'train:source_test': 0, 'val:source_test': 0}`.

| Source split | Stable-like | Both-like | Trajectory groups |
|---|---:|---:|---:|
| train | 136 | 136 | 140 |
| val | 48 | 47 | 29 |
| source_test | 48 | 47 | 30 |

## Source Grouped Test

| Variant | Model | Balanced accuracy | 95% group CI | Precision both | Recall both |
|---|---|---:|---:|---:|---:|
| X1 | logreg | 0.696 | [0.594, 0.802] | 0.655 | 0.809 |
| X1 | mlp | 0.728 | [0.632, 0.820] | 0.678 | 0.851 |
| X2 | logreg | 0.717 | [0.612, 0.820] | 0.679 | 0.809 |
| X2 | mlp | 0.717 | [0.628, 0.802] | 0.679 | 0.809 |
| X3 | logreg | 0.717 | [0.607, 0.824] | 0.679 | 0.809 |
| X3 | mlp | 0.686 | [0.584, 0.787] | 0.639 | 0.830 |

## Replay-Audited Frozen Test

| Variant | Model | Balanced accuracy | 95% group CI | Precision both | Recall both |
|---|---|---:|---:|---:|---:|
| X1 | logreg | 0.480 | [0.399, 0.558] | 0.281 | 0.525 |
| X1 | mlp | 0.510 | [0.436, 0.585] | 0.304 | 0.557 |
| X2 | logreg | 0.511 | [0.430, 0.586] | 0.305 | 0.525 |
| X2 | mlp | 0.539 | [0.460, 0.615] | 0.327 | 0.574 |
| X3 | logreg | 0.511 | [0.429, 0.590] | 0.305 | 0.525 |
| X3 | mlp | 0.527 | [0.450, 0.602] | 0.318 | 0.557 |

## X3 Audited Diagnostics

X3 MLP confusion matrix (`true` rows, `predicted` columns):

| | Pred stable | Pred both |
|---|---:|---:|
| True stable | 72 | 73 |
| True both | 27 | 34 |

Audited provenance breakdown:

| Provenance | Truth | N | Predicted-both rate | Mean p(both) |
|---|---:|---:|---:|---:|
| branch_replay_confirmed_both | 1 | 29 | 0.517 | 0.498 |
| marginal_both | 1 | 32 | 0.594 | 0.520 |
| marginal_signature_only | 0 | 32 | 0.594 | 0.548 |
| v2_heldout_stable | 0 | 113 | 0.478 | 0.472 |

Top standardized X3 LogReg coefficients by absolute magnitude:

| Rank | Feature | Coefficient | Absolute |
|---:|---|---:|---:|
| 1 | contact_point_obj_y | -0.746 | 0.746 |
| 2 | obj_angvel_y | -0.534 | 0.534 |
| 3 | arm_qvel_4 | 0.530 | 0.530 |
| 4 | arm_qvel_5 | 0.486 | 0.486 |
| 5 | contact_normal_obj_z | 0.438 | 0.438 |
| 6 | obj_linvel_y | -0.427 | 0.427 |
| 7 | contact_normal_obj_y | -0.379 | 0.379 |
| 8 | arm_qvel_2 | 0.374 | 0.374 |
| 9 | obj_linvel_z | 0.359 | 0.359 |
| 10 | contact_normal_obj_x | -0.353 | 0.353 |
| 11 | arm_qvel_3 | 0.299 | 0.299 |
| 12 | obj_quat_w | 0.274 | 0.274 |

## Decision Matrix

The decision uses only X3 MLP balanced accuracy on the replay-audited frozen test.

| Criterion | Result | Pass |
|---|---:|---|
| trajectory leakage absent | overlap=0 | **True** |
| audited X3 MLP balanced accuracy >= 0.85 | 0.527 | **False** |
| audited X3 MLP balanced accuracy >= 0.70 | 0.527 | **False** |

**Verdict: `gate_not_deployable`.**

No downstream action was started.

## Unexpected

- X3 MLP drops from source grouped test 0.686 to replay-audited test 0.527 (delta -0.159).
