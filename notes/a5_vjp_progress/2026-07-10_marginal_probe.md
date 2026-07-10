# A5 Marginal Cohort Phase-1 Probe

Date: 2026-07-10

## Scope

The passed action-side v2 stable gate and all of its collection/reset/restore
protocols remained frozen. This probe evaluates a separate rejected cohort; it
does not retrain the stable model and does not start state-side VJP, SHAC, policy
training, Dreamer, MoE, or an abstention gate.

## Commands

```bash
conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage select
conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage replay --gpu-ids 0,1,2,3
conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage labels --gpu-ids 0,1,2,3
conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage analytic --gpu-ids 0,1,2,3 --workers 8
conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage closed --gpu-ids 0,1,2,3 --workers 8
conda run -n genesis --no-capture-output python scripts/arx/a5_marginal_cohort_probe.py --stage finalize --gpu-ids 0,1,2,3
```

The coordinator launched two independent pristine-replay direction sets, one
fresh-process analytic backward per replay-complete anchor, and fixed-scale
closed-loop replay workers.

## Frozen Cohort

- selected: `96` anchors;
- replay-complete: `96`;
- subtype counts: `{'cross_only': 32, 'signature_only': 32, 'both': 32}`;
- weak-y-only candidates were excluded; no Phase-1 anchor may be used for
  later Phase-2 training.

## Label Reliability

| Metric | Median | Q25 | Q75 |
|---|---:|---:|---:|
| independent label y-cosine | 0.363 | -0.168 | 0.887 |
| cross-probe held y-cosine (min) | 0.234 | -0.161 | 0.787 |
| epsilon y-cosine (min) | -0.159 | -0.485 | 0.522 |
| oracle axis held y-cosine | 0.548 | -0.123 | 0.925 |
| same-axis replay y-cosine | 1.000 | 1.000 | 1.000 |
| same-axis replay max abs | 0.000 | 0.000 | 0.000 |
| maximum norm ratio | 3.684 | 2.286 | 6.114 |

## Baseline Results

All closed-loop methods use the stable validation scale `0.01`; no marginal
scale tuning was performed. Global LSQ is frozen from the stable training split.

| Method | Median y-cosine | Sign agreement | Nonzero | Descent | Separated |
|---|---:|---:|---:|---:|---:|
| zero | n/a | 0.000 | 0.000 | 0.000 | 0.000 |
| global | 0.464 | 0.663 | 1.000 | 0.802 | 0.594 |
| learned | 0.555 | 0.668 | 1.000 | 0.812 | 0.635 |
| analytic | n/a | 0.000 | 0.000 | 0.000 | 0.000 |
| oracle | 1.000 | 1.000 | 1.000 | 0.667 | 0.500 |

## Stable-Cohort Degradation

- stable learned median cosine: `0.988`;
- marginal learned median cosine: `0.555`;
- cosine change: `-0.432`;
- stable learned descent: `0.991`;
- marginal learned descent: `0.812`;
- descent change: `-0.179`.

## Results By Marginal Subtype

| Subtype | Replay N | Label-label cosine | Oracle held cosine |
|---|---:|---:|---:|
| cross_only | 32 | 0.196 | 0.343 |
| signature_only | 32 | 0.991 | 0.997 |
| both | 32 | 0.002 | 0.108 |

| Subtype | Method | N | Median y-cosine | Descent |
|---|---|---:|---:|---:|
| cross_only | zero | 32 | n/a | 0.000 |
| cross_only | global | 32 | 0.391 | 0.750 |
| cross_only | learned | 32 | 0.439 | 0.875 |
| cross_only | analytic | 32 | n/a | 0.000 |
| cross_only | oracle | 32 | 1.000 | 0.688 |
| signature_only | zero | 32 | n/a | 0.000 |
| signature_only | global | 32 | 0.892 | 0.906 |
| signature_only | learned | 32 | 0.947 | 0.875 |
| signature_only | analytic | 32 | n/a | 0.000 |
| signature_only | oracle | 32 | 1.000 | 0.969 |
| both | zero | 32 | n/a | 0.000 |
| both | global | 32 | 0.152 | 0.750 |
| both | learned | 32 | 0.089 | 0.688 |
| both | analytic | 32 | n/a | 0.000 |
| both | oracle | 32 | 1.000 | 0.344 |

## Scientific Interpretation

The two replay sets reproduce every shared axis response exactly
(`cosine=1.0`, maximum absolute difference `0.0`), so the disagreement
between independently fitted random-direction matrices is not simulator
noise. It is direction/contact-branch dependence at the tested finite scales.

`signature_only` remains a coherent matrix cohort, while `cross_only` and
`both` do not. A contact-mode feature encoder could describe the anchor more
richly, but it cannot turn a direction-dependent target into one fixed matrix.
The overall learned/global descent rates are therefore useful behavior checks,
not evidence that the combined random-LSQ matrix is valid ground truth.

## Decision Matrix

The decision is label-reliability-first; model cosine is interpreted only
after testing whether one local matrix is a coherent target.

| Ordered rule | Phase-2 direction |
|---|---|
| replay direction unreliable or oracle descent < 0.7 | single-matrix label failure; redesign branch/direction target |
| reliable, learned cosine >= 0.5 and descent >= 0.7 | no contact-feature Phase 2 |
| reliable, learned cosine < 0.5, median magnitude spread > 2x | direction-only target redesign |
| reliable, learned cosine in [0.2, 0.5), magnitude stable | contact-mode-aware features |
| reliable, learned cosine < 0.2, magnitude stable | severe context OOD; contact features, not automatic label redesign |

| Condition | Result |
|---|---|
| replay labels direction-reliable | `False` |
| oracle closed-loop ceiling passes | `False` |
| marginal amplitude stable within 2x median ratio | `False` |
| selected next direction | **`single_matrix_label_failure`** |

The marginal cohort does not provide a sufficiently coherent single-matrix target or oracle descent ceiling. Contact features cannot repair an ill-defined label; redesign must condition on direction/contact branch before model expansion.

## Unexpected Observations

- Genesis analytic statuses: {'ok': 96}; nonzero velocity-y rows: 0/96.
- Contact-signature changes remain common under target-scale perturbations.
- Identical axis probes reproduced exactly across replay sets; random-set label disagreement is direction/branch dependence, not run-to-run noise.
- The first closed-loop coordinator launch exposed a new-script-only Torch default-device bug after gs.init(gpu); forcing checkpoint inputs to CPU fixed it, and all 12 batches were rerun successfully.

## Artifacts

- `/data/hayu/code/GradWM/analysis/2026-07-09_arx_pusher/marginal_probe/a5_marginal_probe.csv`
- `/data/hayu/code/GradWM/analysis/2026-07-09_arx_pusher/marginal_probe/a5_marginal_probe_summary.json`
- `/data/hayu/code/GradWM/analysis/2026-07-09_arx_pusher/marginal_probe/a5_marginal_probe_frozen_selection.csv`
- `/data/hayu/code/GradWM/analysis/2026-07-09_arx_pusher/marginal_probe/a5_marginal_probe_labels.csv`
- `/data/hayu/code/GradWM/analysis/2026-07-09_arx_pusher/marginal_probe/a5_marginal_probe_analytic.csv`
- `/data/hayu/code/GradWM/analysis/2026-07-09_arx_pusher/marginal_probe/a5_marginal_probe_closed_loop.csv`
