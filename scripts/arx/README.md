# ARX/A5 Scripts

The current implementation is the A5 action-side contact VJP v2 pipeline.
Results and caveats are recorded in
`notes/a5_vjp_progress/2026-07-10_a5_action_vjp_v2_final.md`.

## Current Pipeline

- `a5_action_vjp_v2_phase_a.py`: audit the old dataset and run leak-free
  velocity-label pilots.
- `a5_action_vjp_v2_linearity_audit.py`: summarize per-anchor finite-scale
  linearity and held-direction consistency.
- `a5_action_vjp_v2_epsilon_summary.py`: compare epsilon-scale stability.
- `a5_action_vjp_v2_collect_worker.py`: run one pristine serial Genesis worker
  for continuous scans, restore prefilters, or replay labels.
- `a5_action_vjp_v2_collect_grid.py`: schedule independent collectors across
  GPUs without sharing Genesis scenes.
- `a5_action_vjp_v2_build_anchors.py`: build frozen train/validation/ID/OOD
  anchor manifests from collection output.
- `a5_action_vjp_v2_compare_branches.py`: audit restore labels against pristine
  replay labels.
- `a5_action_vjp_v2_train_trusted.py`: train and evaluate matrix-head models on
  replay-consistent anchors.
- `a5_action_vjp_v2_closed_loop.py`: select update scale on validation anchors
  and run the frozen held-out one-step descent gate.
- `a5_marginal_cohort_probe.py`: freeze and evaluate rejected contact cohorts
  with two independent pristine-replay direction sets, Genesis analytic
  gradients, and the already selected closed-loop scale. It does not retrain
  the stable model.
- `a5_both_branch_collect_worker.py`: collect dense one-sided pristine-replay
  responses and post-transition contact diagnostics for the frozen `both`
  branch probe.
- `a5_both_branch_probe.py`: freeze the 30-anchor diagnostic cohort, schedule
  one serial lane per GPU, fit held-out piecewise-linear maps, audit selector
  predictability, and emit the preregistered pass/fail report. It does not train
  a branch-conditional network.

## Supporting Probes

The older tracked `a5_*` scripts remain useful only as scene builders and
diagnostic controls. They are not the authoritative v2 data path. In
particular, do not use the old total contact count, late nominal context, or a
no-argument reset inside a long-lived collector.

## Outputs

Generated outputs are local and live under:

```text
analysis/2026-07-09_arx_pusher/
```

The final replay-consistent artifacts are under
`action_vjp_v2_replay330/final/`. The stable-contact action-side gate passed on
113 held-out anchors. The follow-up `marginal_probe/` result rejects a single
direction-independent matrix target for cross-epsilon/mode-mixed contacts, so
contact-feature training was not started. The subsequent `both_branch_probe/`
diagnostic also rejects a small branch-set repair under the preregistered K and
selector gates. State-side VJP and policy training have not started.
