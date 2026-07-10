# A5 Action-Side Contact VJP v2: Final Report

Date: 2026-07-10

## Verdict

The scoped action-side gate passes on the stable-contact cohort. The result is
not a policy-training or Dreamer claim. State-side VJP, SHAC/BPTT, and Dreamer
were not started.

Final learned-VJP results on 113 frozen replay-consistent held-out anchors:

- nonzero predicted velocity-y VJP: `113 / 113`;
- median action-gradient cosine against the per-anchor replay oracle: `0.988`;
- one-step replay loss descent: `112 / 113` (`99.1%`);
- descent with opposite-direction ascent: `112 / 113` (`99.1%`);
- oracle descent: `113 / 113`;
- global-matrix descent: `112 / 113`, but median gradient cosine only `0.911`.

The action-side method is therefore operational and well above the original
gate (`cosine >= 0.7`, `nonzero >= 90%`, `descent >= 70% over >= 50 held-out`).
Its strongest demonstrated advantage over a global matrix is per-anchor
gradient direction, not binary descent on this easy local velocity objective.

## Root Causes Fixed

1. The old contact count included every scene contact rather than arm-object
   entity-pair contacts.
2. The old nominal context was one simulator step later than the restored
   anchor.
3. Velocity numerical nonzero rate was mistaken for finite-scale linearity.
4. Genesis `Scene.reset(state=...)` replaces the registered `_init_state`.
   The first long-lived worker consequently started each new job from the
   previous anchor. Explicit pristine `base_state` resets removed this order
   dependence with maximum reproduced-state difference `0.0`.
5. Visible `SimState` restore does not reproduce uninterrupted contact
   warm-start history. Continuous phase scans now contain no mid-trajectory
   restore, and final matrices use pristine serial replay.
6. Contact point, normal, and force are now represented in a true
   object-local frame rather than a translated world frame.

Genesis batched environments were tested as a replay acceleration and rejected:
they changed the nominal rigid trajectory and contact event even with
`GS_PARA_LEVEL=1`. Throughput was instead recovered by distributing independent
serial workers across four GPUs.

## Data Funnel

Artifacts:

```text
analysis/2026-07-09_arx_pusher/action_vjp_v2_dense_scan26568/
analysis/2026-07-09_arx_pusher/action_vjp_v2_restore_prefilter1200/
analysis/2026-07-09_arx_pusher/action_vjp_v2_replay330/
```

Funnel:

| Stage | Count | Notes |
|---|---:|---|
| Frozen dense candidates | 26,568 | 3 x positions, dense y/speed/phase grid |
| Real one-step arm-object contacts | 5,449 | uninterrupted nominal scan |
| Frozen restore prefilter sample | 1,200 | 600/150/200/250 by split |
| Restore-trusted | 395 | 32.9% active coverage |
| Serial pristine replayed | 330 | selected from restore-trusted only |
| Replay self-gate pass | 326 | four signature-switch negatives |
| Final replay-consistent usable | 324 | 168 train / 43 val / 55 ID / 58 OOD |

The 805 rejected restore candidates remain explicit marginal-contact negatives.
Failure reasons overlap: cross-epsilon inconsistency `676`, held-direction
inconsistency `650`, contact-signature switch `609`, and weak y signal `77`.

The earlier `action_vjp_v2_full750` and `full750_one_step` datasets are retained
only as reset-contamination diagnostics and must not be used for training.

## Offline Generalization

Final matrix manifest:

```text
analysis/2026-07-09_arx_pusher/action_vjp_v2_replay330/final/
  a5_action_vjp_v2_final_matrices.csv
  a5_action_vjp_v2_branch_summary.json
```

Final five-seed training selected a state-only model on validation loss:

| Cohort | Anchors | Learned median y cosine | Sign agreement | Nonzero |
|---|---:|---:|---:|---:|
| ID test | 55 | 0.990 | 0.988 | 100% |
| OOD test | 58 | 0.985 | 0.931 | 100% |

Contact geometry was not the final validation winner. It did help earlier OOD
and matrix-error pilots, so the correct conclusion is that corrected state is
sufficient for this stable cohort; geometry remains useful diagnostic/context,
not a proven necessary component.

## Closed-Loop Gate

Artifacts:

```text
analysis/2026-07-09_arx_pusher/action_vjp_v2_replay330/final/closed_loop/
  a5_action_vjp_v2_closed_loop_summary.json
  a5_action_vjp_v2_closed_loop_val.csv
  a5_action_vjp_v2_closed_loop_test.csv
```

Validation selected a single normalized action scale of `0.01` on 43 anchors;
validation descent and descent/ascent separation were both 100%. The scale was
then frozen for test.

| Cohort | Learned descent | Separated | Median oracle cosine |
|---|---:|---:|---:|
| ID test | 55/55 | 55/55 | 0.990 |
| OOD test | 57/58 | 57/58 | 0.987 |
| Combined | 112/113 | 112/113 | 0.988 |

Signal-quartile descent rates were `96.4%, 100%, 100%, 100%`, so aggregate
success is not carried by a small high-signal subset.

The only learned failure was anchor `25822` (`x=0.31, y=0.088, speed=1.7,
phase=88`). Its true y-VJP norm was `2.46e-4`, predicted norm was `1.51e-2`, and
gradient cosine was `-0.683`. Restore/replay label cosine was `0.999996`, so this
is a genuine low-signal OOD amplitude/direction extrapolation failure.

## Remaining Bottlenecks

1. Stable-contact coverage is 32.9% of the sampled real-contact candidates;
   marginal/mode-switching contacts still need an abstain/gating or mixture
   treatment rather than one local matrix.
2. Matrix magnitude is less accurate than direction. Validation scale
   calibration is currently necessary.
3. A global matrix matches learned binary descent on 112/113 anchors for this
   local objective, despite much worse gradient cosine. Harder heterogeneous
   tasks or policy optimization are required to establish downstream value.
4. The passed objective is a local one-step object-velocity loss. It isolates
   action-side correctness; it does not validate state-side credit assignment
   or long-horizon position reward optimization.

Per the scope freeze, work stops here. The next goal should begin with
state-side contact VJP or SHAC/BPTT integration only after explicitly accepting
these action-side results and caveats.
