# A5 Roadmap: Action-Side VJP First

Date: 2026-07-10

## Scope Freeze

The immediate research target is the A5 action-side contact VJP:

```text
B_u(z_t) = d object_next_state / d robot_action_t
```

Do not start state-side VJP, hybrid Genesis backward, SHAC/BPTT, Dreamer, or
100k--200k policy comparisons until the action-side gate in this note passes.
Those items remain the downstream roadmap, not current work.

The current action-side bottlenecks are:

1. the position response label is quantized to zero at the short A5 window;
2. the effective number of distinct training contexts is small;
3. the context lacks contact geometry and contains stale nominal-future fields;
4. the current train/test protocol does not provide a clean generalization test;
5. per-anchor subprocess startup and excessive direction probes limit data output.

## Current Status

The action-side gate completed and passed on 2026-07-10. The final
replay-consistent set contains 324 anchors (`168 train / 43 validation / 55 ID
test / 58 OOD test`). Learned held-out median action-gradient cosine is `0.988`,
nonzero rate is 100%, and serial-replay closed-loop descent succeeds on
`112 / 113` held-out anchors. No downstream state-side or policy training work
has started. See `2026-07-10_a5_action_vjp_v2_final.md` for results and caveats.

## Established Diagnosis

Dataset:

```text
analysis/2026-07-09_arx_pusher/stage2_phase_grid_full132/
  a5_stage2_phase_grid_full132.csv
```

Key facts from the 67-anchor / 2609-row dataset:

- `local_response` y signal is nonzero on only `28 / 67` anchors.
- The `39 / 67 zero_vjp_grad` cases are exactly the anchors whose position-y
  response is zero for the selected y-loss cotangent.
- `31` of these anchors have all three position response components equal to
  zero; `8` have some x/z response but zero y response.
- `local_vel_response` y signal is nonzero on `66 / 67` anchors.
- That `66 / 67` figure is a numerical nonzero rate, not a trusted-label rate.
  Across all 67 old anchors, a single per-anchor velocity matrix has median
  held-direction cosine about `0.10`; only `5 / 67` reach cosine `>= 0.7`.
- The median per-anchor velocity-y RMS is approximately `1.12e-4`, but the
  range is very wide. Robust scaling and anchor balancing are required.
- The structured y-split has only `20` train contexts and `47` test contexts.
  The 776 train rows are directional measurements of those 20 contexts, not
  776 independent states.
- The current rich context has 107 dimensions but no contact point, normal, or
  penetration. It also contains nominal post/final information that becomes
  stale after an action update.

`response_steps=1` currently produces `local_steps=response_steps+1`, so the
local label is read after two `scene.step()` calls. The action perturbation is
applied only on the first step. This is still a short enough interval for A5
position differences to collapse at float32 precision.

This also changes the scientific meaning of the label: it estimates
`d v_(t+2) / d a_t`, not the one-transition Jacobian `d s_(t+1) / d a_t`
required by the stated action-side BPTT interface. The first v2 full grid keeps
the old two-step convention for contact-window discovery and an apples-to-
apples label audit. Before the action-side gate, all trusted anchors must be
rechecked with `response_steps=0` (one `scene.step()`). If the single-step
signal is unusable, the method must be named and integrated as a multi-step
VJP repair rather than silently treated as a transition Jacobian.

Source-level audit added three stronger findings:

- the old `_contact_count(scene)` is the total rigid-contact count. It includes
  object-table and arm-table contacts, so it cannot establish that the arm is
  touching the object;
- the old `_nominal_context()` reads `trace[anchor_step]`, which is the state
  after the perturbed transition, while the restored anchor is captured before
  that transition. The nominal context is therefore one simulator step late;
- anchors without an arm-object entity-pair contact still show tiny nonzero
  velocity finite differences. Targeted epsilon sweeps show sign and scale
  instability there, consistent with solver/numerical cross-talk rather than a
  usable contact Jacobian.

The first long-lived v2 collector then exposed a fourth, more serious source
issue. Genesis `Scene.reset(state=...)` assigns the supplied state to
`Scene._init_state`. A subsequent no-argument `scene.reset()` therefore resets
to the previous anchor, not the post-build scene. The worker's FD query changed
the registered initial state, and the next job inherited the preceding object's
quaternion/contact history because `_prepare_anchor()` only reset position and
velocity. Supplying a saved pristine `base_state` for every new job removes the
order dependence exactly: a target anchor preceded by unrelated jobs matches a
fresh worker with maximum state difference `0.0`.

There is a separate restore-versus-replay distinction. `RigidSolverState`
stores generalized/link state but not the complete collider/constraint
warm-start history, while rigid `set_state()` clears those structures. A
mid-trajectory `get_state() -> reset(state=...)` therefore need not reproduce an
uninterrupted contact trajectory. Continuous phase scanning must never insert
such restores. FD labels use a cheap restore stage only as a prefilter and must
be replay-audited from the pristine initial state before becoming final
supervision. On one stable contact the restore/replay y-VJP cosine was
approximately `1.0`; on a marginal contact it remained `0.918--0.994` but its
magnitude differed by about `77%`.

The original full750 v2 run is retained as a diagnostic artifact only. Its
cross-job states are contaminated and it must not enter final training or gate
counts.

The leak-free Phase A matrix-head MLP was consequently close to a zero
predictor. On the old OOD test its relative matrix error was approximately
`1.00` and median action-VJP cosine was negative. This is why Phase B proceeds
with corrected states and contacts even though the naive velocity-label pilot
did not pass: the old dataset is not a valid test of geometry-conditioned
generalization.

## Phase A: Reuse Existing Data

The first v2 pilot must not recollect physics data. The existing CSV already
contains the velocity labels needed to test the main correction.

### A1. Target

Use the first three components of `local_vel_response`:

```text
target = d object_linear_velocity / d robot_action
```

Do not train the six-dimensional object qvel target as one unscaled block.
Angular response is much larger on some anchors and would dominate the loss.

Filtering and metrics must use the exact selected target. Do not retain a row
because `local_state_response` is nonzero while the trained target is zero.

### A2. Per-Anchor Supervision

Fit one robust `3 x 6` action-to-linear-velocity matrix per anchor from its
directional FD rows. Record:

- matrix rank and conditioning;
- train-direction residual;
- held-out-direction residual and cosine;
- contact-trace disagreement across plus/minus probes;
- signal RMS per output component;
- whether the y-loss action VJP is nonzero.

Train the state-to-matrix model with one weighted item per anchor. Directional
rows may still be used in the loss, but they must not make 20 contexts appear
to be 776 independent states.

### A3. Splits

Use disjoint anchor groups:

- train;
- validation for checkpoint and hyperparameter selection;
- in-distribution held-out test;
- separate `obj_y` extrapolation test.

Never select a checkpoint on the test split. Keep interpolation and
extrapolation results separate rather than making the entire test set a single
distribution boundary.

### A4. Baselines And Losses

Compare at least:

- zero matrix;
- global matrix;
- nearest-context matrix;
- matrix-head MLP.

Use robust per-output scaling and equal anchor weights. The primary learning
metric is action-VJP quality for held-out cotangents:

```text
g_action = B_u(z_t)^T lambda_velocity
```

Report cosine, sign agreement, nonzero rate, and downstream descent/ascent.
Response RMSE remains diagnostic and is not the trust signal by itself.

## Phase B: Action-Side Dataset V2

Start new Genesis collection after Phase A determines whether the old data can
support the velocity target. If source audit shows invalid contact identity or
misaligned context, correct those semantics before interpreting MLP capacity.

### B1. Contact Geometry

Capture information available from the nominal forward transition:

- robot qpos and qvel;
- object pose, linear velocity, and angular velocity;
- end-effector/object relative pose and velocity;
- contact link/geom pair identity;
- contact point in an object- or end-effector-relative frame;
- contact normal;
- penetration;
- optionally friction and contact force when reliable.

Keep anchor-time and transition contact features as separate ablations. A
contact that starts on the perturbed transition is absent at the restore
anchor. Its nominal transition geometry is still available when BPTT later
executes the backward pass over the completed forward trace; it is not a policy
observation or test-label lookup.

Variable contact sets require a stable representation. Prefer per-pair
pooling or a small permutation-invariant encoder over raw contact array order.
Contact count alone is not a geometry representation.

### B2. Probe Budget

A `3 x 6` local matrix can be identified by six axis central differences:

```text
6 axes x (plus, minus) = 12 restored rollouts per anchor
```

Use a small number of random directions for validation and nonlinearity tests,
not 32 random directions by default. A provisional budget is six axes plus six
random validation directions, or 24 restored rollouts per anchor. Increase the
budget only for uncertain or nonlinear anchors.

This trades redundant directions for more distinct contact contexts, which is
the current data bottleneck.

### B3. Throughput

Replace one-anchor-per-conda-start collection with long-lived forward-only
workers where practical:

- one `gs.init()` and one `scene.build()` per worker;
- process multiple resettable anchor jobs in that scene;
- batch jobs by compatible scene geometry;
- preserve deterministic restore/repeat checks;
- count all nominal and plus/minus simulator steps.

The initial target is at least 200 usable anchor contexts plus a frozen held-out
set of at least 50 anchors. Keep weak and marginal contacts as gate negatives;
do not silently remove the failure distribution.

Collection is staged to preserve both throughput and transition semantics:

1. scan many phases along an uninterrupted nominal trajectory, with no
   mid-trajectory restore;
2. run cheap corrected-base-state restore probes on contact candidates to
   reject weak, mode-switching, and nonlinear anchors;
3. replay only the restore-trusted subset from the pristine initial state for
   final matrices and restore-versus-replay consistency metrics.

## Phase C: Action-Side Generalization Gate

Choose update scales on validation anchors only, then freeze them for test.
Evaluate all held-out anchors, not a hand-picked subset.

The action-side gate requires:

1. at least 90% nonzero predicted VJP on velocity-active held-out anchors;
2. held-out median action-gradient cosine of at least 0.7 against FD/oracle;
3. at least 70% fixed-protocol forward-loss descent over at least 50 held-out
   anchors;
4. descent success clearly above ascent success;
5. no small subset of high-signal anchors dominating the aggregate result;
6. results reported separately for stable, marginal, interpolation, and
   extrapolation contact cohorts;
7. learned VJP compared with per-anchor LSQ oracle and FD-SPSA in simulator
   calls as well as outcome quality.

If the gate fails, diagnose label quality, contact representation, active gate,
and state coverage before increasing network size.

## Deferred Roadmap

Only after the action-side gate passes:

1. diagnose and repair state-side contact VJP;
2. implement per-timestep residual hybrid VJP rather than a frozen matrix;
3. integrate SHAC/BPTT policy optimization;
4. standardize the A5 RL environment and long-run rollout workers;
5. run 5k--10k policy smoke tests;
6. compare against Dreamer at 100k--200k steps.

For the eventual comparison, pre-register total simulator-step accounting,
restored-label cost, wall-clock, common reward/task/evaluation, and separate
failure diagnostics. These requirements are recorded now but must not distract
from completing the action-side VJP first.
