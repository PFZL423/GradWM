# A5 Action VJP v2: Source-Level Data Diagnosis

Date: 2026-07-10

## Scope

This note records source and data-generation failures found while repairing the
A5 action-side contact VJP dataset. These findings precede model-capacity
changes. State-side VJP, SHAC/BPTT integration, and Dreamer remain out of scope.

## Finding 1: Old Contact Count Was Not Arm-Object Contact

The old datasets used `_contact_count(scene)`, which counts every rigid contact
in the scene. Object-table and arm-table contacts therefore made many anchors
look contact-active even when `obj.get_contacts(with_entity=arm)` was empty.

Observed examples:

- old anchor 51: two real arm-object contacts;
- old anchor 54: no arm-object contact despite a total-contact trace of `[2, 2]`.

Tiny nonzero velocity finite differences on no-arm-object anchors changed sign
and scale across epsilon. They are numerical/solver cross-talk, not usable
contact Jacobians.

## Finding 2: Old Nominal Context Was One Step Late

The old rollout stores `trace[0]` after the first push step. The restored anchor
is captured before the action at `anchor_step`, but `_nominal_context()` reads
`trace[anchor_step]`. Its object state is therefore the post-transition state,
one simulator step later than the state to which the FD branches are restored.

## Finding 3: Velocity Nonzero Rate Was Not a Linearity Gate

The old 67-anchor data has nonzero velocity-y response on `66 / 67` anchors,
but a per-anchor `3 x 6` matrix has median held-direction cosine only about
`0.10`; only `5 / 67` anchors reach cosine `>= 0.7`. Merely changing position
labels to velocity labels does not repair mode-switching or invalid contacts.

## Finding 4: `reset(state)` Mutates The Registered Initial State

Genesis 1.1.1 implements `Scene._reset()` as:

```python
if state is None:
    state = self._init_state
else:
    self._init_state = state
```

Local source:

```text
/data/hayu/envs/genesis/lib/python3.11/site-packages/genesis/engine/scene.py
```

The first v2 worker called `scene.reset(state=anchor_state)` for every FD query,
then began the next job with no-argument `scene.reset()`. The next job therefore
started from the previous anchor. Setting object position with
`zero_velocity=True` did not reset the inherited object quaternion and all
other state.

The fix saves a pristine post-build `base_state` and calls
`scene.reset(state=base_state)` at the start of every independent job/group.
After the fix, a target anchor preceded by two unrelated-speed jobs matches a
fresh-worker target exactly in object pose/qvel, arm qpos/qvel, and tip
pose/velocity: maximum elementwise difference `0.0`.

Affected artifacts are diagnostic only and must not be used for final training:

```text
analysis/2026-07-09_arx_pusher/action_vjp_v2_full750/
analysis/2026-07-09_arx_pusher/action_vjp_v2_full750_one_step/
```

## Finding 5: A Visible `SimState` Restore Is Not An Uninterrupted Contact Replay

`RigidSolverState` stores generalized/link state, while rigid `set_state()`
clears collider and constraint state. It does not restore the complete contact
warm-start history. A phase scanner that peeks forward and restores visible
state changed a later anchor's object qvel by as much as `0.40` relative to a
fresh uninterrupted trajectory.

The corrected phase scanner advances one uninterrupted nominal trajectory. It
records the pre-action state, executes one nominal transition, records contact,
and continues without any mid-trajectory reset. Its phase-56 state matches the
fresh-worker state with maximum difference `0.0`.

For FD labels, restore and full replay were compared directly:

- stable phase 56: y-VJP cosine approximately `1.0`, relative difference below
  `0.2%`;
- marginal phase 96: y-VJP cosine `0.918` at epsilon `0.003` and `0.994` at
  epsilon `0.01`, but magnitude relative difference about `77%`.

Restore probes are therefore retained only as a cheap prefilter. Final trusted
supervision must be replayed from the pristine initial state, or at minimum
pass an explicit replay audit.

A batched-env replay optimization was also tested and rejected. With 30
parallel branch environments, the nominal phase-56 trajectory changed arm and
object state and lost the arm-object contact entirely. Forcing
`GS_PARA_LEVEL=1` (the non-batched GPU level) did not restore equivalence.
Genesis batched rigid dynamics is therefore not treated as a numerically
equivalent acceleration path for this contact dataset. Final labels use serial
pristine replay.

The first 20 restore-trusted anchors were then serially replayed. All 20 passed
the replay matrix gate and restore/replay consistency gate. Velocity-y row
cosine had minimum `0.99987` and median approximately `1.0`; norm ratio ranged
from `0.999` to `1.007`, and median relative error was `2.5e-4`. The stable
restore gate is therefore doing useful scientific work: it excludes the
warm-start-sensitive marginal contacts for which restore/replay scale differed,
while retaining a cohort whose branch dynamics is reproducible.

## Corrected Collection Protocol

1. Freeze anchor-level train/validation/test assignments before labels.
2. Scan contacts along uninterrupted nominal trajectories at one-step timing.
3. Use entity-pair arm-object contact geometry, not total contact count.
4. Run corrected-base-state restore probes to reject weak, nonlinear, and
   contact-signature-switching candidates.
5. Replay restore-trusted candidates from pristine initial state for final FD
   matrices and contact traces.
6. Train only on replay-trusted matrices; choose checkpoints/scales on
   validation anchors and keep all held-out anchors untouched.

## Early Corrected-Data Falsification Check

A partial corrected restore-prefilter set of 300 candidates produced 94 trusted
anchors (`61 train / 15 validation / 18 ID-test`). A one-seed matrix-head pilot
obtained:

- ID-test median velocity-y action-VJP cosine: `0.989`;
- y-row sign agreement: `0.963`;
- global-matrix y cosine: `0.920`;
- nearest-context y cosine: `0.964--0.972`.

On the old contaminated data the comparable MLP cosine was near zero or
negative. This strongly shifts the diagnosis away from MLP capacity and toward
data semantics/reset contamination. It is not the final result: the partial set
contains no OOD anchors and uses restore-prefilter rather than final replay
labels. Contact geometry did not improve ID direction cosine in this small
pilot, although transition-pooled features reduced matrix relative error from
about `0.80` (state-only) to `0.70`.

A later 700-candidate partial set produced 212 trusted anchors, including 55
frozen held-out anchors (`33 ID / 22 OOD`). Validation selected the
transition-pooled model. Its median velocity-y VJP cosine was `0.985` on ID and
`0.991` on OOD, with sign agreement `0.980 / 0.985` and 100% nonzero outputs.
State-only OOD cosine was `0.979`, anchor-pooled was `0.986`, and
transition-pooled was `0.991`. These restore-prefilter results pass the offline
direction/nonzero gate provisionally, but final claims still require replay
labels and forward-loss descent.
