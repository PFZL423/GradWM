# Research Direction Summary - 2026-07-06

This is the current consolidated record. It replaces the scattered working conclusions in earlier notes when there is a conflict.

## Current One-Line Position

The strongest direction is not to rely on Genesis contact gradients directly, and not to learn a full OrbiSim-style simulator. The current direction is:

> Use Genesis as the accurate forward engine and data oracle, then learn a local differentiable backward / response model around real rollout states, especially for the contact-mediated object-loss-to-action path.

## Established Genesis Facts

Genesis is not globally broken. Non-contact and self-state rigid gradients can be usable. In approach-only trajectory optimization, loss decreased by about `1.95%` with no gradient NaNs, and finite-difference checks in no-contact phases matched analytic gradients well.

The problematic regime is specifically:

```text
object/rope loss -> contact -> robot or pusher action
```

This is the path needed for manipulation rewards. In the two-box contact probe, forward simulation clearly changes the object state when pusher velocity changes, but analytic gradient to the pusher action is zero in the tested Genesis 1.1.1 rigid backward path. This supports the motivation for learned-backward / contact-gradient repair.

The earlier rope close-onset gradient drop should not be used as evidence that rope contact gradients flow correctly. Rope/no-rope controls showed the arm-qvel gradient curves are almost identical around close onset, so that signal mainly comes from arm/finger qvel loss structure, not from rope contact gradient back to the arm.

The current careful wording is:

> Genesis 1.1.1 standard rigid backward pipeline has usable smooth/self gradients, but contact-mediated action-to-object gradients are missing or unreliable in our tested setups.

## Senior's Route

Senior's main route is a general local-anchor learned-backward framework:

1. Run the real engine online and get accurate anchors:
   `sbar_t, abar_t, sbar_{t+1} = Engine(sbar_t, abar_t)`.
2. Around each anchor, sample local perturbations:
   `delta_s_t, delta_a_t`.
3. Query the engine forward:
   `Engine(sbar_t + delta_s_t, abar_t + delta_a_t)`.
4. Train a local model to predict finite-scale response around the anchor:
   `m_theta(anchor, delta_s_t, delta_a_t) ~= Engine(perturbed) - Engine(anchor)`.
5. During optimization, keep forward rollout tied to the engine, but use the learned local model / learned Jacobian for backward.

This route is all-step and full-transition oriented. It is not limited to contact windows. Its central claim is that a local learned response around real engine states can provide a useful differentiable backward path while preserving engine forward accuracy.

The key engineering requirement behind this route is state restore:

```text
online rollout -> save s_t, a_t -> offline restore s_t -> replay a_t or a_t + delta_a
```

## My Current Route

The independent route should not simply duplicate senior's full-transition local-anchor model. The best current positioning is:

> Contact/object-centric VJP repair for engine-forward manipulation optimization.

The core question is:

> Is general local-anchor response learning enough for object-centric through-contact manipulation gradients, or do we need explicit contact/object VJP training?

Candidate pipeline:

1. Use Genesis rollout to collect anchors:
   `sbar_t, abar_t, sbar_{t+1}`, online contact features, object/rope state, and task loss context.
2. Restore anchors and query local perturbations:
   mainly `delta_a_t`, optionally contact-local `delta_s_t`.
3. Train a contact/object model with two related targets:
   - response head: local object/rope response, such as `delta pose`, `delta velocity`, or rope endpoint displacement;
   - VJP / gradient head: object-loss-to-action direction, such as `dL_object / da_robot`.
4. Use Genesis for forward rollout and learned contact/object backward for optimization.
5. Start with offline gradient/usefulness benchmarks and action-sequence optimization before policy training.

This route can still share infrastructure with senior's method: engine anchors, state restore, perturbation sampling, tasks, baselines, and evaluation. The independent claim is narrower and more manipulation-specific: contact-rich tasks may need explicit object-centric backward repair, not only a general full-state response model.

## Relation Between The Two Routes

The routes are compatible but not identical.

Senior route:

```text
general full-transition local response around every engine anchor
```

My route:

```text
contact/object-centric local response and VJP for the object-loss-to-action path
```

Possible merged version:

```text
general local-anchor response backbone + contact/object VJP head or contact-specific loss
```

If the methods merge, the contact/object route can become an ablation or extension that shows why explicit manipulation-gradient supervision helps. If they do not merge, the contact route can still be a smaller independent paper/experiment line.

## Latest State-Restore Finding

Genesis state restore is more promising than the early July 3 note suggested.

Implemented test:

- `scripts/genesis_state_restore_check.py`
- outputs:
  - `analysis/genesis_state_restore_no_contact.json`
  - `analysis/genesis_state_restore_contact.json`

Minimal two-box result:

| Case | Result |
| --- | --- |
| no contact | restore + replay exactly matches online next state; max diff `0.0` |
| contact | qpos/link pos diff about `1.49e-11`, qvel diff about `1.73e-7`, main residual in `dofs_acc` about `2.56e-4`; contact count recovers after one step |
| repeated perturbation query | exact repeat in tested setup |
| cross-process restore | works with manual pickle wrapper after clearing scene references and detaching solver tensors |
| public `SimState.serializable()` | broken in installed version: `AttributeError: property 'scene' of 'SimState' object has no setter` |

Meaning:

> If we save Genesis internal `SimState`, local-anchor data collection is initially feasible in a minimal rigid contact scene, including manual cross-process replay. This does not prove that saving only observation arrays is enough.

Important caveat:

Genesis reset clears collider / constraint cache. At a restored contact anchor, contact count can be 0 immediately after reset even if the online anchor had contacts; after one step, contacts recover. Therefore contact features should be saved from the online rollout or recomputed through a controlled procedure, not blindly read immediately after reset.

## What Is Outdated And Should Not Be Reused

Do not say:

```text
Genesis cannot restore online states offline.
```

Current correction: Genesis can restore internal `SimState` in the minimal test; manual pickle cross-process restore works. The public helper is buggy, not the whole capability.

Do not say:

```text
Cross-process state restore is unverified.
```

Current correction: it is verified in the two-box minimal setup with a custom wrapper. It is still unverified for full rope/manipulation scenes.

Do not say:

```text
The rope close-onset gradient drop proves contact gradients are flowing.
```

Current correction: rope/no-rope curves almost overlap; the signal is mostly arm/finger qvel loss structure.

Do not frame the main task as:

```text
Find more differentiable simulators.
```

Current correction: that is secondary background. The main requirement is state restore for local-anchor perturbation data.

Do not overclaim:

```text
Genesis gradients are useless.
```

Current correction: smooth/self gradients can be usable; the failure is the contact-mediated action-to-object path.

## What To Say To Senior

Short version:

> 我现在理解您的路线核心不是再找一个可微仿真器，而是需要仿真器能把在线 rollout 的 `s_t, a_t` 存下来，离线恢复到同一个 `s_t`，再 replay 或扰动 action 来采局部 response 数据。我做了 Genesis two-box restore test：无接触完全一致，接触时 qpos/qvel/link pose 也几乎一致，手动 pickle 到新进程再恢复也能复现。所以 Genesis 作为 local-anchor forward oracle 初步可行。但 reset 会清 contact/constraint cache，恢复瞬间 contact feature 不能直接信，需要在线保存或稳定重算；公开 `SimState.serializable()` 有 bug，要写 wrapper。

Then add:

> 我自己的线可以和您的线共用这些 anchor/perturbation 数据，但我想重点测试 contact/object-centric VJP 是否比一般 full-transition response 更适合 manipulation 的 object-loss-to-action 梯度。

## Immediate Next Steps

1. Extend state-restore test from two-box to the actual task family:
   - first small rigid contact manipulation;
   - then rope/grasp scene.
2. Implement a clean Genesis state export/import wrapper:
   - avoid relying on the broken top-level `SimState.serializable()`;
   - save action, online contact features, object state, and metadata together.
3. Build a tiny local perturbation dataset:
   - one contact anchor;
   - 8-32 action perturbations;
   - compare response smoothness and repeatability.
4. Define the first benchmark before policy training:
   - finite-difference / SPSA alignment;
   - one-step predicted improvement;
   - action-sequence optimization loss drop.
5. Compare three backward options:
   - Genesis analytic gradient;
   - senior-style local response autograd;
   - explicit contact/object VJP head.

## Existing References In This Repo

- `notes/GENESIS_CONTACT_GRADIENT_REPORT.md`: detailed Genesis contact-gradient failure report.
- `notes/CONTACT_VJP_ROUTE_NOTE_2026-07-03.md`: earlier route note after senior meeting.
- `notes/LOCAL_ANCHOR_STATE_RESTORE_SURVEY_2026-07-04.md`: state-restore experiment details.
- `scripts/genesis_state_restore_check.py`: state restore / cross-process replay script.
- `scripts/two_box_same_mjcf_contact_grad.py`: two-box contact-gradient probe.
- `scripts/rope_no_rope_grad_compare.py`: rope/no-rope gradient comparison.
- `analysis/genesis_state_restore_no_contact.json`: no-contact state-restore result.
- `analysis/genesis_state_restore_contact.json`: contact state-restore result.
