# Genesis Contact Gradient Investigation Report

Date: 2026-06-21

## Executive Summary

This investigation supports the project motivation, but the conclusion must be stated precisely:

> Genesis 1.1.1 provides usable gradients for non-contact and self-state rigid dynamics. The problematic regime is contact-mediated action-to-object gradients: the loss is placed on the contacted object or rope, the optimization variable is the robot or pusher action, and the influence must pass backward through contact.

In this regime, forward simulation shows that the action changes the object state, but the standard Genesis backward path can return zero or unreliable gradients to the manipulator-side input. This is directly relevant to manipulation tasks such as pushing, grasping, rope manipulation, insertion, and in-hand manipulation.

## What Was Tested

The tests covered four levels:

| Test | Purpose | Main result |
| --- | --- | --- |
| Rope finite difference | Compare analytic gradient with FD in scripted rope grasp | Non-contact gradients match FD; sustained contact becomes trustworthy only in a tiny local neighborhood |
| Trajectory optimization | Check whether Adam can descend using Genesis gradients | Loss descends, but qvel losses mostly optimize arm/finger self dynamics |
| Rope vs no-rope control | Check whether qvel gradient structure comes from rope contact | Curves are nearly identical, so qvel loss is not contact-sensitive |
| Two-box rigid contact | Minimal action-to-object contact gradient probe | Forward sensitivity is nonzero, but analytic pusher gradient is zero |

## Key Results

### 1. Non-contact gradients are usable

In the approach-only trajectory optimization:

```text
loss: 12.263251 -> 12.024584
relative drop: 1.94620%
grad_nan: 0
```

Finite-difference checks also showed near-perfect agreement in the approach phase:

```text
step 7, no contact: cosine similarity = 0.99999
```

This rules out the simple explanation that "Genesis gradients are globally broken."

### 2. Arm qvel losses are not a good rope-contact diagnostic

For H=30 trajectory loss, rope and no-rope runs were almost identical:

```text
rope:    126.477486 -> 125.428574, relative drop 0.82933%
no-rope: 126.478333 -> 125.429024, relative drop 0.82964%
```

A full 60-step gradient comparison also showed that the close-onset gradient drop appears in both rope and no-rope scenes:

```text
step 15 ratio rope/no-rope = 0.988849
step 17-20 ratio ~= 0.99998+
```

Interpretation: the earlier close-onset gradient drop mostly comes from the arm/finger qvel program and qvel loss structure, not from rope contact gradient flowing back to the arm.

### 3. Sustained rope contact has a small trust region

Finite-difference checks at rope contact step 18 showed:

```text
eps=1e-5: cosine similarity = 0.99760
eps=1e-4: cosine similarity = 0.00456
eps=1e-3: cosine similarity = 0.03503
```

This does not mean every contact gradient is wrong. It means the gradient can be locally valid at a very small perturbation scale, but becomes unreliable once the perturbation changes the contact mode.

### 4. Two-box contact exposes the core failure mode

In the minimal rigid pushing task:

```text
pusher action -> rigid contact -> object state loss
```

Forward simulation is sensitive to pusher velocity, but analytic gradient to the pusher is zero.

Cross-entity box push:

```text
vx=0.39 loss=0.00293188
vx=0.41 loss=0.00291338
FD slope ~= -9.25e-4
analytic pusher grad = 0.0
```

Same-MJCF two-box setup:

```text
vx=0.39 loss=0.21372852
vx=0.41 loss=0.21637067
FD slope ~= 0.1321
analytic pusher grad = 0.0
object grad norm sum ~= 0.9274
```

Additional checks ruled out common user-side explanations:

- Contacts are present throughout the rollout.
- Input tensors are leaf tensors with `requires_grad=True`.
- The result persists for both `each-step` and `initial-only` drive modes.
- The issue is not only caused by using two separate Genesis entities.

## Source-Level Interpretation

Genesis copies gradients back to user velocity tensors through `process_input_grad()`, which reads internal `dofs_state.vel.grad`.

The contact constraint Jacobian builder appears to include both contact bodies, so the issue is not simply that the Jacobian is object-only. Genesis also contains a hand-written constraint adjoint, including `ConstraintSolver.backward()` and kernels that compute gradients of force, Jacobian, reference acceleration, and constraint parameters.

However, source search found no normal `loss.backward()` call path that invokes `ConstraintSolver.backward()` in the standard rigid pipeline. The strongest current explanation is:

> Genesis 1.1.1 has contact constraint adjoint code, but the standard rigid backward path does not propagate solved contact sensitivity back to the manipulator-side action velocity in the tested setups.

This is not a proof that no local patch is possible. It is enough evidence for a research motivation: the off-the-shelf differentiable simulator forward can be useful while the contact-mediated backward path is incomplete or unreliable.

## Implications for SHAC and World Models

Directly running SHAC on Genesis gradients is risky because SHAC needs:

```text
policy action -> robot motion -> contact -> object/rope reward -> gradient -> policy
```

The tested failure mode breaks exactly the contact-to-action part of this chain. SHAC may still learn self-motion or approach behavior, but it may fail to learn genuine manipulation from object-centered rewards.

The stronger project direction is not "train a full world model and run SHAC." That overlaps heavily with OrbiSim. A better positioning is:

> Keep Genesis for forward simulation, geometry, contact detection, and non-contact dynamics. Learn or correct only the missing contact-mediated gradient/local response needed for manipulation policy optimization.

In short:

- OrbiSim: learn a full differentiable simulator.
- This project: repair the contact-gradient path of an existing simulator.

## Recommended Figures and Table

1. **Main figure:** two-box forward FD sensitivity is nonzero, but analytic pusher gradient is zero.
2. **Control figure:** rope/no-rope qvel gradient curves nearly overlap, proving arm qvel loss should not be overinterpreted as rope-contact evidence.
3. **Capability table:**

| Regime | Genesis gradient status |
| --- | --- |
| Non-contact arm/self dynamics | Usable |
| Single object self-state gradient | Usable |
| Arm qvel trajectory loss | Usable but not rope-contact-sensitive |
| Contact-mediated action-to-object gradient | Missing or unreliable in tested setups |

## Next Technical Step

The investigation loop is complete enough for motivation. The next step should be paper/report consolidation, not more blind SHAC runs:

- turn the two-box probe into a clean reproducible figure;
- keep the rope/no-rope comparison as a control;
- define the proposed method as hybrid contact-gradient repair;
- optionally test a small manual hook into Genesis contact adjoint only as a sanity check, not as the main project direction.

## Relevant Artifacts

Scripts:

- `scripts/legacy/grad_fd_check.py`
- `scripts/legacy/traj_opt_grasp.py`
- `scripts/legacy/rope_no_rope_grad_compare.py`
- `scripts/legacy/box_pos_grad_probe.py`
- `scripts/legacy/traj_opt_box_push.py`
- `scripts/legacy/two_box_same_mjcf_contact_grad.py`

Outputs:

- `analysis/grad_fd_check*.csv`
- `analysis/traj_opt_loss_gpu_*.csv`
- `analysis/rope_no_rope_grad_compare.csv`
- `analysis/rope_no_rope_grad_compare.png`
- `analysis/box_push_fd_v039.csv`
- `analysis/box_push_fd_v041.csv`
