# Local-Anchor State Restore Survey - 2026-07-04

Status: current working conclusion for senior discussion. This note intentionally treats "other differentiable simulators" as secondary background. The core question is whether an engine can support local-anchor data collection:

`online rollout -> save s_t, a_t -> offline restore s_t -> query Engine(s_t, a_t + delta_a)`.

## Main Takeaway

For Genesis, state restore is usable in a minimal rigid two-box test. In no-contact and contact cases, restoring `scene.get_state()` with `scene.reset(state=...)`, replaying the next velocity action, and stepping once reproduced the online next state to near numerical precision. Perturbation queries from the restored anchor were also exactly repeatable in the tested setup.

Cross-process / disk restore also worked in the minimal test after a small manual workaround: set the top-level `SimState._scene = None`, call each solver state's `serializable()`, pickle the state, rebuild the same scene in a fresh Python process, load the state, call `scene.reset(state=...)`, and replay the action. The public helper `SimState.serializable()` itself fails in the installed version with `AttributeError: property 'scene' of 'SimState' object has no setter`, so the capability is feasible but needs a wrapper/fix before it is treated as a clean API.

## What Senior Actually Needs

The required capability is not just simulator differentiability. It is state restore for local perturbation labels:

1. During online rollout, save the full simulator state `s_t`, the executed action `a_t`, and the next state `s_{t+1}`.
2. Later restore the simulator to `s_t`.
3. Apply `a_t` again and check that `s_{t+1}` is reproduced.
4. Apply nearby actions `a_t + delta_a` and query local response labels.

For contact-rich manipulation, storing only observation-level data is probably not enough. A useful saved state may need qpos, qvel, actuator/control state, contact/warm-start state, material parameters, plugin state, random state, and controller hidden state depending on the engine.

## Genesis Source Evidence

Local Genesis source paths checked:

- `/home/ubuntu/miniforge3/envs/genesis/lib/python3.11/site-packages/genesis/engine/scene.py`
- `/home/ubuntu/miniforge3/envs/genesis/lib/python3.11/site-packages/genesis/engine/simulator.py`
- `/home/ubuntu/miniforge3/envs/genesis/lib/python3.11/site-packages/genesis/engine/states/solvers.py`
- `/home/ubuntu/miniforge3/envs/genesis/lib/python3.11/site-packages/genesis/engine/solvers/rigid/rigid_solver.py`
- `/home/ubuntu/miniforge3/envs/genesis/lib/python3.11/site-packages/genesis/engine/solvers/rigid/collider/collider.py`
- `/home/ubuntu/miniforge3/envs/genesis/lib/python3.11/site-packages/genesis/engine/solvers/rigid/constraint/solver.py`

Relevant observations:

- `Scene.get_state()` returns a `SimState`.
- `Scene.reset(state=...)` accepts a `SimState` and forwards solver states into each solver's `set_state`.
- Rigid state stores qpos, dof velocity, dof acceleration, link position, link quaternion, inertial position shift, mass shift, and friction ratio.
- Rigid `set_state` clears collider / constraint solver state before writing the restored state. It also zeros current control force and resets control mode, so the action/control must be replayed after restore.
- Because contact caches are cleared, the restored anchor snapshot can show contact count 0 before the next collision pass, even if the original anchor had active contacts. After one restored step in the experiment, the contact count recovered.
- `SimState.serializable()` appears intended for detaching state from a scene, but currently fails in this installed version because it tries to assign `self.scene = None` while `scene` is a read-only property. A manual wrapper that writes `state._scene = None` before serializing solver states worked in the minimal cross-process test.

## Genesis Experiment

Added script:

`scripts/genesis_state_restore_check.py`

Commands run:

```bash
conda run -n genesis --no-capture-output python scripts/genesis_state_restore_check.py --cases no_contact --out analysis/genesis_state_restore_no_contact.json
conda run -n genesis --no-capture-output python scripts/genesis_state_restore_check.py --cases contact --out analysis/genesis_state_restore_contact.json
```

The machine reported `gs.gpu` unavailable and fell back to CPU. This is acceptable for the first state-restore check, but GPU should be retested later if the production data generator uses GPU.

Experiment setup:

- Same-MJCF two-box rigid scene.
- No ground plane, so `no_contact` means true zero contact.
- `contact` starts with two boxes in contact.
- Roll `pre_steps=8`, save `anchor_state = scene.get_state()`.
- Compare:
  - original online next step under `vx=0.40`;
  - restored next step under the same `vx=0.40`;
  - two repeated restored perturbation queries under `vx=0.45`;
  - manual pickle/cross-process restore of the anchor state.

Results:

| Case | Anchor contacts | Same-process same-action restore | Repeated perturbation query | Manual cross-process pickle restore | Public `SimState.serializable()` |
| --- | ---: | --- | --- | --- | --- |
| no_contact | 0 | exact match, max state diff `0.0` | exact match, max state diff `0.0` | exact match, max state diff `0.0` | failed: `AttributeError` |
| contact | 5 | near match: qpos/link pos `~1.49e-11`, qvel `~1.73e-7`, dofs_acc `~2.56e-4`; contact count recovered to 5 | exact match, max state diff `0.0` | same as same-process: qpos/link pos `~1.49e-11`, qvel `~1.73e-7`, dofs_acc `~2.56e-4` | failed: `AttributeError` |

Interpretation:

- Local-anchor querying is promising in both same-process and manually serialized cross-process settings.
- Contact restore is not bitwise identical at the acceleration/cache level, but the next qpos/qvel/link state is close enough for a first local response data-collection test.
- The contact cache clearing behavior matters if the model input needs contact features exactly at anchor time. Contact features may need to be saved from the online rollout, recomputed carefully, or queried after a collision update.
- Cross-process restore should use an explicit wrapper rather than the broken public `SimState.serializable()` helper.

## State-Restore Comparison Table

| Engine / framework | Restore capability for local-anchor queries | Confidence | Notes |
| --- | --- | --- | --- |
| MuJoCo C/Python | Strong. Official API exposes state size/get/set/copy functions and state bitmasks. | High | For exact continuation, use a broad state spec such as integration state, not just qpos/qvel. |
| MJX-JAX | Strong in principle. Dynamics are functional over JAX pytrees such as `mjx.Data`. | High | Store the full data pytree. MJX-JAX has differentiability support; MJX-Warp does not claim autodiff. |
| Brax | Strong. Environment API is functional: `step(state, action) -> state`. | High | Store full `State` / `pipeline_state`, not just observations. More relevant as background than as current manipulation engine. |
| Genesis | Same-process restore passed the minimal two-box test. Manual pickle/cross-process restore also passed after clearing scene references and detaching solver tensors. | Medium-high | Public `SimState.serializable()` is broken in this installed version; contact cache/warm-start is cleared on reset; action/control must be replayed. |
| Warp custom simulation | Depends on the user-written simulator. | Medium | If all state is explicit arrays, restore is natural. There is no universal robotics-engine state API. |
| Isaac Sim / Isaac Lab / PhysX | State manipulation exists for RL workflows. | Medium | More useful as a forward engine; not the primary differentiable/local-anchor target here. |

## Differentiable-Engine Background

This is secondary because MuJoCo/Newton/Genesis have already been assigned and investigated in the group. Still, useful names to mention:

| Engine / framework | Gradient claim relevance |
| --- | --- |
| Genesis | Claims differentiable simulation, but our local result is that rigid self/non-contact gradients are usable while contact-mediated object-loss-to-action gradients are missing/unreliable in tested Genesis 1.1.1 setups. |
| MuJoCo MJX-JAX | Official MJX docs mark MJX-JAX as supporting differentiability, while MJX-Warp does not support autodiff. |
| Brax | Official README describes Brax as a fully differentiable physics engine, though current docs point many physics users toward MJX / MuJoCo Warp. |
| NVIDIA Warp | Official docs describe differentiable kernels usable in ML pipelines; it is a framework rather than one fixed robotics simulator. |
| Dojo / Nimble / DiffTaichi / JAX MD / Tiny Differentiable Simulator | Useful related work family for differentiable physics and contact gradients, but API maturity and direct manipulation usefulness need separate checking. |

## Recommended Next Step

For the senior's line, I would not spend much more time surveying engines unless a specific alternative engine becomes necessary. The important next experiment is:

1. Extend `scripts/genesis_state_restore_check.py` from two-box to the actual task family, starting with a small contact manipulation scene.
2. Save online contact features along with `SimState`, because Genesis reset clears contact/cache state.
3. Query 8-32 local perturbations around the same anchor and measure repeatability plus local response smoothness.
4. Decide the data-generation architecture:
   - same-process worker: feasible in the minimal test;
   - cross-process saved-state replay: feasible in the minimal test with a custom serializer wrapper, but do not rely on the current public `SimState.serializable()` implementation directly.

## Short Communication Version

"我查了一下，关键不是还有没有别的可微引擎，而是 local-anchor 需要的 state restore。Genesis 源码上有 `get_state/reset(state)`，我做了 two-box continuation test：无接触时恢复后一步完全一致，接触时 qpos/qvel/link pose 也几乎一致，扰动查询可重复。手动 detach 后 pickle 到新进程再恢复也能复现，所以用 Genesis 做 anchor 附近扰动采样是有希望的。但它 reset 会清 contact/constraint cache，恢复瞬间 contact count 会丢，要保存或重算 contact features；另外当前公开的 `SimState.serializable()` helper 本身会报错，需要我们写一个小 wrapper。"

## External Sources

- MuJoCo API functions: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
- MuJoCo API types / state bitmasks: https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html
- MuJoCo MJX docs: https://mujoco.readthedocs.io/en/stable/mjx.html
- Brax repository: https://github.com/google/brax
- NVIDIA Warp repository: https://github.com/NVIDIA/warp
- JAX MD repository: https://github.com/jax-md/jax-md
