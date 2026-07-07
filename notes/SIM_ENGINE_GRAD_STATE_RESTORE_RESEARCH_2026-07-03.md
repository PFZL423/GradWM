# Simulator Gradient and State-Restore Research - 2026-07-03

Status: superseded working note. The current consolidated conclusion is in `notes/RESEARCH_DIRECTION_SUMMARY_2026-07-06.md`, and the detailed state-restore experiment is in `notes/LOCAL_ANCHOR_STATE_RESTORE_SURVEY_2026-07-04.md`.

Update after later experiments: the "Immediate Verification Plan" below has been executed in a minimal two-box setup. Genesis internal `SimState` restore works for no-contact and contact cases in that setup, and manual pickle/cross-process restore also works after clearing scene references and detaching solver tensors. The public `SimState.serializable()` helper is still broken in this installed version, so use a custom wrapper. Treat this file as historical background, not the latest conclusion.

This note records the two practical questions that must be answered before the local-anchor learned-backward route is committed.

## Senior's Two Questions

1. Which simulators / physics frameworks explicitly claim gradient or differentiability support?
2. For a stored online rollout, can the simulator restore an offline state and continue from it?
   - Online: collect `s_t, a_t, s_{t+1}`.
   - Offline: restore `s_t`, apply `a_t` or `a_t + delta_a`, and query the next state.
   - This is needed to collect local perturbation labels around an engine anchor.

Important distinction: storing observations is not enough. The local-anchor route needs a simulator-internal state snapshot, or at least enough state variables to make the next transition reproducible. For contact-rich systems, qpos/qvel alone may be insufficient if solver history, actuator state, warm-start acceleration, contact cache, random seeds, or controller hidden state affects the next step.

## Question 1: Engines / Frameworks Claiming Gradient Support

| Engine / framework | Gradient claim | Current relevance |
| --- | --- | --- |
| Genesis / Genesis World | Official repo describes a multi-physics simulation stack with compiler-level autodiff machinery. Local project evidence shows Genesis 1.1.1 rigid gradients are partially useful, but contact-mediated object-loss-to-action gradients are missing/unreliable in tested setups. | Main forward engine and motivation source. Need be precise: not "Genesis is not differentiable"; rather "standard rigid backward path does not give the contact gradient we need." |
| MuJoCo MJX-JAX | Official MJX docs say MJX-JAX is a JAX reimplementation and "roughly supports gradients"; feature table marks differentiability yes for MJX-JAX and no for MJX-Warp. | Strong candidate for comparison / reference. But MJX-JAX has feature limitations; MJX-Warp is faster but explicitly not autodiff. |
| Brax | Official README calls Brax a fast fully differentiable physics engine and mentions analytic policy gradients, but also says physics users should now use MJX or MuJoCo Warp rather than Brax as a MuJoCo physics wrapper. | Good background / baseline family, less central for current manipulation. |
| NVIDIA Warp | Official README says Warp kernels are differentiable and usable inside PyTorch/JAX/Paddle ML pipelines. | More a differentiable simulation programming framework than a ready robotics engine. Useful if building custom local perturbation kernels. |
| JAX MD | Official README calls it accelerated, end-to-end differentiable molecular dynamics. | Relevant as differentiable physics background, not a robot manipulation engine. |
| Dojo | Paper describes a differentiable physics engine for robotics with hard contact gradients via implicit differentiation. | Important related work for contact-gradient claims. |
| Nimble Physics | Paper describes differentiable articulated rigid-body simulation with hard contact and analytical gradients through LCP contact. | Important related work for rigid contact gradients. |
| DiffTaichi / Taichi differentiable simulation | Paper describes differentiable programming for physical simulation using gradient kernels and reverse replay. | Framework-level related work; not a drop-in robotics simulator. |
| Tiny Differentiable Simulator / DiffCoSim / Jade / other contact simulators | Survey and papers list them as differentiable simulators with contact-gradient formulations. | Good related work list, but API maturity and direct manipulation benchmark usability need separate checking. |
| NVIDIA Newton | Mentioned in some robotics-simulation discussions as differentiable / GPU accelerated, but current public official API evidence should still be checked before relying on it. | Treat as watchlist unless the installed source/API confirms state restore and gradient behavior. |

## Question 2: Offline State Restore Support

| Engine | Restore answer | Confidence | Caveat for local-anchor data collection |
| --- | --- | --- | --- |
| MuJoCo C/Python | Strong yes. Official API includes `mj_stateSize`, `mj_getState`, `mj_setState`, and `mj_copyState`. `mjtState` includes `qpos`, `qvel`, actuator activation, history, warm-start, controls, applied forces, mocap, userdata, plugin state, and `mjSTATE_INTEGRATION`. | High | Need use a broad enough state signature, likely `mjSTATE_INTEGRATION`, not just qpos/qvel. |
| MJX-JAX | Likely yes. `mjx.Data` is a JAX pytree; `mjx.step(model, data)` consumes and returns a data object. | High | Store the full `mjx.Data`, not just observation. MJX-Warp has no autodiff; MJX-JAX has feature limits. |
| Brax | Yes in functional API. `Env.step(state, action) -> State`; `State` contains `pipeline_state`, obs, reward, done, metrics, info. | High | Store full `State` or at least full `pipeline_state`. Current Brax maintainers point physics users toward MJX. |
| Genesis | API exists. Local source shows `scene.get_state()` returns `SimState`, and `scene.reset(state=...)` restores a `SimState`. Rigid state stores qpos, dof velocity/acceleration, link pose/quaternion, inertial shift, mass shift, and friction ratio. | Medium-high | Must experimentally test exact continuation. Local source shows rigid `set_state` clears collider / constraint solver state and zeros current control force/mode, so contact warm-start equivalence is not guaranteed. |
| Warp custom simulators | Usually possible if the state is represented as explicit Warp arrays. | Medium | There is no universal engine-level state API; the user-defined simulator must decide which arrays constitute the state. |
| Isaac Sim / Isaac Lab / PhysX | State manipulation exists in RL workflows, but this is not primarily an autodiff simulator path. | Medium | Useful as a forward engine only if state restore is sufficient; not a gradient engine baseline. |
| Dojo / Nimble / research differentiable engines | Likely possible in principle because state variables are explicit in their optimization/simulation formulation. | Low-medium | Need check actual maintained API and installation practicality before treating as engineering baseline. |

## Genesis Source Notes

Local source checked:

- `/home/ubuntu/miniforge3/envs/genesis/lib/python3.11/site-packages/genesis/engine/scene.py`
- `/home/ubuntu/miniforge3/envs/genesis/lib/python3.11/site-packages/genesis/engine/simulator.py`
- `/home/ubuntu/miniforge3/envs/genesis/lib/python3.11/site-packages/genesis/engine/states/solvers.py`
- `/home/ubuntu/miniforge3/envs/genesis/lib/python3.11/site-packages/genesis/engine/solvers/rigid/rigid_solver.py`
- `/home/ubuntu/miniforge3/envs/genesis/lib/python3.11/site-packages/genesis/engine/solvers/rigid/collider/collider.py`
- `/home/ubuntu/miniforge3/envs/genesis/lib/python3.11/site-packages/genesis/engine/solvers/rigid/constraint/solver.py`

Observed details:

- `Scene.get_state()` returns a `SimState`.
- `Scene.reset(state=...)` accepts a `SimState` and forwards it to solver `set_state`.
- `SimState.serializable()` detaches solver tensors and removes scene references, suggesting cross-process storage may be intended.
- Rigid `set_state` clears or resets collider / constraint solver state before writing qpos/qvel/qacc/link poses.
- Rigid `set_state` also zeros current control force and sets control mode to force, so exact continuation requires replaying the same control/action after restore.

## Immediate Verification Plan

For Genesis, write a continuation test before claiming support:

1. Build a simple scene, preferably two-box push plus a no-contact control.
2. Roll out for `K` steps and save `state_k = scene.get_state()` plus the next action/control `a_k`.
3. Continue the original rollout one step and record `obs_{k+1}^{online}`: qpos/qvel/object pose/contact count/loss.
4. Reset the same scene to `state_k`, reapply `a_k`, step once, and record `obs_{k+1}^{restored}`.
5. Compare online vs restored in no-contact and contact frames.
6. If same-process restore works, call `state.serializable()`, pickle it, reload in a fresh subprocess with the same scene build, and repeat the comparison.

Pass criteria should not require bitwise identity at first. For local perturbation labels, the practical criterion is: restored next-step state and contact set are close enough that local perturbation queries are stable and repeatable.

## Communication Takeaway

The short answer to the senior is:

"有不少引擎/框架宣称可微，包括 MJX-JAX、Brax、Warp、Dojo、Nimble、DiffTaichi/JAX-MD/Genesis 等；但真正适合我们这条线的，不只是有没有梯度，而是能否保存完整 simulator state 后离线恢复并做局部扰动查询。MuJoCo/MJX/Brax 这类 functional/state API 比较明确；Genesis 有 `get_state/reset(state)`，但接触缓存和 constraint warm-start 是否能等价恢复需要马上做 continuation test。"
