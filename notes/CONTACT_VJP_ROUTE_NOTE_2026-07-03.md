# Contact VJP Route Note - 2026-07-03

Status: historical route note. For the latest consolidated view, including the Genesis state-restore result and updated "what not to say" list, see `notes/RESEARCH_DIRECTION_SUMMARY_2026-07-06.md`.

## Current Collaboration Situation

- Senior's main route is a general local-anchor learned-backward framework:
  - Engine forward gives accurate transitions.
  - A local anchor model learns finite-scale perturbation response around each true engine transition.
  - Forward remains exactly engine state; backward uses the learned local response.
  - The route is all-step / full-transition, not only contact-window-specific.
- My potential independent route should not fully duplicate that framework.
- Best current positioning:
  - Share infrastructure: tasks, engine rollout, perturbation sampling, ST layer ideas, evaluation metrics.
  - Keep independent claim: contact-rich manipulation needs object-centric through-contact gradient repair.

## After Senior Meeting - 2026-07-03

- Senior did not ask me to abandon my own route. His suggestion is to run two lines in parallel:
  - senior route: a more general, all-step local-anchor learned-backward method;
  - my route: a contact/object-centric method if I can make it technically complete.
- If the two lines merge cleanly, they can become one stronger method. If they do not merge, shared tasks, baselines, datasets, and evaluation code should still be reusable.
- Immediate responsibility from senior:
  - survey whether other simulators claim gradient support;
  - verify whether the engine can restore stored online states offline, so local perturbation data can be queried around real rollout anchors.
- Near-term personal task:
  - turn the contact-local VJP idea into a concrete route, not just an intuition;
  - decide whether the independent claim is "contact-specific backward repair is needed beyond general local-anchor response learning";
  - start from a minimal offline benchmark before full policy optimization.

## Candidate Independent Route

Working title:

`Contact-Local VJP Repair for Engine-Forward Manipulation Optimization`

Core question:

`Is a general local-anchor transition surrogate enough for object-centric through-contact manipulation gradients, or do we need explicit contact/object VJP training?`

Pipeline:

1. Use engine rollout to collect anchors:
   - `sbar_t`, `abar_t`, `sbar_{t+1} = Engine(sbar_t, abar_t)`
   - contact context `c_t`
   - object-side loss gradient `lambda_obj = dL/ds_object`
2. Sample local perturbations:
   - mainly action perturbations `delta_a_t`
   - optionally contact-state perturbations `delta_s_contact`
3. Query engine forward for perturbation labels:
   - `s_{t+1}^{pert} = Engine(sbar_t + delta_s, abar_t + delta_a)`
   - local response label `delta_s_obj = s_{obj,t+1}^{pert} - sbar_{obj,t+1}`
   - finite-scale loss change / directional derivative label from object loss
4. Train a contact/object model:
   - shared contact encoder over anchor, action, contact features, and object context
   - response head predicts local object/rope response
   - VJP head predicts object-loss-to-action gradient `dL/da_t`
   - optional confidence/uncertainty head for gating
5. Use engine-forward / learned-contact-backward optimization:
   - forward rollout state always comes from engine
   - contact windows use learned VJP or confidence-weighted learned backward
   - start with action-sequence optimization before full policy training

Main distinction from senior route:

- Senior route learns a general local perturbation response for the full transition.
- This route targets the contact/object-centric gradient needed by manipulation:
  `object loss -> contact -> robot action`.
- It explicitly tests whether response-autograd is sufficient, or whether VJP/gradient-usefulness training is necessary.

## Near-Term Research Tasks

Senior requested two concrete investigations:

1. Which simulators/physics engines currently claim differentiability or gradient support?
2. For the local-anchor route, whether engines can restore an online state from offline stored state/action data:
   - During online rollout, save a sequence of states and actions.
   - Later, reload the stored state into the engine offline.
   - Continue rollout or query perturbed transitions from the exact stored state.

Need produce a short table:

- engine
- gradient claim/support
- state snapshot/get API
- state restore/set API
- action/control replay support
- limitations for contact/internal solver/cache state
- source links

Important risk:

- Restoring qpos/qvel alone may be insufficient for exact continuation if the engine has hidden solver/contact/warm-start/cache state.
- For local perturbation queries, exact bitwise replay may be less important than restoring enough state to query a consistent next-step transition.
