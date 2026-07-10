# Current Research Notes

Updated: 2026-07-10

The current A5 contact-VJP record consists of six documents:

1. `a5_vjp_progress/2026-07-10_a5_action_side_first_roadmap.md`
   records the accepted route, gates, and downstream order.
2. `a5_vjp_progress/2026-07-10_a5_action_vjp_v2_final.md`
   is the authoritative action-side result and bottleneck summary.
3. `a5_vjp_progress/2026-07-10_action_vjp_v2_source_diagnosis.md`
   records the Genesis reset/restore and data-generation root causes.
4. `a5_vjp_progress/2026-07-10_marginal_probe.md`
   tests the rejected cohort and records why a single local matrix is not a
   reliable target for cross-epsilon/mode-mixed contacts.
5. `a5_vjp_progress/2026-07-10_both_branch_probe.md`
   tests whether the dominant `both` cohort admits a small piecewise-linear
   Jacobian set and a deployable branch selector.
6. `a5_vjp_progress/2026-07-10_gate_feasibility_probe.md`
   tests whether frozen forward-pass state, contact geometry, and nominal
   action features can deployably separate stable-like from `both` anchors.

Current status: the stable-contact action-side gate passed with `112 / 113`
held-out descent and median oracle-gradient cosine `0.988`. State-side VJP,
SHAC/BPTT, and Dreamer comparisons have not started. The remaining action-side
limitations are `32.9%` stable-contact coverage, magnitude calibration, and no
binary-descent advantage over the global matrix on the current local task.

The frozen 96-anchor marginal probe found that `signature_only` labels remain
coherent, but `cross_only` and `both` do not admit a reliable direction-
independent local matrix at the tested finite scales. The contact-feature Phase
2 was therefore not started.

The follow-up 30-anchor `both` diagnostic rejects the proposed small
branch-conditional repair: 29 anchors replay-confirmed, selected K had median
`6`, 18/29 saturated the tested K=6 ceiling, and the deployable direction-only
selector reached only `0.341` balanced accuracy. Oracle branch assignment fit
responses well, but branch complexity and pre-transition selection failed the
preregistered gates. No branch model or downstream policy training was started.

The binary-gate diagnostic also failed its preregistered deployment floor. X3
plus the two-layer MLP reached `0.686` balanced accuracy on the source grouped
test but only `0.527` (95% trajectory-group bootstrap CI `[0.450, 0.602]`) on
the replay-audited frozen test, below the `0.70` soft-gate floor. Train and
validation trajectories had zero overlap with the audited test. No MoE or
expert training was started.

Older route drafts, meeting briefs, and Step 1--5 progress logs were removed
after their valid conclusions were consolidated into the documents above. Git
history remains the archive for those superseded records.
