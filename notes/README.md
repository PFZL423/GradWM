# Current Research Notes

Updated: 2026-07-10

The current A5 contact-VJP record consists of four documents:

1. `a5_vjp_progress/2026-07-10_a5_action_side_first_roadmap.md`
   records the accepted route, gates, and downstream order.
2. `a5_vjp_progress/2026-07-10_a5_action_vjp_v2_final.md`
   is the authoritative action-side result and bottleneck summary.
3. `a5_vjp_progress/2026-07-10_action_vjp_v2_source_diagnosis.md`
   records the Genesis reset/restore and data-generation root causes.
4. `a5_vjp_progress/2026-07-10_marginal_probe.md`
   tests the rejected cohort and records why a single local matrix is not a
   reliable target for cross-epsilon/mode-mixed contacts.

Current status: the stable-contact action-side gate passed with `112 / 113`
held-out descent and median oracle-gradient cosine `0.988`. State-side VJP,
SHAC/BPTT, and Dreamer comparisons have not started. The remaining action-side
limitations are `32.9%` stable-contact coverage, magnitude calibration, and no
binary-descent advantage over the global matrix on the current local task.

The frozen 96-anchor marginal probe found that `signature_only` labels remain
coherent, but `cross_only` and `both` do not admit a reliable direction-
independent local matrix at the tested finite scales. The contact-feature Phase
2 was therefore not started; the next action-side method question is a
direction- or branch-conditioned target for those cohorts.

Older route drafts, meeting briefs, and Step 1--5 progress logs were removed
after their valid conclusions were consolidated into the documents above. Git
history remains the archive for those superseded records.
