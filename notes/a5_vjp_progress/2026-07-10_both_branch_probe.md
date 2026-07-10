# A5 Both-Cohort Branch Diagnostic

## Scope

This is a diagnostic-only audit. It does not train a branch-conditional model and does not modify the frozen v2 or marginal-probe pipelines.
The source non-weak `both` population is 481/715 marginal candidates (67.3%) and 481/1200 of the frozen contact scan (40.1%). After reserving the old 32 `both` diagnostic anchors, 449 remain eligible here.

The dense label uses one simulator step after the anchor (`response_steps=0` in the frozen worker convention). Each central probe is decomposed into nominal-to-plus and nominal-to-minus one-sided responses. Branches are fitted as held-out piecewise-linear maps `r = A_k v`, not as clusters of response vectors.
The 200 random directions are frozen as `120` for fitting candidate K values, `40` for selecting K, and `40` for final testing; the six axis directions are final test only. After K selection, the first `160` directions are refitted before final evaluation.

## Commands

```bash
conda run -n genesis --no-capture-output python scripts/arx/a5_both_branch_probe.py --stage select
conda run -n genesis --no-capture-output python scripts/arx/a5_both_branch_probe.py --stage collect --gpu-ids 0,1,2,3 --workers 4
conda run -n genesis --no-capture-output python scripts/arx/a5_both_branch_probe.py --stage analyze
```

## Frozen Selection

- eligible non-weak both anchors after excluding the old frozen probe: `449`;
- selected: `30`; old-probe overlap: `0`;
- x/y/speed/phase unique values: `3/17/9/20`.

## Sampling

- successful anchors: `30/30`;
- kept rows: `7560/7560`;
- pristine query replays: `15180`; wall clock: `5053.6s`;
- replay-confirmed both: `29/30`.
- wall clock was measured while unrelated pre-existing training occupied all four GPUs, so it is an execution record rather than a clean throughput benchmark.

The source both label came from restore prefiltering. Non-confirmed anchors are retained and never replaced, avoiding survivor bias.

## Branch Discovery

The decision metrics below use the `29` replay-confirmed both anchors; all-selected metrics remain in the JSON summary.

| K | Anchors | Fraction |
|---:|---:|---:|
| 1 | 0 | 0.000 |
| 2 | 0 | 0.000 |
| 3 | 1 | 0.034 |
| 4 | 4 | 0.138 |
| 5 | 6 | 0.207 |
| 6 | 18 | 0.621 |

- selected K mean/median: `5.414/6.000`;
- alternate-seed K mean/median: `5.621/6.000`; exact per-anchor K agreement: `0.414`;
- held-out response cosine: median `0.999`, Q25/Q75 `0.996/1.000`;
- branch-level held-out cosine: median `1.000`, Q25/Q75 `0.996/1.000`;
- branch/contact-mode purity association: median `0.700` (post-transition diagnostic, not a selector);
- epsilon=0.01 transfer cosine: median `0.998`;
- cosine gain over K=1: median `0.448`; relative-RMSE reduction: median `0.525`.

## Selector Diagnostic

Only direction is deployable before observing the perturbed transition. Post-contact variants are reported as oracle association diagnostics and are not treated as deployable selectors.

| Variant | Split | Balanced accuracy | Raw accuracy | 1/K chance |
|---|---|---:|---:|---:|
| direction_deployable | hold_eps003 | 0.341 | 0.380 | 0.167 |
| direction_deployable | hold_eps010 | 0.139 | 0.141 | 0.167 |
| post_contact_oracle | hold_eps003 | 0.252 | 0.304 | 0.167 |
| post_contact_oracle | hold_eps010 | 0.206 | 0.337 | 0.167 |
| direction_plus_post_contact_oracle | hold_eps003 | 0.443 | 0.500 | 0.167 |
| direction_plus_post_contact_oracle | hold_eps010 | 0.180 | 0.261 | 0.167 |

## Interpretation

- `18/29` confirmed anchors select the maximum tested K=6. Together with only `0.414` exact-K seed agreement, the experiment does not identify an exact physical branch count; it shows that the proposed K<=3 representation is too small and that complexity frequently saturates the tested ceiling.
- Oracle branch assignment yields held-out cosine `1.000` and transfers to epsilon=0.01 at `0.998`. Thus several local linear maps can describe the responses after branch membership is known.
- The deployable direction-only selector reaches balanced accuracy `0.341` versus median 1/K chance `0.167`. It is above chance but below both the 0.60 hard floor and the 0.80 success threshold.
- Post-transition contact features are outcome-side oracle diagnostics. Their scores cannot be used as a deployable selector claim, and even the direction-plus-contact oracle remains below the success threshold.

## Decision Matrix

| Criterion | Threshold | Result | Pass |
|---|---:|---:|---|
| pristine replay both confirmation | >= 20 anchors | 29 | **True** |
| median K | <= 3 | 6.000 | **False** |
| held-out branch response cosine | >= 0.90 | 1.000 | **True** |
| K>1 benefit over K=1 | cosine +0.15 or RMSE -30% | `0.448` / `0.525` | **True** |
| epsilon=0.01 transfer cosine | >= 0.85 | 0.998 | **True** |
| deployable direction selector balanced accuracy | >= 0.80 | 0.341 | **False** |

**Direct verdict: `not_supported_for_branch_training`.**

The few-branch hypothesis and deployable-selector floor fail: selected K saturates the tested upper range while direction-only branch prediction remains below 0.60. The preregistered result does not support starting branch-conditional training.

No downstream training was started.
