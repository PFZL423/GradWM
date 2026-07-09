# ARX Scripts

Current ARX/A5 entry points:

- `arx_model_load_sanity.py`: load A5 and ACone assets in Genesis and record joint metadata.
- `arx_a5_diagnostics.py`: A5 import, no-contact backward, far-box gradient, and rough push-forward checks.
- `a5_ee_trajectory_scan.py`: FK-based search for A5 poses/actions whose end-effector proxy sweeps horizontally.
- `a5_pusher_scene_sweep.py`: subprocess-based sweep over object placement / speed / horizon.
- `a5_pusher_forward_sanity.py`: forward-only A5 Pusher-like sanity at the current horizontal-sweep candidate.
- `a5_contact_fd_diag.py`: FD-vs-autograd diagnostic at the current A5 contact anchor.
- `a5_fd_response_dataset.py`: collect FD response labels for action-side VJP supervision; now records both object position and object qvel response.
- `a5_restore_response_dataset.py`: faster same-process restored-anchor sampler for local response labels.
- `a5_fit_linear_response.py`: least-squares sanity fit of a single-anchor response matrix.
- `a5_multi_anchor_fd_dataset.py`: collect/aggregate finite-scale labels across nearby anchors; supports `--sampler slow` and `--sampler restore`.
- `a5_train_response_mlp.py`: train/evaluate a tiny `A_theta(z) v` matrix-head MLP.
- `a5_response_fit_summary.py`: compare global vs per-anchor linear response fits.

Main outputs live in:

- `analysis/2026-07-09_arx_pusher/`

The current default A5 anchor is:

- `qpos = [0.0, 1.4, -0.4, 0.5, 0.0, 0.0]`
- `qvel = [1.6, 0.0, 0.0, 0.0, 0.0, 0.0]`
- object position `(0.306, 0.076, 0.120)`

It is now useful for reproducing the object-loss-to-action gradient gap. The
first FD response dataset is available, but the single-anchor linear fit is
noisy. The next step is to turn this into a small multi-anchor dataset with
finite-scale bins.

Current potential check:

- A 9-anchor position-only dataset has been collected.
- Per-anchor linear fits are much better than one global matrix, so context
  matters.
- The first tiny matrix-head MLP does not yet beat the global baseline, so
  training design/data coverage still needs work before policy optimization.
- A newer state-response dataset shows that local contact signal is mostly in
  object qvel rather than immediate object position.
- With `local_state_response` labels, the first positive MLP signal appears on
  row split (`0.94 -> 0.67` relative RMSE on the 4-anchor set; `5.80 -> 1.13`
  after target-norm filtering on the 9-anchor set), but leave-one-anchor
  generalization is still not stable.
- The restored-anchor sampler is now the preferred data path. A 4-anchor
  restored dataset kept `36/36` rows with estimated step-reuse speedup around
  `13x`; row split improved from `0.947` to `0.371`. A 9-anchor restored
  dataset kept `97` rows; with target-norm `<=50`, row split improved from
  `1.289` to `0.287`.
- Repeat queries from the same restored anchor were exactly repeatable in the
  tested runs (`repeat diff = 0`). Some anchors still show online-vs-restored
  nominal mismatch, so large-scale data should include anchor-quality filtering
  and contact-regime bucketing.
- Anchor-quality gates are now first-class sampler options:
  `--max-restore-local-state-diff`, `--max-repeat-local-state-diff`,
  `--min-horizontal-disp`, `--max-abs-vertical-disp`, and
  `--drop-bad-anchor`.
- Large aggregation can use `--omit-rows-json` to keep the JSON as a manifest
  instead of duplicating all rows already stored in CSV/shards.
- The first clean-gated 9-anchor run kept `8/9` anchors and `59` rows. It
  rejected the known bad anchor with
  `restore_local_state_diff|vertical_disp_too_large`; row split improved from
  `1.215` to `0.257`.
