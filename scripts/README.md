# Scripts Layout

This directory keeps only current entry points at the top level. Most earlier
one-off Genesis diagnostics have been moved into `legacy/`.

- `arx/`: ARX A5 / ACone asset import and A5 model diagnostics.
- `pusher/`: Pusher-like and box-push contact-gradient diagnostics.
- `archive/`: low-priority or off-track probes kept for reference.
- `legacy/`: older tracked experiments retained for reproducibility.
- `rope_solver_probe/`: rope solver experiments.

Top-level scripts:

- `arm_rope_contact_sanity.py`: current rope/contact sanity under active edits.
- `genesis_state_restore_check.py`: state-restore diagnostic with box-push case.
- `make_arm_mjcf.py`: shared handwritten arm MJCF generator.

Legacy scripts may have old command examples and import assumptions. Treat them
as archived experiment records unless they are explicitly revived.
