# Legacy Scripts

This folder contains older tracked Genesis experiments moved out of the root
`scripts/` directory to keep the active workspace readable.

Useful historical probes:

- `two_box_same_mjcf_contact_grad.py`: two-box contact-gradient probe.
- `traj_opt_box_push.py`: primitive rigid box-push trajectory optimization.
- `grad_fd_check.py`: early FD-vs-analytic check for rope/grasp scenes.
- `rope_no_rope_grad_compare.py`: rope/no-rope gradient comparison.
- `grasp_scene.py`: older handwritten arm + rigid cable scene.
- `traj_opt_grasp.py`: older grasp trajectory optimization.

Some command examples inside these files still use their old root-level paths.
If reviving a script, first check imports and output paths.
