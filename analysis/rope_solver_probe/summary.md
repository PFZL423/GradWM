# Rope Solver Matrix Summary

| solver | n_particles | fwd_fps | bwd_s | grad_norm | grad_nan | sag_mm | peak_mem_mb | grad_status | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PBD-Cloth | 244 | 116.29 | 0.7 | 0 | 100.0% | 35.09 | 711 | no_contact_path | blocked |
| FEM-Elastic | 1059 | 314.67 | nan | nan | 100.0% | nan | 443 | error:AttributeError("'FEMEntity' object has no attribute 'collect_output_grads'") | blocked |
| MPM-Elastic | 99 | 145.81 | 21.29 | 0.3789 | 0.0% | 2.4 | 1689 | ok | conditional |

PBD-Cloth: backward 100.0% non-finite gradient, 0.0% finger.grad finite, fps 116.29, sag 35.09 mm, status no_contact_path - blocked.

FEM-Elastic: backward 100.0% non-finite gradient, 0.0% finger.grad finite, fps 314.67, sag nan mm, status error:AttributeError("'FEMEntity' object has no attribute 'collect_output_grads'") - blocked.

MPM-Elastic: backward NaN-free, 100.0% finger.grad finite, fps 145.81, sag 2.4 mm, status ok - conditional.
