# Cluster job scripts

These are the Slurm scripts of the computations that do not fit on a workstation, as they were run on the Fir and Nibi clusters of the Digital Research Alliance of Canada between September and November 2026. They are records of the runs and templates for repeating them. Site-specific values (account, virtual environment, paths) are placeholders in angle brackets. Each script changes to `code/kummer/` and sets `KUMMER_OUT`, the folder where the modules read and write their records.

| Script | Command | Requested | Produced |
|---|---|---|---|
| `sweep_L7_reference.sh` | `r2g_generic_sweep.py qgen 7`, then `r2g_first_order_from_sweep.py qgen 7` | 32 cores, 256 GB, 36 h | truncated Hamiltonian of the reference direction through $E=7$ (matrices, not in this repository) |
| `firstorder_L7_reference.sh` | `r2g_first_order_from_sweep.py qgen 7` | 32 cores, 96 GB, 30 h | `Generic_firstorder_qgen_L7.npz`, `.json` (10.9 h) |
| `probe_E8_reference.sh` | `r2g_generic_probe_mf.py qgen 8` | 48 cores, 500 GB, 2 days | `Generic_probe_qgen_E8_mf_values.npz`, `.json` (34 h, 233 GB used) |
| `lean_E8_reference_validation.sh` | `r2g_generic_probe_mf_lean.py qgen 8 --irreps=2` | 32 cores, 128 GB, 12 h | `Generic_probe_qgen_E8_lean_rho2_values.npz`, `.json` |
| `hidden_firstorder_E8.sh` | `r2g_hidden_firstorder.py 6` as a gate, then `r2g_hidden_firstorder.py 8 --save-gram` | 32 cores, 160 GB, 8 h | `Hidden_firstorder_E8_values.npz`, `.json`, and the two logs (4.3 h, 78 GB used) |
| `scan_L6_random_directions.sh` | `r2g_generic_sweep.py qrand<seed> 6`, then `r2g_first_order_from_sweep.py qrand<seed> 6`, seeds 101 to 116 | 8 cores, 24 GB, 6 h per direction | `scan_L6/Generic_firstorder_qrand<seed>_L6.json` |
| `lean_E7_random_directions_fir.sh`, `lean_E7_random_directions_nibi.sh` | `r2g_generic_probe_mf_lean.py qrand<seed> 7` | 16 cores, 48 GB per direction | `Generic_probe_qrand<seed>_E7_lean_values.npz`, `.json` |
| `lean_E8_random_directions_fir.sh`, `lean_E8_random_directions_nibi.sh` | `r2g_generic_probe_mf_lean.py qrand<seed> 8 --irreps=<r>`, one task per direction and character | 32 cores, 64 GB per task | the per-character pieces of `Generic_probe_qrand<seed>_E8_lean_values.npz`, merged afterwards with `--merge` |
| `singlet_matrices_L8.sh` | `r2c_fir_run.py 8 prim` | 16 cores, 250 GB, 16 h | the singlet's matrices through $E=8$ (not in this repository) |
| `singlet_firstorder_L8.sh` | `r2c_singlet_firstorder_L8.py` | 16 cores, 48 GB, 6 h | `Singlet_firstorder_L8.npz`, `.json` (400 s) |

Three random directions (`qrand106`, `qrand111`, `qrand112`) were run on Fir and the other seventeen on Nibi. The Nibi script for $E=7$ also reran `qrand106` and compared it with the Fir values.

The reference direction at $E=7$ (`Generic_probe_qgen_E7_mf_values.npz`) was computed with `r2g_generic_probe_mf.py qgen 7` in 2.0 h. Its job script is not included. `code/figure-reproduction/check_lean_probe_E7.py` recomputes this record on a workstation.
