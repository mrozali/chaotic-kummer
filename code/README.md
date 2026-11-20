# Code

Three folders:

- `code/kummer/` is the pipeline: 39 Python modules that build the free-field spaces of the Kummer model, the first-order matrix $M_{\rm c}$ of an exactly marginal deformation, its symmetry resolution, and the statistics of the Letter.
- `code/figure-reproduction/` holds one script per figure and the check scripts. Its `README.md` maps every figure, table and quoted number of the Letter to a command. Start there to reproduce or check a result.
- `code/cluster/` holds the Slurm job scripts of the computations that need a cluster, as records of how they were run.

The interpreter and the four packages are described in `environment/README.md`. The stored results are described in `data/README.md`. Costs marked (cluster) are those of the original cluster runs. The others are for the tested 14-core workstation.

## How the pipeline is organized

The modules are the authors' working code. Their names and the stage labels in their docstrings ("Route 2", "Phase A", "Gate 19") follow the history of the project, and each module documents its method, conventions and usage in its docstring. Modules import each other by name, so run them from `code/kummer/` or with that folder on the path, as the check scripts do.

**Records.** A pipeline module reads and writes its records in one folder. By default this is `data/r2_out/`. The environment variable `KUMMER_OUT` names another folder. A production run with the default would overwrite the shipped record of the same name, so name a scratch folder for each run of your own, for example `KUMMER_OUT=/tmp/kummer_run .venv/bin/python code/kummer/r2g_wall_scan.py 4`. Do not export the variable in your shell: the figure and check scripts read the stored records from the same folder. The check scripts run the modules in their own sandboxes. `r2c_group.py` caches the group tables of the point group in `code/kummer/_cache/` on first use (2 s), which is not committed.

**Vocabulary of the code.** The level $E$ is `th`, and `th_max` is the highest level of a run. A direction is named `qgen` (the reference direction of the Letter, called "direction 1" in some messages), `qgen2` (a second random direction, not used in the Letter), `qrand<seed>` or `singlet`. `r2g_generic_probe.cmat` defines them. The mirrors of the Letter are "walls". `Gtilde` is the double cover of the point group, of order 384. `H_c` is the stabilizer of a direction, the Letter's ${\cal G}_{\rm c}$. `nu` and `nt` are the numbers of untwisted and twisted states of a sub-block. A "leaf" is a fully resolved sub-block. The hidden $\mathbb Z_2$ is called `S` along the singlet. `kappa` is the normalization $\sqrt2\,\pi$ of `eq:Mc-chiral`.

### Free-field construction

| Module | Content |
|---|---|
| `r2_algebra.py` | chiral mode algebra and Fock spaces of the two-copy theory, untwisted and twisted |
| `r2_twist.py`, `r2_mom.py` | the $\mathbb Z_2$ twist map on oscillator states and on momentum states, from the covering-space formulas of Avery, Chowdhury and Mathur |
| `r2_rel.py` | the relative theory: Virasoro modes, internal $SU(2)$ rotations, projectors on massive ${\cal N}=4$ primaries |
| `r2_lift.py`, `r2_d0.py` | the chiral twist maps ${\cal A}_{\dot A}$ and the first-order deformation operator |
| `r2c_chiral.py`, `r2c_space.py` | chiral spaces by level, the spin-zero state space, and the matrix of the deformation between levels |
| `f1mom_group.py`, `r2c_group.py`, `r2c_irreps.py`, `r2c_proj.py` | the point group $\widetilde G$, its character table, its action on the Fock spaces, and isotypic projectors |

### First-order spectra

| Module | Content | Cost |
|---|---|---|
| `r2g_generic_probe.py` | first-order matrix of a direction, its stabilizer and symmetry resolution, dense. Defines the directions. | $E\le5$, 9.5 GB at $E=5$ |
| `r2g_generic_probe_L6.py` | the same at $E=6$ with sparse group matrices | |
| `r2g_generic_probe_mf.py` | matrix-free version | $E=7$: 2 h (cluster). $E=8$: 34 h, 233 GB (cluster) |
| `r2g_generic_probe_mf_lean.py` | matrix-free version that projects only the untwisted side, one character at a time | $E=7$: 10 minutes, 9.4 GB. $E=8$: 44 GB per character |
| `r2g_wall_scan.py` | the lean probe with the direction dependence pulled out: one basis per sub-block, then many directions | $E=3,4,5$: seconds. $E=6$: 4.5 minutes, 3.5 GB |
| `r2g_wall_scan_targets.py`, `r2g_wall_scan_ref.py` | directions placed near a mirror, and random-matrix reference curves near a mirror | |

### Hidden $\mathbb Z_2$

| Module | Content | Cost |
|---|---|---|
| `r2g_hidden_firstorder.py` | eigenspaces from the first-order matrices of one level at two directions (overlap graph of singular vectors), then the spectra inside them for many directions | $E=4$: seconds. $E=6$: 1 minute, 3.8 GB. $E=8$: 4.3 h on 32 cores, 78 GB |
| `r2g_generic_sweep.py` | truncated Hamiltonian $H_0+\lambda{\cal D}_{{\rm c},0}$ on the primaries through `th_max`, resolved by the stabilizer | `th_max` = 6: workstation. 7: 256 GB |
| `r2c_gate19_lib.py` | the search for jointly invariant subspaces: eigenvectors at $\lambda=0.2$ and $0.35$, overlap graph, connected components (`fully_resolve`) | |
| `r2g_first_order_from_sweep.py` | first-order spectra level by level inside the leaves found by `fully_resolve` | `th_max` = 7: 10.9 h (cluster) |
| `r2g_hidden_lean.py` | the same search restricted to the one sub-block that carries the hidden $\mathbb Z_2$, without the dense deformation matrix | 6 minutes per direction on 4 threads, 5.4 GB |
| `r2c_build.py`, `r2c_sym.py`, `r2c_stats.py`, `r2c_gate10.py`, `r2c_gate11.py`, `r2c_gate12_run.py` | assembly of the resolved truncated Hamiltonian, its further $\mathbb Z_2$ symmetries, and statistics. Imported by the modules above and by the singlet run. | |

### Singlet direction

| Module | Content | Cost |
|---|---|---|
| `r2c_fir_run.py` | sub-sector matrices of the truncated Hamiltonian along the singlet through `th_max` = 8 | 250 GB requested |
| `r2c_singlet_firstorder_L8.py` | first-order spectra of the singlet at $E\le8$ from those matrices, with and without the hidden $\mathbb Z_2$ | 400 s |

### References and controls

| Module | Content |
|---|---|
| `r2g_firstorder_null_n3000.py` | chiral GOE reference of the gap ratio, 3000 samples per level, bidiagonal model. The column $\sigma$ of the Letter's table. |
| `r2g_firstorder_null_n300.py`, `r2g_firstorder_null_E78.py` | the earlier 300-sample reference, partly from dense matrices. Its record supplies the matrix shapes. |
| `r2c_singlet_firstorder_null_v3.py` | chiral GOE reference for the singlet's sub-blocks |
| `r2g_E4_deficit.py` | the analysis of the level $E=4$ over 3000 directions |
| `r2g_unfolding_control.py` | control of the unfolding rule |
| `f2_common.py` | gap ratios and other random-matrix helpers |

## Which module made which result

`data/README.md` lists, for every stored record, the command that made it. The chain for the reference direction is:

1. $E\le7$: `r2g_generic_sweep.py qgen 7`, then `r2g_first_order_from_sweep.py qgen 7`. This is the truncated-Hamiltonian route, which also finds the hidden $\mathbb Z_2$ at $E=2,4,6$.
2. $E=7$ and $E=8$: `r2g_generic_probe_mf.py qgen 7` and `qgen 8`.
3. Hidden $\mathbb Z_2$ at $E=8$: `r2g_hidden_firstorder.py 8`.

For the 20 random directions at $E=3,\dots,6$, Fig. 2 uses the truncated-Hamiltonian cluster runs (`r2g_generic_sweep.py`, `r2g_first_order_from_sweep.py`, records in `scan_L6/`) for `qrand101` to `qrand116`, and for `qrand117` to `qrand120` `r2g_wall_scan.py` with `r2g_hidden_lean.py` for the hidden $\mathbb Z_2$ at $E=4,6$. The second route was run for all 20 directions and agrees with the first. At $E=7,8$ all 20 directions come from `r2g_generic_probe_mf_lean.py`, and at $E=8$ the hidden $\mathbb Z_2$ from `r2g_hidden_firstorder.py 8`.

## Platform

The modules were run on macOS and Linux. `r2g_hidden_lean.py` uses `fcntl`, and `r2g_firstorder_null_n300.py` and `r2g_firstorder_null_n3000.py` use the `fork` start method, so these three do not run on Windows. `verify.sh` needs a POSIX shell.
