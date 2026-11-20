# Data

All data of this repository are in `data/r2_out/`: 177 files, 24 MB in total. Nothing has to be downloaded. The folder name is the authors' name for the output folder of the pipeline. It is kept because the modules of `code/kummer/` and their docstrings refer to records as `r2_out/<file>`.

The records hold first-order spectra: the non-zero singular values $\nu_i$ of the first-order matrix $M_{\rm c}$ of the Letter (eq. `eq:Mc`), separately for every symmetry-resolved sub-block. A level of the undeformed theory at energy $E$ then splits into $E\pm\lambda\nu_i$. The figures and the table of the Letter, and its numerical statements, are computed from these records by the scripts of `code/figure-reproduction/`. Figs. S1 and S2 need no data, and the right panel of Fig. 1 is computed afresh from the free-field code.

## Conventions

**Arrays.** Each `.npz` file holds one-dimensional float64 arrays, one per sub-block, except for the tables of the singlet record described below. Unless stated otherwise an array holds the distinct non-zero singular values, rounded to 9 decimals, in increasing order. Each `.npz` file has a `.json` companion with the dimensions, ranks and gap ratios of the same sub-blocks, and the run time.

**Levels.** `E<n>` in a file name or key is the level $E=n$. The code calls the level `th`. `L<n>` in a file name means that the run covered all levels $E\le n$ (the code's `th_max`).

**Directions.** A direction of deformation is a unit quaternion $q$ (eq. `eq:quaternion`).

| Name | Direction |
|---|---|
| `qgen` | The reference direction $q_{\rm ref}$ of the Letter (eq. `eq:qgen`): the normalized vector `numpy.random.default_rng(2026).standard_normal(4)`. |
| `qrand<seed>` | A random direction: the same recipe with `default_rng(seed)`. The 20 random directions of Fig. 2 are `qrand101` to `qrand120`. The direction scans also contain `qrand121` to `qrand148`. |
| `singlet` | The singlet direction $q=(0,0,-1,0)$. |
| `near<nn>_coord`, `near<nn>_diag` | The 20 directions of Fig. 2 (right) placed at chosen angles from a mirror $q_\mu=0$ (`coord`) or $q_\mu=\pm q_\nu$ (`diag`). |
| `qgen2` | A second random direction, from `default_rng(99)`. It is $1.55^\circ$ from a mirror. The Letter does not use it. It appears in several multi-direction records as an extra check. |

**Sub-blocks of a generic direction.** A key `rho<r>_lab<f><z>` names a sub-block. `r` is the code's index of a character of the stabilizer ${\cal G}_{\rm c}$ on the scalar primaries. The four characters that occur are `rho0`, `rho2`, `rho3` and `rho6`, and `rho0` is the trivial one. The two letters after `lab` are the signs of two right-moving involutions, `p` for $+$ and `m` for $-$. The second sign is $Z_R$ of the Letter. The first belongs to an involution that the code calls `F_R`, which is $+$ on every stored sub-block. The sub-blocks that mix at first order have $Z_R=(-1)^E$: they are `labpm` at odd $E$ and `labpp` at even $E$. A key `sv_rho...` holds all singular values of the sub-block, zeros included and unrounded.

**Hidden $\mathbb Z_2$.** At the even levels the sub-block `rho0_labpp` splits further into the two eigenspaces of the hidden $\mathbb Z_2$ (End Matter). They are stored in two ways, depending on the method that found them. The two-direction records use `<direction>__leaf0`, `<direction>__leaf1` and, for their union, `<direction>__unresolved`. The truncated-Hamiltonian records use `E<n>__b1+` and `E<n>__b1-`.

**Leaves of the reference direction.** `Generic_firstorder_qgen_L6.npz` and `..._L7.npz` use keys `E<n>_leaf<k>`, where `k` indexes the list `leaf_tags` of the companion `.json`. Leaves 0, 3, 5, 7 are the four characters with $Z_R=-1$ and carry the odd levels. Leaves 4, 6, 8 are the three non-trivial characters with $Z_R=+1$, and leaves 1 and 2 are the two eigenspaces of the hidden $\mathbb Z_2$ in the trivial character. Together these five carry the even levels.

**Singlet.** `Singlet_firstorder_L8.npz` uses keys `A_k<n>_E<E>` and `B_k<n>_E<E>`, `B_k<n>p_E<E>`, `B_k<n>m_E<E>`. Here `k` numbers the sub-sectors of the singlet's symmetry group. Resolution `A` leaves the hidden $\mathbb Z_2$ unresolved. Resolution `B` resolves it, with `p` and `m` its two eigenspaces. These arrays hold all singular values, with zeros and repeated values. The scripts reduce them to distinct non-zero values. `A_meta` and `B_meta` are integer tables with one row per sub-block and level. Counting columns from 0, they hold the sub-sector index `k` (column 0), the character of the double cover of the point group (1), the signs of the two further involutions that the code resolves along the singlet (2, 3), the level (4), the numbers of untwisted and twisted states (5, 6), the rank (7) and the number of distinct values (8). `B_meta` has three more columns: `k` again (9), the index of the sub-block's name in the string array `B_tags` (10), and the sign of the hidden $\mathbb Z_2$, $+1$, $-1$, or 0 where the sub-block is not split (11).

**Mirrors.** The code and the records call the 16 mirrors of the Letter "walls". In the direction scans, `delta` is the angle in degrees to the nearest mirror and `nearest` names it.

## Records

| Files | Content | Made by | Used for |
|---|---|---|---|
| `Generic_firstorder_qgen_L7.npz`, `.json` | Reference direction, $E=1,\dots,7$, every sub-block fully resolved (hidden $\mathbb Z_2$ included). The `.json` holds $d_\pm$ and the ranks. | `r2g_generic_sweep.py qgen 7`, then `r2g_first_order_from_sweep.py qgen 7` (cluster) | Table ($E\le7$), Figs. 1, 3, 4 |
| `Generic_firstorder_qgen_L6.npz`, `.json` | The same from the run through $E=6$. The values agree with the `L7` record. | the same modules with `6` (cluster) | Fig. 2 ($E\le6$), shapes of the chiral GOE matrices |
| `Generic_probe_qgen_E7_mf_values.npz`, `.json` | Reference direction, $E=7$, four sub-blocks of 3536 values. | `r2g_generic_probe_mf.py qgen 7` (cluster) | Table and Fig. 2 at $E=7$ |
| `Generic_probe_qgen_E8_mf_values.npz`, `.json` | Reference direction, $E=8$: the three non-trivial characters (9208 values each) and the trivial character with the hidden $\mathbb Z_2$ unresolved (9216 values). | `r2g_generic_probe_mf.py qgen 8` (cluster) | Table, Figs. 2 and 3 at $E=8$ |
| `Generic_probe_qgen_E8_lean_rho2_values.npz`, `.json` | One $E=8$ sub-block of the reference direction recomputed with the lean probe. It equals the record above. | `r2g_generic_probe_mf_lean.py qgen 8 --irreps=2` (cluster) | validation of the lean probe |
| `Hidden_firstorder_E8_values.npz`, `.json` | $E=8$, trivial character: the two eigenspaces of the hidden $\mathbb Z_2$ (6137 and 3079 values) and their union, for 22 directions (`qgen`, `qgen2`, `qrand101` to `qrand120`). The `.json` holds the gates of the computation. | `r2g_hidden_firstorder.py 8` (cluster) | Table, Figs. 2 and 3 at $E=8$, End Matter |
| `Hidden_firstorder_E6_values.npz`, `.json` | The same at $E=6$ (899 and 451 values), 22 directions. | `r2g_hidden_firstorder.py 6` | End Matter: the two methods agree |
| `Hidden_firstorder_E4_values.npz`, `.json` | The same at $E=4$ (50 and 26 values), 6 directions. | `r2g_hidden_firstorder.py 4 --dirs=...` | End Matter: the two methods agree |
| `fir_hidden_firstorder_E8.log`, `fir_hidden_firstorder_E6.log` | Logs of the cluster job that made the $E=8$ record, with its $E=6$ gate. | `code/cluster/hidden_firstorder_E8.sh` | provenance |
| `Wall_scan_E3_values.npz` to `Wall_scan_E6_values.npz`, `.json` | Direction scans: 50 directions (`qgen`, `qgen2`, `qrand101` to `qrand148`) at $E=3,4,5,6$. All four characters at odd $E$, the three non-trivial characters at even $E$. The `.json` holds $q$ and the angles to the 16 mirrors. | `r2g_wall_scan.py <E>` | Fig. 2, random directions at $E=3,\dots,6$ |
| `Wall_scan_targets20.json`, `...20b.json`, `...20c.json` | The 20 placed directions of Fig. 2 (right). The set `20c` is the one in the figure. It follows from `20` by two documented replacements, recorded in the files. | `r2g_wall_scan_targets.py` | Fig. 2 (right) |
| `Wall_scan_E5_targeted20c_values.npz`, `.json` | $E=5$ spectra of the 20 placed directions. | `r2g_wall_scan.py 5 --seeds= --extra= --qfile=... --tag=_targeted20c` | Fig. 2 (right) |
| `Wall_scan_ref.json` | Reference curves for a direction near a single mirror: chiral GOE blocks that split in two as the mirror is approached. | `r2g_wall_scan_ref.py` | Fig. 2 (right), limiting values |
| `Hidden_lean_qrand<seed>_L6_values.npz`, `.json`, seeds 101 to 120 | The two eigenspaces of the hidden $\mathbb Z_2$ at $E=2,4,6$ for each random direction, found from the truncated Hamiltonian through $E=6$. | `r2g_hidden_lean.py qrand<seed>` | Fig. 2 at $E=4,6$ |
| `scan_L6/Generic_firstorder_qrand<seed>_L6.json`, seeds 101 to 116 | Summaries of earlier cluster runs of the full truncated-Hamiltonian pipeline for 16 random directions: gap ratios and dimensions of the fully resolved sub-blocks at $E\le6$. No arrays. | `r2g_generic_sweep.py`, `r2g_first_order_from_sweep.py` (cluster) | Fig. 2 at $E=3,\dots,6$ for these 16 directions, and the reference that `r2g_hidden_lean.py` validates itself against |
| `Generic_probe_qrand<seed>_E7_lean_values.npz`, `.json`, seeds 101 to 120 | $E=7$, four sub-blocks of 3536 values, for each random direction. | `r2g_generic_probe_mf_lean.py qrand<seed> 7` (cluster) | Fig. 2 at $E=7$ |
| `Generic_probe_qrand<seed>_E8_lean_values.npz`, `.json`, seeds 101 to 120 | $E=8$, four sub-blocks, the trivial character unresolved. | `r2g_generic_probe_mf_lean.py qrand<seed> 8 --irreps=<r>`, one run per character, then `--merge` (cluster) | Fig. 2 at $E=8$, with the eigenspaces of `Hidden_firstorder_E8_values.npz` |
| `Singlet_firstorder_L8.npz`, `.json` | Singlet direction, $E=1,\dots,8$, in the resolutions `A` and `B`. | `r2c_fir_run.py 8 prim` (cluster), then `r2c_singlet_firstorder_L8.py` | Figs. 2 and 3, singlet numbers of the Letter |
| `Generic_probe_singlet_L5.json` | Singlet direction, $E\le5$, summary of the dense implementation. | `r2g_generic_probe.py singlet 5` | cross-check in Fig. 2 |
| `Generic_firstorder_null_n3000.json` | Chiral GOE reference for the generic directions: 3000 samples of the pooled $\langle r\rangle$ per level, $E=3,\dots,8$, with the matrix shapes and the seed. Also 1000 samples at $E=8$ with a different division of the twisted states (`rows_check`). | `r2g_firstorder_null_n3000.py` | column $\sigma$ of the table, gray bands of Fig. 2 |
| `Generic_firstorder_null_n300.json`, `Generic_firstorder_null_E78.json` | Earlier references with 300 samples per level, partly from dense matrices, and the dense samples at $E=7,8$ that entered them. Superseded for the Letter by the 3000-sample record. | `r2g_firstorder_null_n300.py`, `r2g_firstorder_null_E78.py` | matrix shapes for the record above, comparison of the dense and bidiagonal generators |
| `Singlet_firstorder_L8_null_v3.json` | Chiral GOE reference for the singlet's sub-blocks, $E=5,\dots,8$. | `r2c_singlet_firstorder_null_v3.py` | green bands of Fig. 2 |
| `E4_deficit.json` | The analysis of the level $E=4$: 3000 random directions, the top of the spectrum, the singlet limit, and the levels $E=3,5,6$ for comparison. | `r2g_E4_deficit.py --levels=3,5,6 --nother=150` | End Matter, sentences on $E=4$ |
| `Unfolding_control.json` | Number variance and staircase residuals for several unfolding rules, for the data and for chiral GOE spectra of the same shapes. | `r2g_unfolding_control.py` | Supplemental Material, unfolding paragraph |

The modules are in `code/kummer/`. `code/figure-reproduction/README.md` maps the figures and the quoted numbers to scripts, and gives the cost of each computation.

## Where the records come from

All checks of this repository run on a workstation, and so do the computations at $E\le6$ and, one direction at a time, at $E=7$. The larger computations were cluster jobs on Fir and Nibi, of the Digital Research Alliance of Canada, between September and November 2026. Their job scripts are in `code/cluster/`.

| Record | Resources of the original run |
|---|---|
| `Generic_firstorder_qgen_L7` | 32 cores and 256 GB requested. The sweep ran in a job with a 36 h limit, and the first-order step took 10.9 h in a second job. |
| `Generic_probe_qgen_E7_mf` | 2.0 h |
| `Generic_probe_qgen_E8_mf` | 34 h, 233 GB |
| lean probe at $E=7$, per direction | 12 to 53 minutes on 16 cores, about 8 GB |
| lean probe at $E=8$, per direction and character | 33 to 82 minutes on 32 cores (median 41), about 44 GB |
| `Hidden_firstorder_E8` | 4.3 h on 32 cores, 78 GB |
| `Singlet_firstorder_L8` | a job with 16 cores and 250 GB requested for the matrices, then 400 s |
| `scan_L6` | 8 cores and 24 GB requested per direction |
| `Wall_scan_E3` to `E6`, `Hidden_lean_*`, the chiral GOE references, `E4_deficit`, `Unfolding_control` | workstation, seconds to minutes each |

## Notes on the records

- **Text fields.** The `.json` files are the authors' records with two edits to text fields, made for this repository: a field `doc`, which held a copy of the generating module's docstring at the time of the run, was removed from seven files, and the name of a machine was removed from a note in `Singlet_firstorder_L8_null_v3.json`. No number was changed.
- **The $E=3$ scan** was made with the option `--direct-stab` of `r2g_wall_scan.py`, which finds the stabilizer by a direct numerical test and takes minutes. The default run takes seconds and gives the same spectra (`check_first_order_rebuild.py`).
- **`gmax`** in the scan and lean-probe summaries is a diagnostic that depends on the basis. It is not expected to reproduce between runs.
- **Sub-blocks with few values.** A pooled $\langle r\rangle$ stored in a summary leaves out sub-blocks with fewer than 20 distinct values. This matters only for the singlet at low levels.
- **Run-time notes inside the records** sometimes name scripts that are not part of this repository, and use the working names "direction 1" for the reference direction and "September" for the truncated-Hamiltonian cluster runs of September 2026.

## What is not included

- The large intermediates of the cluster runs: the matrices of the truncated Hamiltonian along the reference direction, the singlet's matrices through $E=8$, and the Gram matrices of the $E=8$ hidden-$\mathbb Z_2$ run (14 GB). The modules that write them are included.
- The per-character pieces of the $E=8$ lean runs. Their arrays are identical to the merged files that are included.
- The separate records of `qgen2`.
