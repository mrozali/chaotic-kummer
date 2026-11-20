#!/usr/bin/env python3
"""The chiral GOE reference of the gap ratio: the column sigma of the Letter's table and the gray bands of Fig. 2.

The record data/r2_out/Generic_firstorder_null_n3000.json holds, for each level E = 3..8, the pooled mean gap ratio of
3000 chiral GOE realizations.  One realization is one real Gaussian matrix per fully resolved sub-block, with as many
rows as the sub-block has twisted states and as many columns as it has distinct non-zero singular values.  The record
was made by code/kummer/r2g_firstorder_null_n3000.py, which draws the singular values from the bidiagonal model of
Dumitriu and Edelman.  Sample k of level E uses numpy's default_rng([20261116, E, k]).

This script checks three things.
  1. The standard deviation of the stored samples at each level, which is the sigma of the table.
  2. The stored samples follow from the seeds: the first samples of each level are regenerated with the module's own
     functions and compared with the record.
  3. The bidiagonal model agrees with dense Gaussian matrices: at E = 3 and 4, where dense matrices are cheap, 3000
     dense realizations give the same standard deviation within the sampling error.

Run from the repository root:  .venv/bin/python code/figure-reproduction/check_null_sigma.py      (about half a minute)
"""
import os
import sys

import numpy as np

from checklib import J, KUMMER, R_GOE, Report, gap_ratios

sys.path.insert(0, KUMMER)
import r2g_firstorder_null_n3000 as null          # noqa: E402

rep = Report("Chiral GOE reference of the gap ratio")
rec = J("Generic_firstorder_null_n3000.json")
N_REGEN = {3: 200, 4: 200, 5: 200, 6: 100, 7: 20, 8: 6}

rep.section("1. Standard deviation and mean of the stored samples")
for E in range(3, 9):
    L = rec["levels"][str(E)]
    v = np.array(L["samples"])
    sd = v.std(ddof=1)
    rep.check("E = %d: %d samples, sigma = %.5f +- %.5f" % (E, v.size, sd, sd / np.sqrt(2 * (v.size - 1))),
              abs(sd - L["sd"]) < 1e-12 and v.size == 3000)
    rep.check("E = %d: mean %.5f within a tenth of sigma of the GOE value 0.5307" % (E, v.mean()), abs(v.mean() - R_GOE) < 0.1 * sd,
              "%+.2f sigma" % ((v.mean() - R_GOE) / sd))

rep.section("2. The stored samples regenerated from their seeds")
for E in range(3, 9):
    L = rec["levels"][str(E)]
    shapes = [tuple(s) for s in L["shapes"]]
    fresh = np.array([null.sample(("main", E, k, shapes))[3] for k in range(N_REGEN[E])])
    dev = float(np.max(np.abs(fresh - np.array(L["samples"][:N_REGEN[E]]))))
    rep.check("E = %d: first %d samples, shapes %s" % (E, N_REGEN[E], sorted(set(shapes))), dev < 1e-9, "largest difference %.1e" % dev)

rep.section("3. Bidiagonal model against dense Gaussian matrices")
for E in (3, 4):
    L = rec["levels"][str(E)]
    shapes = [tuple(s) for s in L["shapes"]]
    rng = np.random.default_rng([2026, E])
    dense = []
    for _ in range(3000):
        dense.append(np.concatenate([gap_ratios(np.linalg.svd(rng.standard_normal(s), compute_uv=False)) for s in shapes]).mean())
    ratio = np.std(dense, ddof=1) / L["sd"]
    rep.check("E = %d: sigma of 3000 dense realizations over sigma of the record" % E, abs(ratio - 1) < 0.06, "%.3f +- 0.018" % ratio)

rep.finish()
