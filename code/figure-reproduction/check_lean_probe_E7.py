#!/usr/bin/env python3
"""Rebuild the first-order spectrum at E = 7 for one direction with the lean matrix-free probe, and compare.

The script runs code/kummer/r2g_generic_probe_mf_lean.py <direction> 7 in a sandbox.  That module builds the space of
massive primaries at E = 7 (28288 untwisted and 28224 twisted states) and the singular values of the first-order
matrix in the four sub-blocks of the generic stabilizer with Z_R = -1, each with 3536 non-zero values.

For the reference direction (qgen, the default) the fresh spectra are compared with the record behind the Letter's
table, data/r2_out/Generic_probe_qgen_E7_mf_values.npz, which was made on a cluster by a different implementation
(r2g_generic_probe_mf.py).  For qrand101 to qrand120 they are compared with Generic_probe_<direction>_E7_lean_values.npz.

Run from the repository root:
  .venv/bin/python code/figure-reproduction/check_lean_probe_E7.py [direction]
About 10 minutes on 14 cores and 9.4 GB of memory.  It is part of ./verify.sh full.
"""
import os
import sys

from checklib import R_GOE, R2, RANDOM20, Report, Z, distinct, match_blocks, pooled, run_module, sandbox

name = sys.argv[1] if len(sys.argv) > 1 else "qgen"
assert name == "qgen" or name in RANDOM20, "direction must be qgen or one of qrand101 to qrand120"
stored = "Generic_probe_%s_E7_%s_values.npz" % (name, "mf" if name == "qgen" else "lean")
rep = Report("First-order spectrum at E = 7 rebuilt with the lean probe, direction %s" % name)
out = sandbox("lean_probe_E7_%s" % name)
status, seconds, log = run_module("r2g_generic_probe_mf_lean.py", [name, 7], out, threads=os.cpu_count())
rep.check("r2g_generic_probe_mf_lean.py %s 7 ran" % name, status == 0, "%.0f s, log %s" % (seconds, log))
if status == 0:
    new, old = Z("Generic_probe_%s_E7_lean_values.npz" % name, out), Z(stored)
    a = {k: distinct(new[k]) for k in new.files if k.startswith("rho")}
    b = {k: distinct(old[k]) for k in old.files if k.startswith("rho")}
    rep.check("four sub-blocks of 3536 values", sorted(v.size for v in a.values()) == [3536] * 4, str({k: v.size for k, v in a.items()}))
    dev, same_keys = match_blocks(a, b)
    rep.check("spectra equal to data/r2_out/%s" % stored, dev <= 2e-9, "largest difference %.1e" % dev)
    if not same_keys:
        rep.info("note: the characters are numbered in a different order on this machine, the spectra agree as sets")
    r, sem, n = pooled(list(a.values()))
    rep.info("pooled <r> = %.4f +- %.4f over %d ratios (GOE %.4f)" % (r, sem, n, R_GOE))
rep.finish()
