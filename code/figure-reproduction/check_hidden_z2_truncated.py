#!/usr/bin/env python3
"""Rerun the search for the hidden Z_2 with the truncated Hamiltonian for one direction, and compare.

This is the method described in the End Matter.  The script runs code/kummer/r2g_hidden_lean.py in a sandbox.  For
the sub-block of the trivial character with Z_R = +1, that module builds H_0 and the deformation on the primaries
through E = 6, including the matrix elements between different levels, diagonalizes H_0 + lambda D at lambda = 0.2 and
0.35, links eigenvectors of the two couplings whose overlap exceeds a threshold, and reads off the connected
components.  It then evaluates the first-order singular values of E = 2, 4, 6 inside each component.

The fresh eigenspaces are compared with the stored record data/r2_out/Hidden_lean_<direction>_L6_values.npz.  For
qrand101 to qrand116 the module also validates itself against the summaries of the earlier cluster runs
(scan_L6/Generic_firstorder_<direction>_L6.json) and stops if they disagree.

Run from the repository root:
  .venv/bin/python code/figure-reproduction/check_hidden_z2_truncated.py [direction]
The direction is one of qrand101 to qrand120 (default qrand105).  About 6 minutes on 4 cores, 5.4 GB.
"""
import os
import sys

import numpy as np

from checklib import J, R2, RANDOM20, Report, Z, distinct, run_module, sandbox

name = sys.argv[1] if len(sys.argv) > 1 else "qrand105"
assert name in RANDOM20, "direction must be one of qrand101 to qrand120"
rep = Report("Hidden Z_2 from the truncated Hamiltonian through E = 6, direction %s" % name)
inputs = ["Wall_scan_E%d%s" % (E, ext) for E in (3, 4, 5, 6) for ext in (".json", "_values.npz")]
ref = os.path.join("scan_L6", "Generic_firstorder_%s_L6.json" % name)
if os.path.exists(os.path.join(R2, ref)):
    inputs.append(ref)
out = sandbox("hidden_z2_truncated_%s" % name, inputs)
status, seconds, log = run_module("r2g_hidden_lean.py", [name], out, threads=4)
rep.check("r2g_hidden_lean.py %s ran" % name, status == 0, "%.0f s, log %s" % (seconds, log))
if status == 0:
    new, old = Z("Hidden_lean_%s_L6_values.npz" % name, out), Z("Hidden_lean_%s_L6_values.npz" % name)
    rec, rec_old = J("Hidden_lean_%s_L6.json" % name, out), J("Hidden_lean_%s_L6.json" % name)
    rep.check("two components, of the sizes of the stored record", rec["leaf_sizes"] == rec_old["leaf_sizes"], str(rec["leaf_sizes"]))
    rep.check("same arrays as the stored record", sorted(new.files) == sorted(old.files), ", ".join(sorted(new.files)))
    for E in (2, 4, 6):
        a = sorted([distinct(new["E%d__b1+" % E]), distinct(new["E%d__b1-" % E])], key=len)
        b = sorted([distinct(old["E%d__b1+" % E]), distinct(old["E%d__b1-" % E])], key=len)
        same = [x.size for x in a] == [y.size for y in b]
        dev = max(float(np.max(np.abs(x - y))) for x, y in zip(a, b)) if same else np.inf
        rep.check("E = %d: eigenspaces of %d and %d values equal to the stored record" % (E, a[1].size, a[0].size), dev <= 2e-9,
                  "largest difference %.1e" % dev)
    for E in ("4", "6"):
        d = abs(rec["r_full"][E]["r_mean"] - rec_old["r_full"][E]["r_mean"])
        rep.check("E = %s: fully resolved <r> = %.4f equal to the stored record" % (E, rec["r_full"][E]["r_mean"]), d < 1e-8, "difference %.1e" % d)
    if "validation" in rec:
        rep.check("the module's validation against the earlier cluster run", rec["validation"]["ok"], "largest difference of <r> %.1e" % rec["validation"]["max_r_dev"])
rep.finish()
