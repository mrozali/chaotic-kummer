#!/usr/bin/env python3
"""Rebuild the hidden Z_2 of an even level from first-order data at two directions, and compare.

The script runs code/kummer/r2g_hidden_firstorder.py in a sandbox.  For the sub-block of the trivial character with
Z_R = +1 at level E, that module builds the first-order matrices of the four quaternion units, takes the singular
vectors at two random directions, links vectors whose overlap exceeds a threshold, and reads off the connected
components.  These are the eigenspaces of the hidden Z_2.  The module then evaluates the singular values inside each
eigenspace for 22 directions: qgen (the reference direction), qgen2 and qrand101 to qrand120.  Its own gates must pass:
the components are the same for several thresholds, the ten Gram matrices of the quaternion units are block diagonal on
them, and the common null space has the expected dimension.  This is the statement of the End Matter that the
involution does not depend on the direction, at the level of first-order data.

The fresh eigenspaces are compared with
  * the stored record data/r2_out/Hidden_firstorder_E<E>_values.npz,
  * the eigenspaces of the reference direction found from the truncated Hamiltonian (Generic_firstorder_qgen_L7.npz),
  * the eigenspaces of the 20 random directions found from the truncated Hamiltonian (Hidden_lean_<dir>_L6_values.npz).
The comparison is of the first-order singular values inside each eigenspace, which identifies the eigenspaces of the two
methods with each other.

Run from the repository root:
  .venv/bin/python code/figure-reproduction/check_hidden_z2_firstorder.py           E = 4, a few seconds
  .venv/bin/python code/figure-reproduction/check_hidden_z2_firstorder.py 6         E = 6, about a minute, 3.8 GB
E = 8 needs about 80 GB and four hours on 32 cores (code/cluster/hidden_firstorder_E8.sh).  Its stored record is
checked by check_paper_numbers.py.
"""
import sys

import numpy as np

from checklib import J, RANDOM20, Report, Z, distinct, run_module, sandbox

E = int(sys.argv[1]) if len(sys.argv) > 1 else 4
assert E in (4, 6), "this check runs at E = 4 or 6"
SIZES = {4: [26, 50], 6: [451, 899]}[E]
rep = Report("Hidden Z_2 at E = %d from the first-order matrices at two directions" % E)


def pair(z, a, b):
    return sorted([distinct(z[a]), distinct(z[b])], key=len)


def deviation(x, y):
    if [u.size for u in x] != [v.size for v in y]:
        return np.inf
    return max(float(np.max(np.abs(u - v))) for u, v in zip(x, y))


out = sandbox("hidden_z2_firstorder_E%d" % E)
status, seconds, log = run_module("r2g_hidden_firstorder.py", [E], out)
rep.check("r2g_hidden_firstorder.py %d ran and its gates passed" % E, status == 0, "%.0f s, log %s" % (seconds, log))
if status == 0:
    new, rec = Z("Hidden_firstorder_E%d_values.npz" % E, out), J("Hidden_firstorder_E%d.json" % E, out)
    old = Z("Hidden_firstorder_E%d_values.npz" % E)
    rep.check("gates", all(v is not False for v in rec["gates"].values()), str(rec["gates"]))
    dirs = sorted({k.split("__")[0] for k in new.files})
    rep.check("eigenspaces of %d and %d values along each of the %d directions" % (SIZES[1], SIZES[0], len(dirs)),
              all([x.size for x in pair(new, d + "__leaf0", d + "__leaf1")] == SIZES for d in dirs))
    common = sorted({k.split("__")[0] for k in old.files} & set(dirs))
    dev = max(deviation(pair(new, d + "__leaf0", d + "__leaf1"), pair(old, d + "__leaf0", d + "__leaf1")) for d in common)
    rep.check("equal to the stored record for its %d directions" % len(common), dev <= 2e-9, "largest difference %.1e" % dev)
    L7 = Z("Generic_firstorder_qgen_L7.npz")
    dev = deviation(pair(new, "qgen__leaf0", "qgen__leaf1"), pair(L7, "E%d_leaf1" % E, "E%d_leaf2" % E))
    rep.check("reference direction: same values in each eigenspace as found from the truncated Hamiltonian", dev <= 2e-9, "largest difference %.1e" % dev)
    dev = max(deviation(pair(new, d + "__leaf0", d + "__leaf1"),
                        pair(Z("Hidden_lean_%s_L6_values.npz" % d), "E%d__b1+" % E, "E%d__b1-" % E)) for d in RANDOM20)
    rep.check("20 random directions: same values in each eigenspace as found from the truncated Hamiltonian", dev <= 2e-9, "largest difference %.1e" % dev)

rep.finish()
