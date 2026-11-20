#!/usr/bin/env python3
"""Rebuild first-order spectra from the free-field code and compare them with the stored records.

For each requested level E the script runs code/kummer/r2g_wall_scan.py in a sandbox.  That module builds the space of
massive primaries at level E, the four first-order matrices M_mu of the quaternion units, and the singular values of
M_c = sum_mu q_mu M_mu in every sub-block of the generic stabilizer, for 50 directions: the reference direction (qgen),
a second random direction (qgen2) and qrand101 to qrand148.  Nothing is read from data/ during the rebuild.

The fresh spectra are then compared with
  * data/r2_out/Wall_scan_E<E>_values.npz, array by array, for all 50 directions,
  * the sub-blocks of the reference direction behind the Letter's table (Generic_firstorder_qgen_L7.npz), which came
    from a different code path (truncated Hamiltonian, cluster run).
At even E the scan covers the three non-trivial characters.  The trivial character, which carries the hidden Z_2, is
rebuilt by check_hidden_z2_firstorder.py.

Arrays are matched by content, so the comparison does not depend on the order in which the characters are numbered.

Run from the repository root:
  .venv/bin/python code/figure-reproduction/check_first_order_rebuild.py            E = 3, 4, 5, about half a minute
  .venv/bin/python code/figure-reproduction/check_first_order_rebuild.py 6          E = 6, about 4.5 minutes
Memory: 0.2 GB at E = 3 and 4, 0.6 GB at E = 5, 3.5 GB at E = 6.
"""
import sys

import numpy as np

from checklib import Report, Z, distinct, match_blocks, run_module, sandbox

levels = [int(a) for a in sys.argv[1:]] or [3, 4, 5]
rep = Report("First-order spectra rebuilt from the free-field code, E = %s" % ", ".join(map(str, levels)))
L7 = Z("Generic_firstorder_qgen_L7.npz")

for E in levels:
    rep.section("E = %d" % E)
    out = sandbox("rebuild_E%d" % E)
    status, seconds, log = run_module("r2g_wall_scan.py", [E], out)
    if not rep.check("r2g_wall_scan.py %d ran" % E, status == 0, "%.0f s, log %s" % (seconds, log)):
        continue
    new, old = Z("Wall_scan_E%d_values.npz" % E, out), Z("Wall_scan_E%d_values.npz" % E)
    names = sorted({k.split("__")[0] for k in old.files})
    rep.check("same directions and sub-blocks as the stored record", sorted(new.files) == sorted(old.files),
              "%d arrays for %d directions" % (len(new.files), len(names)))
    worst, keys_ok = 0.0, True
    for n in names:
        dev, same = match_blocks({k: distinct(new[k]) for k in new.files if k.startswith(n + "__")},
                                 {k: distinct(old[k]) for k in old.files if k.startswith(n + "__")})
        worst, keys_ok = max(worst, dev), keys_ok and same
    rep.check("spectra equal to the stored record for all %d directions" % len(names), worst <= 2e-9, "largest difference %.1e" % worst)
    if not keys_ok:
        rep.info("note: the characters are numbered in a different order on this machine, the spectra agree as sets")
    leaves = [distinct(L7[k]) for k in L7.files if k.startswith("E%d_leaf" % E)] if E <= 7 else []
    if leaves:
        devs = []
        for k in new.files:
            if k.startswith("qgen__"):
                v = distinct(new[k])
                devs.append(min([float(np.max(np.abs(v - w))) for w in leaves if w.size == v.size] or [np.inf]))
        rep.check("reference direction: each rebuilt sub-block equals a sub-block of the Letter's table", max(devs) <= 2e-9,
                  "%d sub-blocks, largest difference %.1e" % (len(devs), max(devs)))

rep.finish()
