#!/usr/bin/env python3
"""The explicit form of the hidden Z_2 at E = 2 (Supplemental Material, eq. Sreflection), recomputed.

The Supplemental Material states that along the singlet, in the sector that is trivial under the point group and has
Z_R = +1, the level E = 2 has one twisted state and three untwisted states: the oscillator state |osc> (a = b = 0),
the combination |m+w> of the pure-momentum (b = a) and pure-winding (b = -a) states, and the state |mw> (b != +-a).
On them the involution S acts as the reflection that keeps |v> = 2|osc> + |m+w> - |mw> and reverses its orthogonal
complement, and it acts as -1 on the twisted state.  Along the singlet |v> therefore stays at E = 2 to first order.

This script rebuilds that statement with the modules of code/kummer:
  1. the truncated Hamiltonian H_0 + lambda D along the singlet on the massive primaries through E = th_max, in the
     sector of the trivial character of the point group with Z_R = +1 (and even under the left-right exchange),
  2. the search for jointly invariant subspaces of the End Matter (r2c_gate19_lib.find_split: eigenvectors at
     lambda = 0.2 and 0.35, overlap graph, connected components), which gives S as the difference of the two projectors,
  3. S restricted to the four states of E = 2, and the weight of its invariant vector |v> on the momentum classes.
The weights of 2|osc> + |m+w> - |mw> are 4/6 on the oscillator state, 1/12 each on momentum and winding, and 1/6 on the
mixed state.  The relative signs inside |v> depend on the phases of the three states and are not checked.

Run from the repository root:
  .venv/bin/python code/figure-reproduction/check_hidden_z2_E2.py          th_max = 4, about 2 s
  .venv/bin/python code/figure-reproduction/check_hidden_z2_E2.py 6        th_max = 6, about 20 s
"""
import sys

import numpy as np

from checklib import KUMMER, Report

sys.path.insert(0, KUMMER)
import r2c_gate19_lib as L          # noqa: E402

TH = int(sys.argv[1]) if len(sys.argv) > 1 else 4
assert TH in (4, 5, 6), "th_max must be 4, 5 or 6"
rep = Report("Hidden Z_2 at E = 2 along the singlet, from the truncated Hamiltonian through E = %d" % TH)

S = L.get_space(TH, "prim", verbose=False)
sec = L.get_irrep(TH, 0, "prim", verbose=False)            # irrep 0: the trivial character of the point group
names, parts, prov = L.split_all_prov(sec, S)
rep.check("further exact involutions of the sector: left-right exchange and Z_R", names == ["Pi", "Z_R"], str(names))
label = (1, 1)
sub, emb = parts[label], prov[label]
Eu, Et = sub.E0[:sub.nu], sub.E0[sub.nu:]
iu, it = np.where(Eu == 2)[0], sub.nu + np.where(Et == 2)[0]
rep.check("E = 2: three untwisted states and one twisted state", (iu.size, it.size) == (3, 1), "%d and %d" % (iu.size, it.size))

Sop, info = L.find_split(sub, TH)
ok = rep.check("the sector of %d states splits into two jointly invariant subspaces" % sub.n, Sop is not None and info["n_components"] == 2,
               "component sizes %s, %d states without first-order shift left out" % (info["comp_sizes"], info["n_unlifted"]))
if ok:
    H0 = np.diag(sub.E0)
    c0 = np.linalg.norm(Sop @ H0 - H0 @ Sop)
    c1 = np.linalg.norm(Sop @ sub.H(0.3) - sub.H(0.3) @ Sop)
    rep.check("S commutes with H_0 and with H_0 + lambda D at lambda = 0.3", max(c0, c1) < 1e-8, "norms of the commutators %.1e and %.1e" % (c0, c1))
    block = Sop[np.ix_(iu, iu)]
    w, V = np.linalg.eigh((block + block.conj().T) / 2)
    k = 0 if abs(w[0] - w[1]) > abs(w[1] - w[2]) else 2                    # the eigenvalue that occurs once
    sign = np.sign(w[k])                                                   # overall sign of S: a convention
    v = V[:, k]
    others = np.delete(w, k)
    rep.check("on the untwisted states S is a reflection: one invariant vector |v>, the other two reversed",
              abs(abs(w[k]) - 1) < 1e-8 and np.max(np.abs(others + w[k])) < 1e-8, "eigenvalues %s" % np.round(sign * w, 8))
    st = float(np.real(Sop[it[0], it[0]]))
    rep.check("S = -1 on the twisted state", abs(sign * st + 1) < 1e-8, "%.8f" % (sign * st))
    full = emb["Qu"][:, iu] @ v                                            # |v> in the basis of the irrep sector
    coef = full[sec.uoff[2]: sec.uoff[2] + sec.mu[2]]
    weight = {"osc": 0.0, "momentum": 0.0, "winding": 0.0, "mixed": 0.0}
    for (a, b), x in L.member_weight_table(sec, 2, coef).items():
        a, b = np.array(a), np.array(b)
        if not a.any() and not b.any():
            weight["osc"] += x
        elif np.array_equal(a, b):
            weight["momentum"] += x
        elif np.array_equal(a, -b):
            weight["winding"] += x
        else:
            weight["mixed"] += x
    total = sum(weight.values())
    expected = {"osc": 4 / 6, "momentum": 1 / 12, "winding": 1 / 12, "mixed": 1 / 6}
    dev = max(abs(weight[c] / total - expected[c]) for c in weight)
    rep.check("|v> has the weights of 2|osc> + |m+w> - |mw>: 4/6, 1/12, 1/12, 1/6", dev < 1e-8 and abs(total - 1) < 1e-8,
              ", ".join("%s %.6f" % (c, weight[c] / total) for c in weight))
    M = sub.M[np.ix_(it - sub.nu, iu)]                                     # first-order matrix of the level, 1 x 3
    rep.check("|v> stays at E = 2 to first order: M |v> = 0", np.linalg.norm(M @ v) < 1e-9 * np.linalg.norm(M),
              "|M v| = %.1e, |M| = %.3f" % (np.linalg.norm(M @ v), np.linalg.norm(M)))
rep.finish()
