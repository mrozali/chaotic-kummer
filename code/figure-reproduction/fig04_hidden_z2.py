#!/usr/bin/env python3
"""Fig. 4 of the Letter (fig:hiddenZ2, End Matter): the hidden Z_2 along the reference direction at E = 6.

The sub-block of the trivial character of the stabilizer with Z_R = +1 has 1350 first-order values
(Generic_firstorder_qgen_L7.npz).  The hidden Z_2 splits them into the eigenspaces E6_leaf1 (899 values) and E6_leaf2
(451 values).

Top: 40 consecutive levels of the unresolved sub-block, numbers 472 to 511 of 1350, and the same levels split into the
two eigenspaces.  In this window all seven pairs of neighbors closer than a fifth of the mean spacing join levels from
different eigenspaces (asserted).

Bottom: distribution P(s) of the unfolded nearest-neighbor spacings for the unresolved sub-block and for its two
eigenspaces (each unfolded on its own, spacings pooled), with the GOE Wigner surmise and with two independent GOE
spectra superposed in the proportion 899 : 451.  The unfolding is that of Fig. 3: Chebyshev degree max(12, round(n/100)),
5% trimmed at each end, monotone fit asserted.  The superposed spectra are GOE matrices scaled to the same semicircle,
300 samples from numpy's default_rng(6), sent through the same unfolding.

The script also prints the gap ratios quoted in the End Matter: the unresolved sub-block, the two eigenspaces, and the
mean and standard deviation of <r> over the 300 superposed samples (central 90% of each sample kept).

Input:   data/r2_out/Generic_firstorder_qgen_L7.npz.
Output:  code/figure-reproduction/generated/fig_hiddenZ2.pdf and .png, compared with paper/fig_hiddenZ2.pdf.
         About 12 s.

Run from the repository root:  .venv/bin/python code/figure-reproduction/fig04_hidden_z2.py
"""
import os
import sys

import numpy as np
from numpy.polynomial import chebyshev as cheb
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import R2, BLUE, INK, INK2, GRAY, gap_ratios, save

z = np.load(os.path.join(R2, "Generic_firstorder_qgen_L7.npz"), allow_pickle=False)
A = np.unique(np.round(np.asarray(z["E6_leaf1"], float), 9))
B = np.unique(np.round(np.asarray(z["E6_leaf2"], float), 9))
assert (A.size, B.size) == (899, 451)
U = np.sort(np.concatenate([A, B]))
assert np.unique(np.round(U, 9)).size == U.size
rU, rA, rB = gap_ratios(U), gap_ratios(A), gap_ratios(B)
rAB = np.concatenate([rA, rB])

# unfolded spacings, as for Fig. 3 (fig_spacings_singlet.py)
def unfold(x, deg=None, trim=0.05):
    x = np.unique(np.round(np.asarray(x, dtype=float), 9))
    n = x.size
    if deg is None:
        deg = max(12, int(round(n / 100)))          # one fit parameter per 100 levels, at least degree 12
    t = (2.0 * x - x[0] - x[-1]) / (x[-1] - x[0])
    c = cheb.chebfit(t, np.arange(1, n + 1), deg)
    xi = cheb.chebval(t, c)
    k = int(trim * n)
    tt = np.linspace(t[k], t[n - k - 1], 4000)
    assert np.all(cheb.chebval(tt, cheb.chebder(c)) > 0), "non-monotone unfolding"
    return xi[k:n - k]


def spacings(xi):
    s = np.diff(xi)
    return s / s.mean()


sU = spacings(unfold(U))
sAB = np.concatenate([spacings(unfold(A)), spacings(unfold(B))])

# two independent GOE spectra superposed in the proportion 899 : 451: gap ratios (central 90%) and unfolded spacings
NSIM = 300
rng = np.random.default_rng(6)
sim, simS = [], []
for _ in range(NSIM):
    lev = []
    for n in (A.size, B.size):
        h = rng.standard_normal((n, n))
        lev.append(np.linalg.eigvalsh((h + h.T) / np.sqrt(2 * n)))        # semicircle of radius 2
    m = np.sort(np.concatenate(lev))
    k = int(0.05 * m.size)
    sim.append(gap_ratios(m[k:m.size - k]))
    simS.append(spacings(unfold(m)))
sim_means = np.array([g.mean() for g in sim])  # per-sample <r>: its mean and sd are quoted in the End Matter
sim = np.concatenate(sim)
simS = np.concatenate(simS)
f1 = A.size / U.size
for nm, x in (("unresolved", sU), ("two sectors", sAB), ("two GOE spectra (sim.)", simS)):
    print(f"P(s): {nm:24s} {x.size:6d} spacings, fraction below s = 0.1: {np.mean(x < 0.1):.3f}, <s^2> = {np.mean(x ** 2):.3f}")
print(f"GOE surmise: fraction below 0.1 = {1 - np.exp(-np.pi * 0.01 / 4):.3f}, <s^2> = {4 / np.pi:.3f}; "
      f"two independent spectra {A.size}:{B.size}: P(0) = 2 f (1 - f) = {2 * f1 * (1 - f1):.3f}")

# the window of the barcode
I0, NW = 472, 40
lab = np.concatenate([np.zeros(A.size, int), np.ones(B.size, int)])
vv = np.concatenate([A, B])
o = np.argsort(vv)
vv, lab = vv[o], lab[o]
win, wl = vv[I0:I0 + NW], lab[I0:I0 + NW]
d = np.diff(win)
close = np.where(d < 0.2 * d.mean())[0]
assert np.all(wl[close] != wl[close + 1]), "a close pair within one sector"

print(f"<r>: unresolved {rU.mean():.3f} +- {rU.std() / np.sqrt(rU.size):.3f} ({rU.size}); sectors {rA.mean():.3f} and "
      f"{rB.mean():.3f}, pooled {rAB.mean():.3f}; two GOE spectra 899:451 {sim.mean():.3f}; GOE 0.531")
print(f"two GOE spectra 899:451, per-sample <r>: {sim_means.mean():.4f} +- {sim_means.std(ddof=1):.4f} "
      f"over {NSIM} samples")
print(f"window: sigma {win[0]:.3f}..{win[-1]:.3f}, {np.sum(wl == 0)} + {np.sum(wl == 1)} levels, {close.size} close pairs "
      f"(all cross-sector), at s/mean = {np.round(d[close] / d.mean(), 3)}")

# ---------------------------------------------------------------- figure (single column, stacked)
FIG_W, FIG_H = 3.4, 2.85                          # inches; \columnwidth of revtex prl is 3.40 in
X0T, X0B, X1 = 0.62, 0.47, 3.32                   # left edges (top: row labels, bottom: y label) and common right edge
fig = plt.figure(figsize=(FIG_W, FIG_H))
axl = fig.add_axes([X0T / FIG_W, 2.16 / FIG_H, (X1 - X0T) / FIG_W, 0.62 / FIG_H])
axr = fig.add_axes([X0B / FIG_W, 0.33 / FIG_H, (X1 - X0B) / FIG_W, 1.45 / FIG_H])
GREY_FILL = "#d9d8d3"
DASH = (0, (3, 1.6))

pad = 0.4 * d.mean()
rows = [(2, win, GRAY, "unresolved"), (1, win[wl == 0], BLUE, "sector 1"), (0, win[wl == 1], INK, "sector 2")]
for y, x, col, _ in rows:
    axl.vlines(x, y - 0.32, y + 0.32, color=col, lw=1.0)
axl.set_yticks([2, 1, 0], [r[3] for r in rows])
axl.set_ylim(-0.55, 2.55)
axl.set_xlim(win[0] - pad, win[-1] + pad)
axl.set_xticks([2.90, 2.95, 3.00, 3.05])
axl.tick_params(axis="y", length=0, pad=3)
for sp in ("left", "right", "top"):
    axl.spines[sp].set_visible(False)
axl.set_xlabel(r"first-order shift $\nu$", labelpad=2)

bins = np.arange(0.0, 3.51, 0.15)
axr.hist(sU, bins=bins, density=True, histtype="stepfilled", color=GREY_FILL, lw=0)
axr.hist(sU, bins=bins, density=True, histtype="step", color=GRAY, lw=1.0)
axr.hist(sAB, bins=bins, density=True, histtype="step", color=BLUE, lw=1.1)
ss = np.linspace(0, 3.5, 400)
axr.plot(ss, np.pi / 2 * ss * np.exp(-np.pi * ss ** 2 / 4), color=INK, lw=1.0)
hs, _ = np.histogram(simS, bins=bins, density=True)
axr.plot(0.5 * (bins[1:] + bins[:-1]), hs, color=INK2, lw=0.9, ls=DASH)
axr.set_xlim(0, 3.5)
axr.set_ylim(0, 1.1)
axr.set_yticks([0, 0.5, 1])
axr.set_xlabel(r"unfolded spacing $s$", labelpad=2)
axr.set_ylabel(r"$P\hspace{0.1}(s)$", labelpad=2)
hr = [plt.Rectangle((0, 0), 1, 1, fc=GREY_FILL, ec=GRAY, lw=1.0), plt.Line2D([], [], color=BLUE, lw=1.1),
      plt.Line2D([], [], color=INK, lw=1.0), plt.Line2D([], [], color=INK2, lw=0.9, ls=DASH)]
axr.legend(hr, ["unresolved", "two sectors", "GOE Wigner surmise", "two GOE spectra"],
           loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=False, fontsize=7, handlelength=1.6,
           labelspacing=0.3, borderaxespad=0.2)

save(fig, __file__, "fig_hiddenZ2")
