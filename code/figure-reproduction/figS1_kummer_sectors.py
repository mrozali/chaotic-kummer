#!/usr/bin/env python3
"""Fig. S1 of the Supplemental Material (fig:sectors): the levels of the Kummer point by symmetry sector.

Left: the momenta (a_i, b_i) of one circle of the torus at the free-fermion radius: a_i, b_i in Z/2 with a_i - b_i in 2Z.
The integer momenta (s_i = 0) have energy (a_i^2 + b_i^2)/2 in Z, the half-integer ones (s_i = 1) in Z + 1/4.  The
circles are the curves of constant energy.

Right: the energies E = h + hbar of the spin-zero states, h = hbar, in the symmetry sectors of weight |s| = 0, ..., 4,
twisted (left) and untwisted (right) in each column.  The twisted tower is the same in every sector.  The untwisted
energies are |a|^2 + 2 N_L with N_L an integer: Z in the invariant sector, Z + |s|/4 for |s| = 1, 2, 3, and only the odd
integers for |s| = 4.  The levels are enumerated from the lattice by the script, not typed in.  Twisted and untwisted
states are degenerate, and mix at first order, where both columns have a level above E = 0 (shaded).

No data files are read.
Output:  code/figure-reproduction/generated/fig_kummer_sectors.pdf and .png, compared with
         paper/fig_kummer_sectors.pdf.

Run from the repository root:  .venv/bin/python code/figure-reproduction/figS1_kummer_sectors.py
"""
import itertools
import os
import sys
from fractions import Fraction as Fr

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import INK, INK2, GRAY, LIGHT, save

FILL = "#cde2fb"      # step 100 of the blue ramp: the blocks that split at first order
EMAX = 4

# ---------------------------------------------------------------- the lattice of one circle, and the sector levels
HALF = [Fr(k, 2) for k in range(-9, 10)]


def circle_momenta(si):
    return [(a, b) for a in HALF for b in HALF
            if (2 * a) % 2 == si and (2 * b) % 2 == si and (a - b) % 2 == 0]


def untwisted_levels(s):
    """Energies E <= EMAX of the spin-zero untwisted states of the sector s (momenta and integer oscillator levels)."""
    per = [sorted({(a * a / 2, b * b / 2) for a, b in circle_momenta(si) if a * a / 2 <= EMAX and b * b / 2 <= EMAX})
           for si in s]
    out = set()
    for combo in itertools.product(*per):
        h0, hb0 = sum(x for x, _ in combo), sum(y for _, y in combo)
        for NL in range(EMAX + 1):
            NR = h0 + NL - hb0                       # h = hb fixes the right-moving level
            if NR >= 0 and NR.denominator == 1 and 2 * (h0 + NL) <= EMAX:
                out.add(2 * (h0 + NL))
    return sorted(out)


LEVELS_U = {}
for s in itertools.product((0, 1), repeat=4):
    lv = untwisted_levels(s)
    w = sum(s)
    assert LEVELS_U.setdefault(w, lv) == lv           # the same in every sector of a given weight
LEVELS_T = list(range(EMAX + 1))                      # eq:weights_tw: every integer, in every sector
for w, lv in LEVELS_U.items():
    print(f"|s| = {w}: untwisted spin-zero levels E <= {EMAX}:", [str(e) for e in lv])

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(6.5, 2.5))                 # no panel titles: the axes keep their size
axa = fig.add_axes([0.075, 0.165, 0.272, 0.8085])
axb = fig.add_axes([0.42, 0.165, 0.565, 0.8085])

# (a) one circle
ax = axa
R = 2.3
th = np.linspace(0, 2 * np.pi, 400)
for E in (Fr(1, 4), 1, Fr(5, 4), 2, Fr(9, 4)):
    r = np.sqrt(2 * float(E))
    integer = Fr(E).denominator == 1
    ax.plot(r * np.cos(th), r * np.sin(th), color=GRAY if integer else LIGHT, lw=0.6,
            ls="-" if integer else (0, (1.5, 1.5)), zorder=1)
for si, kw in ((0, dict(mfc=INK, mec=INK)), (1, dict(mfc="white", mec=INK))):
    pts = np.array([(float(a), float(b)) for a, b in circle_momenta(si) if abs(a) <= R and abs(b) <= R])
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=3.6, mew=0.8, zorder=3, **kw)
# the energy (a^2 + b^2)/2 of a few momenta
for (a, b), lab in (((1, 1), "1"), ((2, 0), "2"), ((0.5, 0.5), r"$\frac{1}{4}$"), ((1.5, -0.5), r"$\frac{5}{4}$"),
                    ((1.5, 1.5), r"$\frac{9}{4}$")):
    ax.text(a + 0.13, b + 0.13, lab, fontsize=7, color=INK2, ha="left", va="bottom",
            bbox=dict(boxstyle="square,pad=0.04", fc="white", ec="none"), zorder=4)
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(-2.6, 3.35)
ax.set_aspect("equal")
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-2, -1, 0, 1, 2])
ax.spines["left"].set_bounds(-2.6, 2.6)
ax.set_xlabel(r"left-moving momentum $a_i$")
ax.set_ylabel(r"right-moving momentum $b_i$", labelpad=1)
hs = [Line2D([], [], ls="", marker="o", ms=3.6, mfc=INK, mec=INK, mew=0.8),
      Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec=INK, mew=0.8)]
ax.legend(hs, [r"$s_i=0$", r"$s_i=1$"], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
          handlelength=0.8, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.1, fontsize=7.5)

# (b) the levels by sector weight
ax = axb
DX, HW = 0.2, 0.15                                       # half distance of the two sub-columns, half width of a level
for E in range(EMAX + 1):
    ax.axhline(E, color=LIGHT, lw=0.4, zorder=0)
for w in range(5):
    xt, xu = w - DX, w + DX
    for E in LEVELS_T:
        col = GRAY if E == 0 else INK
        ax.plot([xt - HW, xt + HW], [E, E], color=col, lw=1.4, solid_capstyle="butt", zorder=3)
    for E in LEVELS_U[w]:
        col = GRAY if E == 0 else INK
        ax.plot([xu - HW, xu + HW], [float(E), float(E)], color=col, lw=1.4, solid_capstyle="butt", zorder=3)
        if E > 0 and E in LEVELS_T:
            ax.add_patch(FancyBboxPatch((xt - HW - 0.04, float(E) - 0.11), 2 * DX + 2 * HW + 0.08, 0.22,
                                        boxstyle="round,pad=0,rounding_size=0.05", fc=FILL, ec="none", zorder=2))
    ax.text(xt, -0.2, "tw", fontsize=6.5, color=INK2, ha="center", va="top")
    ax.text(xu, -0.2, "untw", fontsize=6.5, color=INK2, ha="center", va="top")
for w, lab in ((1, r"$+\frac{1}{4}$"), (2, r"$+\frac{1}{2}$"), (3, r"$+\frac{3}{4}$")):
    ax.text(w + DX + HW + 0.03, w / 4, lab, fontsize=7, color=INK2, ha="left", va="center")
ax.set_xlim(-0.55, 4.55)
ax.set_ylim(-0.55, EMAX + 0.72)
ax.set_xticks(range(5))
ax.set_xticklabels([rf"$|\mathbf{{s}}|={w}$" + "\n" + f"({m} sector{'' if m == 1 else 's'})"
                    for w, m in zip(range(5), (1, 4, 6, 4, 1))], fontsize=8)
ax.tick_params(axis="x", length=0)
ax.set_yticks(range(EMAX + 1))
ax.spines["left"].set_bounds(0, EMAX)
ax.set_ylabel(r"$E=h+\bar h$ at spin zero")
ax.spines["bottom"].set_visible(False)
hs = [Patch(fc=FILL, ec="none"), Line2D([], [], color=GRAY, lw=1.4)]
ax.legend(hs, ["degenerate, split at first order", "protected ground states"],
          loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.2,
          handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1, fontsize=7.5)

save(fig, __file__, "fig_kummer_sectors")
