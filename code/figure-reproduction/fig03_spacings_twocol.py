#!/usr/bin/env python3
"""Fig. 3 of the Letter (fig:spacings), two-column layout of the arXiv version: unfolded first-order spectra of the
reference direction and of the singlet.

Left: distribution P(s) of the unfolded nearest-neighbor spacings at E = 8.  Right: number variance Sigma^2(L), with an
inset on the reference direction at L >= 4.

Reference direction: at E = 6 and 7 the fully resolved sub-blocks of Generic_firstorder_qgen_L7.npz.  At E = 8 the three
sub-blocks without a hidden Z_2 (Generic_probe_qgen_E8_mf_values.npz) and the two eigenspaces of the hidden Z_2 of the
trivial character, with 6137 and 3079 values (Hidden_firstorder_E8_values.npz).
Singlet q = (0, 0, -1, 0): Singlet_firstorder_L8.npz, resolution B (every known symmetry resolved).  Non-zero singular
values, distinct to 1e-9.

Unfolding (Supplemental Material, Sec. S4): the staircase of a sub-block with n distinct levels is fitted by a Chebyshev
polynomial of degree max(12, round(n/100)), and 5% of the levels are dropped at each end.  The degrees for the
reference direction are 12, 12, 14, 14, 14 at E = 6, 35 at E = 7, and 92, 92, 92, 61, 31 at E = 8.  Every singlet
sub-block has at most 1059 levels and degree 12.  A sub-block is used only if the fitted staircase is monotone over the
retained range.  Every sub-block of the reference direction passes.  For the singlet, among the sub-blocks with at
least 100 levels, the cut keeps 11 of 16 at E = 7, 10 of 26 at E = 8 and one of two at E = 6, so the singlet is shown
at E = 7 and 8.

Controls printed by the script: (1) real Gaussian matrices with the rank-matched shapes of the singlet's sub-blocks,
sent through the same unfolding and cut, reproduce the GOE values of <s^2> and Sigma^2(L).  (2) Chiral GOE spectra with
the shapes of the reference direction's sub-blocks reproduce the GOE Sigma^2(L) for L <= 10.

The two layouts (fig03_spacings_twocol.py for the arXiv version, fig03_spacings_onecol.py for the journal version)
share the data, the unfolding and the printed numbers.  Only the figure section differs.

Inputs:  the files named above, all in data/r2_out.
Output:  code/figure-reproduction/generated/fig_spacings_singlet_twocol.pdf and .png, compared with paper/fig_spacings_singlet_twocol.pdf.  About 50 s.

Run from the repository root:  .venv/bin/python code/figure-reproduction/fig03_spacings_twocol.py
"""
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as cheb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import R2, BLUE, GREEN, GREEN_LIGHT, INK, INK2, GRAY, save

DEG, TRIM = 12, 0.05
LPP = 100          # levels per fit parameter (variant)


def degree(n):
    """Degree of the staircase fit for a sub-block of n distinct levels."""
    return max(DEG, int(round(n / LPP)))


# ---------------------------------------------------------------- unfolding (as in fig_spacings.py, plus the check)
def unfold(x, deg=None, trim=TRIM):
    """Unfolded levels of one sub-block and whether the fitted staircase is monotone on the retained range."""
    x = np.unique(np.round(np.asarray(x, dtype=float), 9))
    n = x.size
    if deg is None:
        deg = degree(n)
    t = (2.0 * x - x[0] - x[-1]) / (x[-1] - x[0])
    # rcond: keep every singular value (the default n * eps drops one at degree 61 for the 6137 eigenspace)
    c, (_, rank, _, _) = cheb.chebfit(t, np.arange(1, n + 1), deg, rcond=1e-15, full=True)
    assert rank == deg + 1, (n, deg, rank)
    xi = cheb.chebval(t, c)
    k = int(trim * n)
    tt = np.linspace(t[k], t[n - k - 1], 40000)
    monotone = bool(np.all(cheb.chebval(tt, cheb.chebder(c)) > 0))
    return xi[k:n - k], monotone


def spacings(xi):
    s = np.diff(xi)
    return s / s.mean()


def number_variance(xi, Ls, npos=4000):
    out = []
    for L in Ls:
        a = np.linspace(xi[0], xi[-1] - L, npos)
        out.append(np.var(np.searchsorted(xi, a + L) - np.searchsorted(xi, a)))
    return np.array(out)


def pooled_sigma2(unf, Ls):
    return np.average([number_variance(x, Ls) for x in unf], axis=0, weights=[x.size for x in unf])


# ---------------------------------------------------------------- reference curves (as in fig_spacings.py)
_t = np.linspace(1e-9, 200.0, 400001)
_dt = np.diff(_t)
_si = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(_t[1:]) / _t[1:] + np.sin(_t[:-1]) / _t[:-1]) * _dt)])
_cm = np.concatenate([[0.0], np.cumsum(0.5 * ((np.cos(_t[1:]) - 1) / _t[1:] + (np.cos(_t[:-1]) - 1) / _t[:-1]) * _dt)])


def Si(x):
    return np.interp(x, _t, _si)


def Ci(x):
    return np.euler_gamma + np.log(x) + np.interp(x, _t, _cm)


def sigma2_gue(L):
    return ((np.log(2 * np.pi * L) + np.euler_gamma + 1 - np.cos(2 * np.pi * L) - Ci(2 * np.pi * L)) / np.pi ** 2
            + L * (1 - 2 / np.pi * Si(2 * np.pi * L)))


def sigma2_goe(L):
    return 2 * sigma2_gue(L) + (Si(np.pi * L) / np.pi) ** 2 - Si(np.pi * L) / np.pi


def p_goe(s):
    return np.pi / 2 * s * np.exp(-np.pi * s * s / 4)


def p_gue(s):
    return 32 / np.pi ** 2 * s * s * np.exp(-4 * s * s / np.pi)


# ---------------------------------------------------------------- data
def dir1_blocks(E):
    if E == 8:
        # all five sub-blocks: the three stored ones and the two eigenspaces of the hidden Z2 of the
        # trivial-character sub-block, whose union is the stored unresolved sub-block
        z = np.load(os.path.join(R2, "Generic_probe_qgen_E8_mf_values.npz"), allow_pickle=False)
        h = np.load(os.path.join(R2, "Hidden_firstorder_E8_values.npz"), allow_pickle=False)
        leaves = [h["qgen__leaf0"], h["qgen__leaf1"]]
        assert [x.size for x in leaves] == [6137, 3079]
        assert np.abs(np.sort(np.concatenate(leaves)) - np.unique(np.round(z["rho0_labpp"], 9))).max() < 2e-9
        return [z[k] for k in ("rho2_labpp", "rho3_labpp", "rho6_labpp")] + leaves
    z = np.load(os.path.join(R2, "Generic_firstorder_qgen_L7.npz"), allow_pickle=False)
    return [z[k] for k in sorted(z.keys()) if k.startswith(f"E{E}_")]


ZS = np.load(os.path.join(R2, "Singlet_firstorder_L8.npz"), allow_pickle=False)
META = ZS["B_meta"]          # rows: k, ..., E, nu_E, nt_E, rank, n_distinct, ...


def extract(sv):
    """Non-zero singular values, distinct to 1e-9 (r2c_singlet_firstorder_L8.extract)."""
    sv = np.sort(np.asarray(sv, dtype=float))[::-1]
    nz = sv[sv > 1e-9 * max(sv[0], 1e-30)]
    return np.unique(np.round(nz, 9))


def singlet_blocks(E):
    keys = [k for k in ZS.keys() if re.fullmatch(rf"B_k\d+[pm]?_E{E}", k)]
    return [extract(ZS[k]) for k in keys]


def usable(blocks, nmin=20):
    unf, sizes = [], []
    for b in blocks:
        if np.unique(np.round(b, 9)).size < nmin:
            continue
        xi, mono = unfold(b)
        if mono:
            unf.append(xi)
            sizes.append(np.unique(np.round(b, 9)).size)
    return unf, sizes


LS = np.linspace(0.05, 10.0, 200)
D1 = {E: usable(dir1_blocks(E)) for E in (6, 7, 8)}
for E in (6, 7, 8):
    assert len(D1[E][0]) == len(dir1_blocks(E)), f"a direction-1 sub-block at E = {E} fails the monotone check"
SG = {E: usable(singlet_blocks(E), nmin=100) for E in (6, 7, 8)}

s8_d1 = np.concatenate([spacings(x) for x in D1[8][0]])
s8_sg = np.concatenate([spacings(x) for x in SG[8][0]])
sig_d1 = {E: pooled_sigma2(D1[E][0], LS) for E in (6, 7, 8)}
SG_LEVELS = (7, 8)
sig_sg = {E: pooled_sigma2(SG[E][0], LS) for E in SG_LEVELS}

LCHK = np.array([1.0, 2.0, 5.0, 10.0])
print(f"reference direction, E = 8: {s8_d1.size} spacings, <s^2> = {np.mean(s8_d1 ** 2):.3f}")
print(f"singlet,     E = 8: {s8_sg.size} spacings from sub-blocks of sizes {SG[8][1]}, <s^2> = {np.mean(s8_sg ** 2):.3f}"
      "   (GOE 1.286, Wigner surmise 1.273, GUE 1.180)")
for E in (6, 7, 8):
    print(f"Sigma^2 at L = {LCHK}: reference direction, E = {E}", np.round(np.interp(LCHK, LS, sig_d1[E]), 3))
for E in (6, 7, 8):
    print(f"singlet E = {E}: usable sub-blocks {SG[E][1]}")
for E in SG_LEVELS:
    print(f"Sigma^2 at L = {LCHK}: singlet,     E = {E}", np.round(np.interp(LCHK, LS, sig_sg[E]), 3))
print("Sigma^2 GOE                               ", np.round(sigma2_goe(LCHK), 3))

# matched null for the singlet: rank-matched real Gaussian matrices of the used sub-blocks, same pipeline and cut
rng = np.random.default_rng(2026)
for E in SG_LEVELS:
    shapes = []
    for n in SG[E][1]:
        row = [m for m in META if m[4] == E and m[8] == n][0]
        nt, nu, rk = int(row[6]), int(row[5]), int(row[7])
        shapes.append((max(nt - nu + rk, rk), rk))
    s2, sg = [], []
    for _ in range(10):
        unf = [unfold(np.linalg.svd(rng.standard_normal(sh), compute_uv=False))[0] for sh in shapes]
        s2.append(np.mean(np.concatenate([spacings(x) for x in unf]) ** 2))
        sg.append(pooled_sigma2(unf, LCHK))
    print(f"null for singlet E = {E} (10 samples): <s^2> = {np.mean(s2):.3f}, Sigma^2 = {np.round(np.mean(sg, axis=0), 3)}")

# control for the reference direction: chiral GOE spectra of its shapes (rows = twisted states, columns = distinct
# values; bidiagonal model of a real Gaussian matrix) through the same pipeline, degrees as for the data
from scipy.linalg import eigvalsh_tridiagonal

NT = {899: 1511, 451: 917, 1366: 2392, 3536: 7056, 9208: 19856, 6137: 13289, 3079: 6667}


def chiral_goe(nt, n, rng):
    d = np.sqrt(rng.chisquare(np.arange(nt, nt - n, -1)))
    e = np.sqrt(rng.chisquare(np.arange(n - 1, 0, -1)))
    return np.sqrt(np.clip(eigvalsh_tridiagonal(d ** 2 + np.concatenate([[0.0], e ** 2]), d[:-1] * e), 0, None))


rng = np.random.default_rng(20261116)
for E in (6, 7, 8):
    sizes = [np.unique(np.round(b, 9)).size for b in dir1_blocks(E)]
    sg, s2 = [], []
    for _ in range(16):
        unf = [unfold(chiral_goe(NT[n], n, rng))[0] for n in sizes]
        sg.append(pooled_sigma2(unf, LCHK))
        s2.append(np.mean(np.concatenate([spacings(x) for x in unf]) ** 2))
    print(f"reference direction, E = {E}: degrees {[degree(n) for n in sizes]}; chiral GOE control (16 samples): <s^2> = {np.mean(s2):.3f}, "
          f"Sigma^2 = {np.round(np.mean(sg, axis=0), 3)} +- {np.round(np.std(sg, axis=0), 3)}")

# ---------------------------------------------------------------- figure (two columns)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

plt.rcParams["path.simplify"] = False   # older matplotlib otherwise visibly kinks the Sigma^2 curves in the PDF
FIG_W, FIG_H = 7.0, 2.25                   # inches; \textwidth of revtex prl is 7.06 in (510 pt)
X0, PW, GAP, Y0, PH = 0.45, 2.50, 0.60, 0.40, 1.76   # narrower panels: the legend sits to their right
LEG_X = X0 + 2 * PW + GAP + 0.12           # left edge of the legend column (in)
fig = plt.figure(figsize=(FIG_W, FIG_H))
axl = fig.add_axes([X0 / FIG_W, Y0 / FIG_H, PW / FIG_W, PH / FIG_H])
axr = fig.add_axes([(X0 + PW + GAP) / FIG_W, Y0 / FIG_H, PW / FIG_W, PH / FIG_H])
TXT, TICK, LAB, LEG = 8.5, 8, 9.5, 9         # curve labels, tick labels, axis labels, legend


def multicolor_math(ax, x, y, s, colors, size):
    """Mathtext s centered at data point (x, y) (baseline), one color per glyph, glyph positions from the layout
    of the whole string (so the spacing is exactly that of a single text)."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import text_to_path
    from matplotlib.transforms import Affine2D, ScaledTranslation
    prop = FontProperties(size=size)
    glyph_info, glyph_map, _ = text_to_path.get_glyphs_mathtext(prop, s)
    assert len(glyph_info) == len(colors), (len(glyph_info), len(colors))
    k = size / text_to_path.FONT_SCALE
    paths = []
    for gid, xpos, ypos, scale in glyph_info:
        v, c = glyph_map[gid]
        paths.append(Path((v * scale + [xpos, ypos]) * k, c))
    x0 = min(q.vertices[:, 0].min() for q in paths); x1 = max(q.vertices[:, 0].max() for q in paths)
    tr = (Affine2D().translate(-(x0 + x1) / 2, 0).scale(1 / 72) + ax.figure.dpi_scale_trans
          + ScaledTranslation(x, y, ax.transData))
    for q, col in zip(paths, colors):
        ax.add_artist(PathPatch(q, transform=tr, fc=col, ec="none", lw=0))

# left: P(s) at E = 8
ss = np.linspace(0.0, 3.6, 400)
bins = np.arange(0.0, 3.61, 0.1)
axl.hist(s8_d1, bins=bins, density=True, histtype="stepfilled", color="#cde2fb", lw=0)
axl.hist(s8_d1, bins=bins, density=True, histtype="step", color=BLUE, lw=0.9)
axl.hist(s8_sg, bins=bins, density=True, histtype="step", color=GREEN, lw=1.0)
axl.plot(ss, p_goe(ss), color=INK, lw=1.0)
axl.plot(ss, p_gue(ss), color=GRAY, lw=0.9, ls=(0, (3, 1.6)))
axl.plot(ss, np.exp(-ss), color=INK2, lw=0.8, ls=(0, (5, 1.5, 1, 1.5)))
axl.set_xlim(0, 3.6)
axl.set_ylim(0, 1.02)
axl.set_xticks([0, 1, 2, 3])
axl.yaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
axl.yaxis.set_major_formatter(FixedFormatter(["0", "0.5", "1"]))
axl.set_xlabel(r"unfolded spacing $s$", labelpad=2, fontsize=LAB)
axl.set_ylabel(r"$P\hspace{0.1}(s)$", labelpad=3, fontsize=LAB)
axl.text(1.0, 0.145, r"$E=8$", ha="center", va="center", fontsize=TXT)   # inside the bell, below the Poisson curve

# right: number variance
shades_d1 = {6: "#9cc3f0", 7: "#5c9be3", 8: BLUE}
SG8 = "#1a865e"                             # GREEN darkened by 15% for the singlet at E = 8
shades_sg = {7: GREEN_LIGHT, 8: SG8}
for E in (6, 7, 8):
    axr.plot(LS, sig_d1[E], color=shades_d1[E], lw=1.1)
for E in SG_LEVELS:
    axr.plot(LS, sig_sg[E], color=shades_sg[E], lw=1.1)
axr.plot(LS, sigma2_goe(LS), color=INK, lw=1.0)
axr.plot(LS, sigma2_gue(LS), color=GRAY, lw=0.9, ls=(0, (3, 1.6)))
axr.plot(LS, LS, color=INK2, lw=0.8, ls=(0, (5, 1.5, 1, 1.5)))
axr.set_xlim(0, 10)
axr.set_yscale("log")
axr.set_ylim(0.1, 10)
axr.set_xticks([0, 2, 4, 6, 8, 10])
axr.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
axr.yaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10"]))
axr.yaxis.set_minor_formatter(NullFormatter())
axr.set_xlabel(r"interval length $L$", labelpad=2, fontsize=LAB)
axr.set_ylabel(r"$\Sigma^2(L)$", labelpad=3, fontsize=LAB)
multicolor_math(axr, 7.4, 1.24, r"$E=6,7,8$", [BLUE, BLUE, shades_d1[6], BLUE, shades_d1[7], BLUE, shades_d1[8]], TXT)
axr.text(8.75, 4.2, r"$E=8$", color=SG8, fontsize=TXT, ha="center", va="center")
axr.text(1.25, 4.0, r"$E=7$", color=GREEN_LIGHT, fontsize=TXT, ha="center", va="center")

for a in (axl, axr):
    a.tick_params(labelsize=TICK, pad=2)

# inset: the three reference-direction curves at L >= 4 on a linear axis, labelled by E at their right ends
ins = axr.inset_axes([0.46, 0.075, 0.40, 0.235])   # top below the GUE curve; bottom raised for the tick labels
for E in (6, 7, 8):
    ins.plot(LS, sig_d1[E], color=shades_d1[E], lw=1.1)
ins.plot(LS, sigma2_goe(LS), color=INK, lw=1.0)
ins.set_xlim(4, 10)
ins.set_ylim(0.72, 1.12)
ins.set_xticks([4, 7, 10])                 # a few small ticks, so that the inset can be read
ins.set_yticks([0.8, 1.0])
ins.tick_params(labelsize=6.5, length=2, width=0.5, pad=1.2, direction="in", colors=INK2)
for sp in ins.spines.values():
    sp.set_linewidth(0.5)
    sp.set_color(INK2)
    sp.set_visible(True)
ylab = {E: float(sig_d1[E][-1]) for E in (6, 7, 8)}        # labels at the curve ends, at least GAPLAB apart
GAPLAB = 0.08
order = sorted(ylab, key=ylab.get)
for _ in range(50):
    for lo, hi in zip(order[:-1], order[1:]):
        if ylab[hi] - ylab[lo] < GAPLAB:
            mid = (ylab[hi] + ylab[lo]) / 2
            ylab[lo], ylab[hi] = mid - GAPLAB / 2, mid + GAPLAB / 2
for E in (6, 7, 8):
    ins.text(10.2, ylab[E], rf"${E}$", color=shades_d1[E], fontsize=7.5,
             ha="left", va="center", clip_on=False)
print("inset: Sigma2(10) =", {E: round(float(sig_d1[E][-1]), 3) for E in (6, 7, 8)}, "GOE", round(float(sigma2_goe(LS)[-1]), 3))

# legend in one column to the right of both panels, centred on them
hd = [plt.Line2D([], [], color=BLUE, lw=1.2), plt.Line2D([], [], color=GREEN, lw=1.2),
      plt.Line2D([], [], color=INK, lw=1.0), plt.Line2D([], [], color=GRAY, lw=0.9, ls=(0, (3, 1.6))),
      plt.Line2D([], [], color=INK2, lw=0.8, ls=(0, (5, 1.5, 1, 1.5)))]
fig.legend(hd, ["reference\ndirection", "singlet", "GOE", "GUE", "Poisson"], loc="center left",
           bbox_to_anchor=(LEG_X / FIG_W, (Y0 + PH / 2) / FIG_H), ncol=1, frameon=False, fontsize=LEG,
           handlelength=2.0, handletextpad=0.5, labelspacing=0.9, borderaxespad=0.0, borderpad=0.0)
r = fig.canvas.get_renderer()
bb = [a.get_tightbbox(r) for a in (axl, axr)] + [fig.legends[0].get_window_extent(r)]
print("extents (in): " + ", ".join(f"[{b.x0 / fig.dpi:.3f}, {b.x1 / fig.dpi:.3f}] x [{b.y0 / fig.dpi:.3f}, {b.y1 / fig.dpi:.3f}]" for b in bb))
save(fig, __file__, "fig_spacings_singlet_twocol")
