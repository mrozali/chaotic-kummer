#!/usr/bin/env python3
"""Fig. 1 of the Letter (fig:spectrum): the scalar spectrum near the Kummer point and its first-order deformation.

Left panel: the massive primaries of the invariant sector at lambda = 0, a comb of exactly degenerate blocks at integer E
(sizes d_+ + d_-), and their first-order deformation along the reference direction.  Each block splits into the band
E +- lambda nu_i, with nu_i the non-zero singular values of the fully resolved first-order sub-blocks, while its
first-order kernel stays at E.  The shading is the density of first-order levels, normalized per block.  The bands are
faded beyond lambda = 1/(2 nu_max), where they reach halfway to their neighbors.

Right panel: direction dependence, exact at first order.  The first-order matrix is linear in the direction,
M_c = sum_mu q_mu M_mu with c = q0 + i(q1 sx + q2 sy + q3 sz).  For one E = 3 sub-block (trivial character of the
stabilizer of the reference direction, Z_R = -1: 25 untwisted and 49 twisted primaries, 25 non-zero singular values) the
four M_mu are built with the modules of code/kummer (r2c_space, r2g_generic_probe), and the singular values are followed
along the great circle q(theta) = cos(theta) q_s + sin(theta) u from the singlet q_s = (0, 0, -1, 0) through the
reference direction (theta = 40.6 deg) to the mirror q2 = 0 (theta = 90 deg).  The stabilizer is the same (order 16) at
every theta > 0 tested, so the sub-block is the same along the path.  At theta = 0 the singlet's larger stabilizer
(order 384) makes the values degenerate.

Checks (asserted, then printed): the values at the reference direction equal the stored sub-block E3_leaf0 of
Generic_firstorder_qgen_L7.npz, the values at the singlet lie in the singlet's E = 3 spectrum
(Singlet_firstorder_L8.npz), and the stabilizer is unchanged at theta = 5, 60 and 89 deg.

Inputs:  data/r2_out/Generic_firstorder_qgen_L7.npz and .json, data/r2_out/Singlet_firstorder_L8.npz, code/kummer.
Output:  code/figure-reproduction/generated/fig_spectrum_direction.pdf and .png, compared with
         paper/fig_spectrum_direction.pdf.  About 10 s.

Run from the repository root:  .venv/bin/python code/figure-reproduction/fig01_spectrum_direction.py
"""
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figstyle import ROOT, R2, BLUE, ORANGE, INK, INK2, GREEN, save   # noqa: E402  (also sets the rcParams)
plt.rcParams["axes.labelsize"] = 10           # axis labels 9 -> 10 pt

sys.path.insert(0, os.path.join(ROOT, "code", "kummer"))
import r2c_space as sp                  # noqa: E402  (free-field spaces, chiral twist maps)
import r2g_generic_probe as gp          # noqa: E402  (raw first-order block, stabiliser, isotypic bases, extra Z2's)

# ---------------------------------------------------------------- data for (a)
fo = np.load(os.path.join(R2, "Generic_firstorder_qgen_L7.npz"), allow_pickle=False)
with open(os.path.join(R2, "Generic_firstorder_qgen_L7.json")) as f:
    census = json.load(f)["blocks"]


def singular_values(E):
    """All first-order singular values at level E, over the fully resolved sub-blocks."""
    keys = [k for k in fo.files if k.startswith(f"E{E}_leaf")]
    return np.concatenate([fo[k] for k in keys])


def block_size(E):
    b = census[str(E)]
    return b["d_plus"] + b["d_minus"]


# ---------------------------------------------------------------- data for (b): exact first order along a great circle
TH = 3
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
QBASIS = [np.eye(2, dtype=complex), 1j * SX, 1j * SY, 1j * SZ]


def c_of(q):
    """c = q0 + i(q1 sx + q2 sy + q3 sz), the paper's eq:quaternion (unit q: the code's normalisation)."""
    return sum(qi * B for qi, B in zip(q, QBASIS))


def direction_block():
    """The four matrices M_mu of the E = 3 sub-block (trivial character of H_c, Z_R = -1) in reference direction's basis."""
    S = sp.Space(TH, mode="prim", verbose=False)
    q1 = np.random.default_rng(2026).standard_normal(4)
    q1 /= np.linalg.norm(q1)
    assert np.allclose(c_of(q1), gp.cmat("qgen")) and np.allclose(c_of([0, 0, -1, 0]), gp.cmat("singlet"))
    Hc, sgn, _ = gp.stabiliser(S, c_of(q1), TH)
    chars, dims, elem_class, phys, _ = gp.subgroup_characters(S.g, Hc)
    cls = gp.classify_ops(gp.raw_extra_ops(S, TH), gp.M_c_raw(S, TH, c_of(q1)))
    syms = {nm: (v[0], v[1]) for nm, v in cls.items() if v[2] != 0}
    names = sorted(syms)
    bases, _, irr = gp.isotypic_bases(S, TH, Hc, sgn, chars, elem_class, dims, phys, lambda s: None)
    r0 = [r for r in irr if dims[r] == 1 and np.allclose(chars[r], 1)][0]
    parts = gp.split_joint(bases[(r0, "u")], bases[(r0, "t")], syms, names, np.random.default_rng(17))
    labs = [lb for lb, (qu, qt) in parts.items() if lb[names.index("Z_R")] == -1 and qu.shape[1] and qt.shape[1]]
    assert len(labs) == 1, labs
    qu, qt = parts[labs[0]]
    print(f"sub-block: H_c of order {len(Hc)}, extra Z2's {names} = {labs[0]}, {qu.shape[1]} untwisted, "
          f"{qt.shape[1]} twisted")
    return S, q1, Hc, sgn, [qt.conj().T @ gp.M_c_raw(S, TH, B) @ qu for B in QBASIS]


S, Q1, HC, SGN, MS = direction_block()
QS = np.array([0.0, 0.0, -1.0, 0.0])                       # the singlet
U = Q1 - (Q1 @ QS) * QS
U /= np.linalg.norm(U)
TH1 = float(np.degrees(np.arccos(Q1 @ QS)))                # reference direction, 40.6 degrees from the singlet
THETA = np.linspace(0.0, 90.0, 1801)


def q_at(t):
    return np.cos(np.radians(t)) * QS + np.sin(np.radians(t)) * U


def svals(q):
    return np.sort(np.linalg.svd(sum(qi * m for qi, m in zip(q, MS)), compute_uv=False))


SV = np.array([svals(q_at(t)) for t in THETA])

# checks
s1 = np.unique(np.round(svals(Q1), 9))
ref1 = np.unique(np.round(fo["E3_leaf0"], 9))
assert s1.size == ref1.size and np.max(np.abs(s1 - ref1)) < 1e-7, "reference direction does not reproduce E3_leaf0"
zs = np.load(os.path.join(R2, "Singlet_firstorder_L8.npz"), allow_pickle=False)
sing3 = np.concatenate([zs[k] for k in zs.files if re.fullmatch(r"B_k\d+[pm]?_E3", k)])
vals0, mult0 = np.unique(np.round(SV[0], 7), return_counts=True)
assert all(np.min(np.abs(sing3 - v)) < 1e-6 for v in vals0), "singlet values not in the singlet's E = 3 spectrum"
for t in (5.0, 60.0, 89.0):
    H, sg, _ = gp.stabiliser(S, c_of(q_at(t)), TH)
    assert set(H) == set(HC) and all(sg[i] == SGN[i] for i in HC), f"stabiliser changes at theta = {t}"
print(f"reference direction at theta = {TH1:.2f} deg reproduces E3_leaf0 ({s1.size} values); "
      f"H_c unchanged at theta = 5, 60, 89 deg")
print("singlet (theta = 0): values", np.round(vals0, 4).tolist(), "multiplicities", mult0.tolist())
print("u =", np.round(U, 4).tolist(), "; rank at theta = 90:", int(np.sum(SV[-1] > 1e-9 * SV[-1].max())))

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(6.5, 3.0))                 # no panel titles: the axes keep their size
axa = fig.add_axes([0.075, 0.137, 0.535, 0.84])
axb = fig.add_axes([0.705, 0.137, 0.28, 0.84])

# ---- (a) comb and first-order bands (as in fig_spectrum_deformation.py)
LMAX, XL = 0.12, -0.03
YMIN, YMAX = -0.45, 6.62
NX, NY = 1500, 1200
xs = np.linspace(0.0, LMAX, NX)
ys = np.linspace(YMIN, YMAX, NY)
dy = ys[1] - ys[0]
alpha = np.zeros((NY, NX))
lstar = {}
for E in range(1, 8):            # E = 7 only enters at the top edge
    sv = singular_values(E)
    smax = sv.max()
    lstar[E] = 1.0 / (2.0 * smax)
    slopes = np.concatenate([sv, -sv])
    cnt = np.zeros((NY, NX))
    for j0 in range(0, NX, 150):
        lam = xs[j0:j0 + 150]
        fy = (E + np.outer(slopes, lam) - YMIN) / dy
        i0 = np.floor(fy).astype(int)
        fr = fy - i0
        cols = np.broadcast_to(np.arange(j0, j0 + lam.size), i0.shape)
        for ii, ww in ((i0, 1.0 - fr), (i0 + 1, fr)):
            ok = (ii >= 0) & (ii < NY)
            np.add.at(cnt, (ii[ok], cols[ok]), ww[ok])
    # density relative to a uniform spread of the band's levels over its rows
    rows = np.maximum(2.0 * smax * xs, dy) / dy
    rho = cnt / (slopes.size / rows)[None, :]
    a = 1.0 - np.exp(-rho)
    fade = np.clip(1.0 - (xs - lstar[E]) / (0.4 * lstar[E]), 0.0, 1.0)
    alpha = np.maximum(alpha, a * fade[None, :])
img = np.ones((NY, NX, 3)) * (1.0 - alpha[..., None]) + np.array(mc.to_rgb(BLUE)) * alpha[..., None]
axa.imshow(img, extent=(0, LMAX, YMIN, YMAX), origin="lower", aspect="auto",
           interpolation="antialiased", zorder=1)

for E in range(1, 7):
    ls_ = lstar[E]
    if ls_ < LMAX:
        axa.plot([0, ls_], [E, E], color=ORANGE, lw=1.0, solid_capstyle="butt", zorder=3)
        seg = np.linspace(ls_, min(LMAX, 1.4 * ls_), 16)
        for i in range(15):
            axa.plot(seg[i:i + 2], [E, E], color=ORANGE, lw=1.0, alpha=1 - (i + 0.5) / 15,
                     solid_capstyle="butt", zorder=3)
    else:
        axa.plot([0, LMAX], [E, E], color=ORANGE, lw=1.0, zorder=3)
    # the comb at lambda = 0
    axa.plot([XL + 0.004, -0.004], [E, E], color=INK2, lw=1.8, solid_capstyle="butt", zorder=3)
    n = block_size(E)
    txt = f"{n:,}".replace(",", r"\,")
    axa.text(XL / 2, E + 0.11, rf"${txt}$", ha="center", va="bottom", fontsize=7.5, color=INK2)
axa.plot([XL + 0.004, -0.004], [0, 0], color=INK2, lw=1.8, solid_capstyle="butt", zorder=3)
axa.plot([0, LMAX], [0, 0], color=INK, lw=1.0, zorder=3)
axa.text(LMAX - 0.001, 0.1, "protected ground state", ha="right", va="bottom", fontsize=8.5)
# E = 0: the single bare primary kept there (the protected Ramond vacuum), labelled like the blocks above it
axa.text(XL / 2, 0 + 0.11, r"$1$", ha="center", va="bottom", fontsize=7.5, color=INK2)

axa.set_xlim(XL, LMAX)
axa.set_ylim(YMIN, YMAX)
axa.set_yticks(range(7))
axa.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
axa.spines["bottom"].set_bounds(0, LMAX)
axa.set_xlabel(r"coupling $\lambda$")
axa.set_ylabel(r"$E=h+\bar h$")
handles = [Line2D([], [], color=BLUE, lw=1.2),
           Line2D([], [], color=ORANGE, lw=1.2)]
axa.legend(handles, [r"$E\pm\lambda\nu_i$", "first-order kernel"],
           loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=8.5, frameon=False,
           handlelength=1.6, handletextpad=0.6, borderaxespad=0.1, labelspacing=0.35)

# ---- (b) the 25 first-order shifts of one E = 3 sub-block along the great circle singlet -> reference direction -> q2 = 0
YB = 8.6
axb.plot(THETA, SV, color=BLUE, lw=0.6, zorder=2)
axb.axvline(TH1, color=INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
axb.plot(np.zeros(vals0.size), vals0, ls="", marker="D", ms=3.2, color=GREEN, mew=0, clip_on=False, zorder=4)
axb.text(TH1 + 1.5, YB - 0.1, "reference direction", ha="left", va="top", fontsize=8, color=INK2)
# small arrows from the two labels to the vertical loci theta = 0 (singlet) and theta = 90 deg (q2 = 0)
ARW = dict(arrowstyle="-|>", mutation_scale=5, lw=0.5, color=INK2, shrinkA=1.0, shrinkB=0.0)
axb.annotate("singlet", xy=(0.0, 0.42), xytext=(7.5, 0.42), ha="left", va="center", fontsize=8, color=INK2,
             arrowprops=ARW)
axb.annotate(r"$q_2=0$", xy=(90.0, 7.75), xytext=(82.5, 7.75), ha="right", va="center", fontsize=8, color=INK2,
             arrowprops=ARW)
axb.set_xlim(0, 90)
axb.set_ylim(0, YB)
axb.set_xticks([0, 30, 60, 90])
axb.set_xticklabels([r"$0\!^\circ$", r"$30\!^\circ$", r"$60\!^\circ$", r"$90\!^\circ$"])
axb.set_yticks(range(0, 9, 2))
axb.set_xlabel(r"direction angle $\theta$")
axb.set_ylabel(r"first-order shift $\nu_i$ at $E=3$")

print("couplings 1/(2 nu_max) beyond which the bands are faded, by level:", {E: round(v, 4) for E, v in lstar.items()})
save(fig, __file__, "fig_spectrum_direction")
