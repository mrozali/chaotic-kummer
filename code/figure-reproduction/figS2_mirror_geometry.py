#!/usr/bin/env python3
"""Fig. S2 of the Supplemental Material (fig:mirrorgeometry): mirrors and chambers on a two-sphere of directions, one
dimension below the three-sphere of unit quaternions.

The nine mirror planes x_i = 0 and x_i = +-x_j of the symmetry group of a cube meet the two-sphere in nine great
circles, which cut it into 48 spherical triangles (one shaded).  The dots are the units, where four mirrors meet.  The
open circle is a generic direction, at the angle delta from the nearest mirror.  On the three-sphere the sixteen
mirrors q_mu = 0 and q_mu = +-q_nu are great two-spheres, which cut it into 384 spherical tetrahedra.  Orthographic
projection, with the far side of the sphere dashed.

No data files are read.
Output:  code/figure-reproduction/generated/fig_walls_geometry.pdf and .png, compared with
         paper/fig_walls_geometry.pdf.

Run from the repository root:  .venv/bin/python code/figure-reproduction/figS2_mirror_geometry.py
"""
import itertools
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import BLUE, INK, GRAY, save

FILL = "#cde2fb"                                         # blue ramp, step 100, as in the other figures
R_LIM = 1.15

fig = plt.figure(figsize=(6.5, 2.05))                     # full text width, so the fonts match the other figures
ax = fig.add_axes([0.2428, 0.0, 0.3154, 1.0])           # a square box, 2.05in on a side; sphere and legend centred
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_xlim(-R_LIM, R_LIM)
ax.set_ylim(-R_LIM, R_LIM)


def unit(*x):
    x = np.array(x, float)
    return x / np.linalg.norm(x)


# ---------------------------------------------------------------- the 2-sphere
v = unit(0.78, 0.5, 0.42)                                # towards the viewer
up = unit(*(np.array([0.0, 0.0, 1.0]) - v[2] * v))
right = np.cross(up, v)
proj = lambda P: np.stack([P @ right, P @ up], axis=-1)
ax.add_patch(Circle((0, 0), 1, fc="white", ec=GRAY, lw=0.8, zorder=0))


def slerp(p, q, n=60):
    om = np.arccos(np.clip(p @ q, -1, 1))
    t = np.linspace(0, 1, n)[:, None]
    return (np.sin((1 - t) * om) * p + np.sin(t * om) * q) / np.sin(om)


E = np.eye(3)
coord = [E[i] for i in range(3)]
diag = [unit(*(E[i] + s * E[j])) for i, j in itertools.combinations(range(3), 2) for s in (1, -1)]
assert len(coord) + len(diag) == 9
for walls, col, lw in ((coord, INK, 1.0), (diag, BLUE, 0.9)):
    for n in walls:
        a = unit(*np.cross(n, [0.3, 0.7, 0.2]))
        b = np.cross(n, a)
        t = np.linspace(0, 2 * np.pi, 1441)[:, None]
        P = np.cos(t) * a + np.sin(t) * b
        front = P @ v >= 0
        xy = proj(P)
        for mask, style in ((front, dict(color=col, lw=lw, zorder=3)),
                            (~front, dict(color=col, lw=0.5, alpha=0.35, ls=(0, (2, 2)), zorder=1))):
            seg = np.where(mask, 1.0, np.nan)[:, None] * xy
            ax.plot(seg[:, 0], seg[:, 1], **style)

A, B, C = E[2], unit(1, 0, 1), unit(1, 1, 1)             # the corners of one chamber
tri = np.vstack([slerp(A, B), slerp(B, C), slerp(C, A)])
assert (tri @ v > 0).all()
ax.add_patch(Polygon(proj(tri), closed=True, fc=FILL, ec="none", zorder=2))
for P in E:                                              # the visible units
    if P @ v > 0:
        ax.plot(*proj(P), "o", ms=5.0, color=INK, mew=0, zorder=6)
G = unit(0.62, 0.30, 0.95)                               # a generic direction and its nearest wall
n = min(coord + diag, key=lambda m: abs(m @ G))
F = unit(*(G - (n @ G) * n))
ax.plot(*proj(slerp(G, F, 20)).T, color=INK, lw=0.9, zorder=7)
ax.plot(*proj(G), "o", ms=5.0, mfc="white", mec=INK, mew=1.0, zorder=8)
gx, gy = proj(G)
fx, fy = proj(F)
ax.text((gx + fx) / 2 + 0.01, (gy + fy) / 2 + 0.035, r"$\delta$", fontsize=8, ha="center", va="bottom", zorder=9)
print("generic direction: nearest wall", np.round(n, 3), f"delta = {np.degrees(np.arcsin(abs(n @ G))):.1f} deg")

# ---------------------------------------------------------------- legend, on the right
hs = [Line2D([], [], color=INK, lw=1.0, marker="o", ms=4.0, mew=0),
      Line2D([], [], color=BLUE, lw=0.9, marker="o", ms=4.0, mew=0)]
fig.legend(hs, [r"mirrors $x_i=0$", r"mirrors $x_i=\pm x_j$"], loc="center left", bbox_to_anchor=(0.5757, 0.5),
           ncol=1, frameon=False, fontsize=8, handlelength=1.6, labelspacing=0.9)

save(fig, __file__, "fig_walls_geometry")
