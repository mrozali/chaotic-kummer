#!/usr/bin/env python3
"""Fig. 2 of the Letter (fig:firstorderE): the symmetry-resolved first-order gap ratio.

Left panel, <r> against the level E, every sub-block fully resolved:
  reference direction   E = 3..6 from the leaves of Generic_firstorder_qgen_L6.json.  E = 7 from the four sub-blocks of
                        Generic_probe_qgen_E7_mf_values.npz.  E = 8 from the three sub-blocks without a hidden Z_2
                        (Generic_probe_qgen_E8_mf_values.npz) and the two eigenspaces of the hidden Z_2 of the trivial
                        character, with 6137 and 3079 values (Hidden_firstorder_E8_values.npz).  The union of the two
                        eigenspaces equals the stored unresolved sub-block (asserted to 2e-9), and the six values round
                        to the Letter's table (asserted).
  20 random directions  qrand101 to qrand120: q = normalized N(0,1)^4 from numpy's default_rng(seed) with seeds 101 to
                        120, no selection.
                        E = 3..6, qrand101-116: the leaves of the truncated-Hamiltonian runs
                        (scan_L6/Generic_firstorder_<dir>_L6.json).
                        E = 3, 5, qrand117-120: the direction scan (Wall_scan_E3.json, Wall_scan_E5.json).
                        E = 4, 6, qrand117-120: the scan's three non-trivial characters pooled with the two eigenspaces
                        of the hidden Z_2 found by r2g_hidden_lean.py (Hidden_lean_<dir>_L6.json).
                        E = 7, 8: the lean probe (Generic_probe_<dir>_E7_lean_values.npz and ..._E8_lean_values.npz), at
                        E = 8 with the two eigenspaces of Hidden_firstorder_E8_values.npz.
                        The sub-block shapes of all 20 directions equal those of the reference direction (asserted), so
                        one band applies to all of them.  This draw lies closer to the mirrors than is typical: the
                        median angle to the nearest mirror is 0.98 deg, against 1.73 deg for the uniform measure.
  singlet               E = 5..8, Singlet_firstorder_L8.json, resolution B (every known symmetry resolved).
  bands                 mean and standard deviation of <r> for chiral GOE matrices of matched shapes:
                        Generic_firstorder_null_n3000.json (3000 samples per level) and
                        Singlet_firstorder_L8_null_v3.json.

Right panel, <r> at E = 5 against the angle to the nearest of the 16 mirrors (logarithmic axis): the reference direction
and the same 20 random directions (disks), and 20 directions placed at chosen angles from a single mirror (squares).
The placed directions come from r2g_wall_scan_targets.py (Wall_scan_targets20c.json) and were run with
r2g_wall_scan.py --qfile (Wall_scan_E5_targeted20c.json).  They are not a random sample: they fill the range of small
angles, where random directions are rare.  Filled and open markers: the nearest mirror is q_mu = 0 or q_mu = +-q_nu.
On a mirror q_mu = 0 all four sub-blocks of E = 5 split into two independent spectra (the two-block value R_TWO).  On a
mirror q_mu = +-q_nu two of them split and the other two are mapped onto each other, so next to such a mirror <r> tends
to (R_GOE + R_TWO)/2 as the angle goes to zero (dash-dotted line).

Inputs:  the files named above, all in data/r2_out.
Output:  code/figure-reproduction/generated/fig_firstorder_vs_E_sample.pdf and .png, compared with
         paper/fig_firstorder_vs_E_sample.pdf.  A few seconds.

Run from the repository root:  .venv/bin/python code/figure-reproduction/fig02_gap_ratios.py
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.colors import to_rgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import R2, BLUE, INK, INK2, GRAY, LIGHT, SHADE, GREEN, R_GOE, R_POI, R_TWO, gap_ratios, save  # noqa: E402

E_RIGHT = 5
N_RANDOM, N_PLACED = 20, 20


def _tint(c, f):
    return tuple(1 - f * (1 - x) for x in to_rgb(c))


GREEN_SHADE, GREEN_LINE = _tint(GREEN, 0.22), _tint(GREEN, 0.5)
SHADE = "#eae9e5"       # grey bands slightly stronger than figstyle's SHADE "#f0efec"


def load(name):
    return os.path.join(R2, name)


def leaves(fname, levels):
    d = json.load(open(load(fname)))["leaves"]
    return {E: (d[str(E)]["r_mean"], d[str(E)]["r_sem"]) for E in levels}


def pooled(fname, keys):
    z = np.load(load(fname), allow_pickle=False)
    g = np.concatenate([gap_ratios(np.unique(np.round(z[k], 9))) for k in keys])
    return float(g.mean()), float(g.std() / np.sqrt(g.size)), g.size


# ---------------------------------------------------------------- left panel
# E = 7, 8: the stored sub-blocks pooled for every generic direction (at E = 8 the trivial irrep keeps a hidden Z_2,
# whose two eigenspaces are added by pooled_e8)
KEYS = {7: ("rho0_labpm", "rho2_labpm", "rho3_labpm", "rho6_labpm"), 8: ("rho2_labpp", "rho3_labpp", "rho6_labpp")}
HID8 = json.load(open(load("Hidden_firstorder_E8.json")))               # code/kummer/r2g_hidden_firstorder.py
assert all(HID8["gates"][k] for k in ("G1_leaves_stable", "G2_block_diagonal", "G3_common_null_space", "G4_stored_values"))
assert HID8["leaf_sizes"] == [6137, 3079] and HID8["rank"] == 9216
HV8 = np.load(load("Hidden_firstorder_E8_values.npz"), allow_pickle=False)


def pooled_e8(name, fname):
    """E = 8 with all five sub-blocks: the three stored ones and the two eigenspaces of the hidden Z_2 of the
    trivial-character sub-block, whose union is the stored unresolved sub-block."""
    z = np.load(load(fname), allow_pickle=False)
    lv = [HV8[f"{name}__leaf{i}"] for i in range(2)]
    assert [x.size for x in lv] == HID8["leaf_sizes"], name
    assert np.abs(np.sort(np.concatenate(lv)) - np.unique(np.round(z["rho0_labpp"], 9))).max() < 2e-9, name
    g = np.concatenate([gap_ratios(np.unique(np.round(z[k], 9))) for k in KEYS[8]] + [gap_ratios(x) for x in lv])
    return float(g.mean()), float(g.std() / np.sqrt(g.size)), g.size


dir1 = leaves("Generic_firstorder_qgen_L6.json", (3, 4, 5, 6))
m1, e1, n1 = pooled("Generic_probe_qgen_E7_mf_values.npz", KEYS[7])
dir1[7] = (m1, e1)
dir1[8] = pooled_e8("qgen", "Generic_probe_qgen_E8_mf_values.npz")[:2]
TABLE = {3: 0.541, 4: 0.471, 5: 0.542, 6: 0.532, 7: 0.528, 8: 0.530}     # the letter's table, reference direction
assert all(abs(dir1[E][0] - TABLE[E]) < 6e-4 for E in TABLE), dir1

sB = json.load(open(load("Singlet_firstorder_L8.json")))["B"]
singlet = {E: (sB[str(E)]["r_mean"], sB[str(E)]["r_sem"]) for E in (5, 6, 7, 8)}
s5 = json.load(open(load("Generic_probe_singlet_L5.json")))["levels"]["5"]
assert abs(singlet[5][0] - s5["r_mean"]) < 1e-12

n300 = json.load(open(load("Generic_firstorder_null_n3000.json")))["levels"]      # 3000 samples per level
assert all(n300[str(E)]["nsamp"] == 3000 for E in range(3, 9))
assert sorted(sh[1] for sh in n300["8"]["shapes"]) == sorted([9208] * 3 + HID8["leaf_sizes"])
null = {E: (n300[str(E)]["mean"], n300[str(E)]["sd"]) for E in range(3, 9)}
ns3 = json.load(open(load("Singlet_firstorder_L8_null_v3.json")))["levels"]
null_singlet = {E: (ns3[str(E)]["nt_x_distinct"]["mean"], ns3[str(E)]["nt_x_distinct"]["sd"]) for E in (5, 6, 7, 8)}

SAMPLE = ["qrand%d" % k for k in range(101, 101 + N_RANDOM)]
SEPT, LEAN = SAMPLE[:16], SAMPLE[16:]          # September Fir leaves / r2g_hidden_lean.py
shape1 = {E: sorted((r["nt"], r["rank"]) for r in json.load(open(load("Generic_firstorder_qgen_L6.json")))["leaves"][str(E)]
                    ["rows"] if r["rank"] > 0) for E in (3, 4, 5, 6)}
scanE = {E: {d["name"]: d for d in json.load(open(load(f"Wall_scan_E{E}.json")))["directions"]} for E in (3, 4, 5, 6)}
sample_lv = {}
for n in SEPT:
    d = json.load(open(load(os.path.join("scan_L6", f"Generic_firstorder_{n}_L6.json"))))["leaves"]
    sample_lv[n] = {E: d[str(E)]["r_mean"] for E in (3, 4, 5, 6)}
    for E in (3, 4, 5, 6):
        assert sorted((r["nt"], r["rank"]) for r in d[str(E)]["rows"] if r["rank"] > 0) == shape1[E], (n, E)
for n in LEAN:
    h = json.load(open(load(f"Hidden_lean_{n}_L6.json")))
    assert h["leaf_sizes"] == [2885, 1630], (n, h["leaf_sizes"])          # the hidden Z_2 split, as for qrand101-116
    sample_lv[n] = {E: h["r_full"][str(E)]["r_mean"] for E in (3, 4, 5, 6)}
    for E in (3, 4, 5, 6):
        sh = [(r["nt"], r["rank"]) for r in scanE[E][n]["rows"] if r["rank"] > 0]
        if E % 2 == 0:
            sh += [(h["leaf_rows"][t][str(E)]["nt"], h["leaf_rows"][t][str(E)]["rank"]) for t in h["leaf_tags"]]
        else:
            assert abs(sample_lv[n][E] - scanE[E][n]["r_mean"]) < 1e-12
        assert sorted(sh) == shape1[E], (n, E, sorted(sh), shape1[E])
h101 = json.load(open(load("Hidden_lean_qrand101_L6.json")))   # the lean code against the September leaves
assert h101["validation"]["ok"] and h101["leaf_sizes"] == [2885, 1630]
assert all(abs(h101["r_full"][str(E)]["r_mean"] - sample_lv["qrand101"][E]) < 1e-9 for E in (3, 4, 5, 6))
rows1 = {E: json.load(open(load(f"Generic_probe_qgen_E{E}_mf.json")))["rows"] for E in (7, 8)}
shape1.update({E: sorted((r["nt"], r["rank"]) for r in rows1[E] if r["rank"] > 0) for E in (7, 8)})
for n in SAMPLE:                                # E = 7, 8 from the lean probe (all four irreps run at both levels)
    for E in (7, 8):
        rec = json.load(open(load(f"Generic_probe_{n}_E{E}_lean.json")))
        rows = [r for r in rec["rows"] if r["rank"] > 0]
        assert sorted((r["nt"], r["rank"]) for r in rows) == shape1[E], (n, E)
        assert all(r["distinct"] == r["rank"] for r in rows), (n, E)    # no exact degeneracies
        m, e, ng = pooled(f"Generic_probe_{n}_E{E}_lean_values.npz", KEYS[E])
        ref = rec if E == 7 else rec["pooled_without_trivial_irrep"]
        assert ng == ref["n_gaps"] and abs(m - ref["r_mean"]) < 1e-9, (n, E, m, ref["r_mean"])
        if E == 8:                              # with the two eigenspaces of the hidden Z_2
            m = pooled_e8(n, f"Generic_probe_{n}_E{E}_lean_values.npz")[0]
        sample_lv[n][E] = m
for E in range(3, 9):
    v = np.array([sample_lv[n][E] for n in SAMPLE])
    m, sd = null[E]
    print(f"{N_RANDOM} directions, E={E}: median {np.median(v):.4f}, range {v.min():.4f}..{v.max():.4f}, "
          f"within 2 sd {int(np.sum(np.abs(v - m) <= 2 * sd))}")

# ---------------------------------------------------------------- right panel
refR = json.load(open(load("Wall_scan_ref.json")))[str(E_RIGHT)]
NB, NSPLIT_DIAG = refR["shape"][0], refR["n_split_diag"]
assert [tuple(s) for s in n300[str(E_RIGHT)]["shapes"]] == [tuple(refR["shape"][1:])] * NB     # same band
assert (NB, NSPLIT_DIAG) == (4, 2)              # E = 5: a wall q_mu = +-q_nu splits two of the four sub-blocks
R_DIAG0 = (R_GOE + R_TWO) / 2                   # so next to it <r> -> (GOE + two blocks) / 2 as delta -> 0
named = scanE[E_RIGHT]
assert all(d["generic"] for d in named.values())
sample_r = [named[n] for n in SAMPLE]
for n in SAMPLE:                                # the left panel's values reappear (odd level, same pooling)
    assert abs(named[n]["r_mean"] - sample_lv[n][E_RIGHT]) < 1e-9, n
assert abs(named["qgen"]["r_mean"] - dir1[E_RIGHT][0]) < 1e-9
MR, SDR = null[E_RIGHT]
# the 20-set after two swaps : near03_diag (0.0089 deg) -> near03b_diag (0.001 deg) gave 20b, then
# near06_coord (0.0266 deg) -> near06c_coord (3.8 deg) gave 20c
PLACED_SET = f"{N_PLACED}c"
placed = json.load(open(load(f"Wall_scan_E{E_RIGHT}_targeted{PLACED_SET}.json")))["directions"]
targets = {d["name"]: d for d in json.load(open(load(f"Wall_scan_targets{PLACED_SET}.json")))["directions"]}
assert len(placed) == N_PLACED and all(d["generic"] for d in placed)
for d in placed:                                # nearest wall and angle as constructed
    assert d["nearest"] == targets[d["name"]]["wall"] and abs(d["delta"] - targets[d["name"]]["delta"]) < 1e-6
print(f"{N_RANDOM} directions, E={E_RIGHT}: within 2 sd {sum(abs(d['r_mean'] - MR) <= 2 * SDR for d in sample_r)}; "
      f"angles {min(d['delta'] for d in sample_r):.4f}..{max(d['delta'] for d in sample_r):.2f} deg, "
      f"{sum(d['delta'] < 0.3 for d in sample_r)} below 0.3 deg")


near0 = {k: [d["r_mean"] for d in placed if targets[d["name"]]["kind"] == k and d["delta"] < 0.1] for k in ("coord", "diag")}
print(f"placed directions below 0.1 deg: wall q_mu=0 mean {np.mean(near0['coord']):.3f} over {len(near0['coord'])} "
      f"(two blocks {R_TWO:.3f}); wall q_mu=+-q_nu mean {np.mean(near0['diag']):.3f} over {len(near0['diag'])} "
      f"((GOE + two blocks)/2 = {R_DIAG0:.3f})")


# ---------------------------------------------------------------- figure
FIG_W, FIG_H = 7.647, 3.0                       # full text width
AX_L, AX_B, AX_H = 0.6, 0.42, 2.49              # inches
AX_W, GAP, BX_W = 4.487, 0.25, 2.22             # right margin 0.09 in
SPREAD_F = 1.2                                  # points and bands of a level spread 1.2 times further
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([AX_L / FIG_W, AX_B / FIG_H, AX_W / FIG_W, AX_H / FIG_H])
bx = fig.add_axes([(AX_L + AX_W + GAP) / FIG_W, AX_B / FIG_H, BX_W / FIG_W, AX_H / FIG_H], sharey=ax)
XMIN, XMAX = 2.65, 8.42
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(0.28, 0.585)

NULL_L, NULL_R = 0.21 * SPREAD_F, 0.15 * SPREAD_F                     # generic band from E - NULL_L to E + NULL_R
for E, (m, s) in null.items():
    ax.add_patch(Rectangle((E - NULL_L, m - s), NULL_L + NULL_R, 2 * s, color=SHADE, lw=0, zorder=0))
    ax.plot([E - NULL_L, E + NULL_R], [m, m], color=LIGHT, lw=0.8, zorder=1, solid_capstyle="butt")
SING_X, SING_W = 0.235 * SPREAD_F, 0.09 * SPREAD_F
for E, (m, s) in null_singlet.items():
    ax.add_patch(Rectangle((E + SING_X - SING_W / 2, m - s), SING_W, 2 * s, color=GREEN_SHADE, lw=0, zorder=0))
    ax.plot([E + SING_X - SING_W / 2, E + SING_X + SING_W / 2], [m, m], color=GREEN_LINE, lw=0.8, zorder=1,
            solid_capstyle="butt")

DASH = (0, (5, 1.5, 1, 1.5))
for a in (ax, bx):
    for y, ls in ((R_GOE, "-"), (R_TWO, DASH), (R_POI, DASH)):
        a.axhline(y, color=INK2, lw=0.7, ls=ls, zorder=1)
# the dash-dotted lines labelled under the line , as in the right panel: above "two GOE blocks" the E = 8
# column has a near-wall direction at <r> = 0.437
for y, lab, below in ((R_GOE, "GOE", False), (R_TWO, "two GOE blocks", True), (R_POI, "Poisson", True)):
    ax.text(XMAX - 0.03, y - 0.0045 if below else y + 0.0035, lab, ha="right", va="top" if below else "bottom",
            fontsize=7, color=INK2)

EB = dict(capsize=0, elinewidth=0.8, zorder=3)
# offsets from E, centred on the grey band (centre BAND_C): at every level the reference direction and the four columns
# of random directions (cycled in the order of the values) as one group
BAND_C = (NULL_R - NULL_L) / 2
GROUP = np.array([-0.09, -0.045, 0.005, 0.055, 0.105]) * SPREAD_F
GROUP = GROUP - (GROUP[0] + GROUP[-1]) / 2 + BAND_C
SPREAD = tuple(GROUP[1:])
X_REF = [E + GROUP[0] for E in dir1]
ax.errorbar(X_REF, [v[0] for v in dir1.values()], fmt="o", ms=4.0, color=BLUE, mew=0, **EB)
for E in range(3, 9):
    for i, n in enumerate(sorted(SAMPLE, key=lambda n: sample_lv[n][E])):
        ax.plot([E + SPREAD[i % 4]], [sample_lv[n][E]], ls="", marker="o", ms=2.9, mfc=GRAY, mec="white", mew=0.35,
                zorder=3)
ax.errorbar(np.array(list(singlet)) + SING_X, [v[0] for v in singlet.values()], fmt="D", ms=3.4, color=GREEN,
            mew=0, **EB)
ax.set_xticks(range(3, 9))
ax.set_xlabel(r"level $E$")
ax.set_ylabel(r"first-order $\langle r\rangle$, fully resolved")
handles = [Line2D([], [], ls="", marker="o", ms=4.0, color=BLUE, mew=0),
           Line2D([], [], ls="", marker="o", ms=2.9, mfc=GRAY, mec="white", mew=0.35),
           Rectangle((0, 0), 1, 1, color=SHADE, lw=0),
           Line2D([], [], ls="", marker="D", ms=3.4, color=GREEN, mew=0),
           Rectangle((0, 0), 1, 1, color=GREEN_SHADE, lw=0)]
labels = [r"reference direction", rf"{N_RANDOM} random directions",
          "chiral GOE, generic shapes", "singlet", "chiral GOE, singlet's shapes"]
leg = ax.legend(handles, labels, loc="lower right", ncol=1, frameon=True, fancybox=False, framealpha=1.0,
                facecolor="white", edgecolor="none", handlelength=1.2, handletextpad=0.5, labelspacing=0.45,
                borderpad=0.35, borderaxespad=0.0, fontsize=7.5)
leg.set_zorder(2)

XL, XR = 6e-4, 10.0
bx.set_xscale("log")
bx.set_xlim(XL, XR)
bx.axhspan(MR - SDR, MR + SDR, color=SHADE, lw=0, zorder=0)
bx.axhline(R_DIAG0, color=INK2, lw=0.7, ls=DASH, zorder=1)    # same style as the GOE / two-block / Poisson lines
# label under the line at its right edge: (GOE + two / GOE blocks)/2, the stack in parentheses at 6 pt and "/2" at
# the 7 pt of the other line labels; it stays clear of the markers just below the line
LAB_STACK = TextArea(r"$\genfrac{(}{)}{0}{0}{\mathrm{GOE} + \mathrm{two}}{\mathrm{GOE\ blocks}}$",
                     textprops=dict(fontsize=6, color=INK2))
LAB_HALF = TextArea(r"$/2$", textprops=dict(fontsize=7, color=INK2))
bx.add_artist(AnnotationBbox(HPacker(children=[LAB_STACK, LAB_HALF], align="center", pad=0, sep=0),
                             (0.985, R_DIAG0 - 0.0035), xycoords=bx.get_yaxis_transform(), box_alignment=(1, 1),
                             frameon=False, pad=0))
# the random directions as grey disks like the left panel's dots (filled: with their white rim), open for a nearest
# wall q_mu = +-q_nu; sized like the squares (apparent diameter about 3.05 pt filled, 3.7 pt open)
DISK_F = dict(ls="", marker="o", ms=3.4, mfc=GRAY, mec="white", mew=0.35)
DISK_O = dict(ls="", marker="o", ms=3.0, mfc="white", mec=GRAY, mew=0.7)
for d in sample_r:
    x = max(d["delta"], 1.1 * XL)
    if d["nearest"].endswith("=0"):
        bx.plot([x], [d["r_mean"]], zorder=3, **DISK_F)
    else:
        bx.plot([x], [d["r_mean"]], zorder=3, **DISK_O)
for d in placed:
    if targets[d["name"]]["kind"] == "coord":
        bx.plot([d["delta"]], [d["r_mean"]], ls="", marker="s", ms=3.0, color=GRAY, mew=0, zorder=3)
    else:
        bx.plot([d["delta"]], [d["r_mean"]], ls="", marker="s", ms=2.9, mfc="white", mec=GRAY, mew=0.7, zorder=3)
bx.plot([named["qgen"]["delta"]], [named["qgen"]["r_mean"]], ls="", marker="o", ms=4.0, color=BLUE, mew=0, zorder=4)
bx.text(0.04, 0.965, rf"$E={E_RIGHT}$, direction dependence", transform=bx.transAxes, ha="left", va="top",
        fontsize=7.5, color=INK)
bx.set_xlabel(r"angle to the nearest mirror (deg)")
bx.set_xticks([1e-3, 1e-2, 1e-1, 1, 10])
bx.set_xticklabels(["0.001", "0.01", "0.1", "1", "10"])
bx.tick_params(axis="y", labelleft=False)
# two entries, each with the disk and the square of that kind of wall (random and placed directions alike)
bh = [(Line2D([], [], **DISK_F), Line2D([], [], ls="", marker="s", ms=3.0, color=GRAY, mew=0)),
      (Line2D([], [], **DISK_O), Line2D([], [], ls="", marker="s", ms=2.9, mfc="white", mec=GRAY, mew=0.7))]
bl = [r"nearest mirror $q_\mu=0$", r"nearest mirror $q_\mu=\pm q_\nu$"]
leg2 = bx.legend(bh, bl, loc="lower right", ncol=1, frameon=True, fancybox=False, framealpha=1.0, facecolor="white",
                 edgecolor="none", handlelength=2.2, handletextpad=0.5, labelspacing=0.45, borderpad=0.35,
                 borderaxespad=0.0, fontsize=7, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)})
leg2.set_zorder(2)

save(fig, __file__, "fig_firstorder_vs_E_sample")
