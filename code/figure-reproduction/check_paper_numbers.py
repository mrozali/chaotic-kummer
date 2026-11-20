#!/usr/bin/env python3
"""Check the numbers quoted in the Letter and its Supplemental Material against the stored records.

Most checks read the arrays of data/r2_out and recompute the quoted number with the code of checklib.py, which is
independent of the pipeline modules and of the figure scripts.  Section (F) compares the E = 4 sentences with the stored
record of r2g_E4_deficit.py, and the column sigma comes from the stored chiral GOE samples.  Section (J) counts the
mirrors, chambers and group orders quoted in the text.  The GOE, GUE, Poisson and two-GOE reference values of the gap
ratio (0.5307, 0.5996, 0.3863, 0.423) are taken from the literature cited in the Letter and are not recomputed here.  The values of the table in the End Matter are parsed
from paper/chaotic-kummer-v1.tex.  The other paper values are written next to each check with their place in the text.
A value passes when it rounds to the digits printed in the paper.

Sections: (A) the table of the End Matter, (B) the Results paragraph, (C) the hidden Z_2, (D) the singlet,
(E) rank bound and zero modes, (F) the E = 4 sentences, (G) reference direction and Fig. 1, (H) unfolded spectra,
(I) the points of Fig. 2, which the figure script takes from stored summaries, recomputed from the arrays,
(J) counts of mirrors, chambers and group elements.
The two-direction computation of the hidden Z_2, the chiral GOE reference, the E = 4 analysis and the unfolding control
have their own scripts, which rerun the computation (see README.md in this folder).

Run from the repository root:  .venv/bin/python code/figure-reproduction/check_paper_numbers.py     (about 12 s)
"""
import os
import re

import numpy as np

from checklib import (ROOT, R2, R_GOE, RANDOM20, J, Z, Report, distinct, gap_ratios, mirror_angle, number_variance,
                      pooled, sigma2_goe, unfold)

rep = Report("Numbers of the Letter and the Supplemental Material, recomputed from data/r2_out")

L7j, L7 = J("Generic_firstorder_qgen_L7.json"), Z("Generic_firstorder_qgen_L7.npz")
E7 = Z("Generic_probe_qgen_E7_mf_values.npz")
E8j, E8 = J("Generic_probe_qgen_E8_mf.json"), Z("Generic_probe_qgen_E8_mf_values.npz")
H8j, H8 = J("Hidden_firstorder_E8.json"), Z("Hidden_firstorder_E8_values.npz")
NULL = J("Generic_firstorder_null_n3000.json")["levels"]
SING, SINGj = Z("Singlet_firstorder_L8.npz"), J("Singlet_firstorder_L8.json")
K8 = ("rho2_labpp", "rho3_labpp", "rho6_labpp")                    # E = 8: the characters without a hidden Z_2


def ref_blocks(E):
    """Fully resolved sub-blocks of the reference direction at level E."""
    if E <= 7:
        return [L7[k] for k in sorted(L7.files) if k.startswith("E%d_leaf" % E)]
    return [E8[k] for k in K8] + [H8["qgen__leaf0"], H8["qgen__leaf1"]]


def singlet_blocks(res, E):
    """Sub-blocks of the singlet at level E: res = 'B' with the hidden Z_2 resolved, 'A' without."""
    return [SING[k] for k in SING.files if re.fullmatch(r"%s_k\d+[pm]?_E%d" % (res, E), k)]


# ---------------------------------------------------------------------------------------------- (A) table
rep.section("(A) Table of the End Matter: d_+, d_-, rank, <r>_ref, sigma")
tex = open(os.path.join(ROOT, "paper", "chaotic-kummer-v1.tex")).read()
table = {}
for line in tex.split("\n"):
    cells = [c.replace("\\;", "").replace("$", "").replace("\\\\", "").strip() for c in line.split("&")]
    if len(cells) == 6 and cells[0] in list("345678"):
        table[int(cells[0])] = cells[1:]
rep.check("the table has the rows E = 3..8", sorted(table) == [3, 4, 5, 6, 7, 8], str(sorted(table)))
zscore = {}
for E in sorted(table):
    dp, dm, rank, rref, sigma = table[E]
    if E <= 7:
        dims = (L7j["blocks"][str(E)]["d_plus"], L7j["blocks"][str(E)]["d_minus"])
    else:
        dims = (E8j["d_plus"], E8j["d_minus"])
    blocks = ref_blocks(E)
    r, sem, ng = pooled(blocks)
    samples = np.array(NULL[str(E)]["samples"])
    sd = samples.std(ddof=1)
    zscore[E] = (r - R_GOE) / sd
    rep.check("E = %d: d_+, d_- (stored integers)" % E, dims == (int(dp), int(dm)), "%d, %d" % dims)
    rep.integer("E = %d: rank = number of distinct non-zero values" % E, sum(distinct(b).size for b in blocks), int(rank))
    rep.value("E = %d: <r>_ref over %d ratios" % (E, ng), r, rref)
    rep.value("E = %d: sigma over %d chiral GOE samples" % (E, samples.size), sd, sigma)
    cols = sorted(s[1] for s in NULL[str(E)]["shapes"])
    rep.check("E = %d: columns of the null matrices = distinct values per sub-block" % E,
              cols == sorted(distinct(b).size for b in blocks), str(cols))
dev = max(float(np.max(np.abs(np.sort(distinct(a)) - np.sort(distinct(b))))) for a, b in
          zip(sorted([L7[k] for k in L7.files if k.startswith("E7_leaf")], key=len), sorted([E7[k] for k in E7.files], key=len)))
rep.check("E = 7: the sub-blocks of the truncated-Hamiltonian run equal those of the matrix-free probe", dev <= 2e-9,
          "largest difference %.1e" % dev)
rep.check("3000 realizations per level (End Matter)", all(len(NULL[str(E)]["samples"]) == 3000 for E in table))
rows8 = sorted(tuple(s) for s in NULL["8"]["shapes"])[:2]
rep.check("E = 8: twisted states of the trivial character divided in proportion to 6137 : 3079",
          rows8 == [(6667, 3079), (13289, 6137)] and 6667 + 13289 == H8j["nt"], "%s of %d" % (rows8, H8j["nt"]))
alt = np.array(J("Generic_firstorder_null_n3000.json")["rows_check"]["samples"])
rep.value("E = 8: sigma with another division of the twisted states (%d samples)" % alt.size, alt.std(ddof=1), table[8][4])

# ---------------------------------------------------------------------------------------------- (B) Results
rep.section("(B) Results paragraph of the Letter")
zmax = max(abs(z) for E, z in zscore.items() if E != 4)
rep.info("(<r>_ref - 0.5307)/sigma for E = 3..8: " + ", ".join("%+.2f" % zscore[E] for E in sorted(zscore)))
rep.check("every level except E = 4 within 1.4 sigma of the GOE value", zmax < 1.45, "largest %.2f" % zmax)
rep.check("E = 4 deviates significantly", abs(zscore[4]) > 3, "%.2f sigma" % zscore[4])
merged6 = np.concatenate([L7["E6_leaf1"], L7["E6_leaf2"]])
rep.value("E = 6 before the hidden Z_2 is resolved", pooled([merged6] + [L7[k] for k in ("E6_leaf4", "E6_leaf6", "E6_leaf8")])[0], "0.509")
rep.value("E = 8, trivial character before the hidden Z_2 is resolved", gap_ratios(distinct(H8["qgen__unresolved"])).mean(), "0.434")
rep.value("E = 8, first eigenspace", gap_ratios(distinct(H8["qgen__leaf0"])).mean(), "0.536")
rep.value("E = 8, second eigenspace", gap_ratios(distinct(H8["qgen__leaf1"])).mean(), "0.531")
lean = Z("Generic_probe_qgen_E8_lean_rho2_values.npz")
dev = float(np.max(np.abs(distinct(lean["rho2_labpp"]) - distinct(E8["rho2_labpp"]))))
rep.check("E = 8: one sub-block recomputed with the lean probe equals the matrix-free record", dev <= 2e-9, "largest difference %.1e" % dev)
nrows = nbad = 0
for E in (3, 4, 5, 6):
    for d in J("Wall_scan_E%d.json" % E)["directions"]:
        for row in d["rows"]:
            nrows += 1
            nbad += row["rank"] != row["distinct"]
for n in RANDOM20:
    for E in (7, 8):
        for row in J("Generic_probe_%s_E%d_lean.json" % (n, E))["rows"]:
            nrows += 1
            nbad += row["rank"] != row["distinct"]
for E in range(1, 8):
    for row in L7j["leaves"][str(E)]["rows"]:
        nrows += 1
        nbad += row["rank"] != row["distinct"]
rep.check("non-zero singular values distinct in every stored sub-block of a generic direction", nbad == 0,
          "%d sub-block records, %d with a repeated value" % (nrows, nbad))

# ---------------------------------------------------------------------------------------------- (C) hidden Z2
rep.section("(C) The hidden Z_2 (End Matter)")
A6, B6 = distinct(L7["E6_leaf1"]), distinct(L7["E6_leaf2"])
U6 = np.sort(np.concatenate([A6, B6]))
rep.check("E = 6: 1350 values split into 899 and 451", (U6.size, A6.size, B6.size) == (1350, 899, 451), "%d = %d + %d" % (U6.size, A6.size, B6.size))
rep.value("E = 6: unresolved sub-block", gap_ratios(U6).mean(), "0.432")
rep.value("E = 6: first eigenspace", gap_ratios(A6).mean(), "0.524")
rep.value("E = 6: second eigenspace", gap_ratios(B6).mean(), "0.519")
rep.value("E = 6: pooled with the three other sub-blocks", pooled(ref_blocks(6))[0], "0.532")
rng = np.random.default_rng(6)                      # the recipe of fig04_hidden_z2.py
means = []
for _ in range(300):
    lev = []
    for n in (899, 451):
        h = rng.standard_normal((n, n))
        lev.append(np.linalg.eigvalsh((h + h.T) / np.sqrt(2 * n)))
    m = np.sort(np.concatenate(lev))
    k = int(0.05 * m.size)
    means.append(gap_ratios(m[k:m.size - k]).mean())
rep.value("two GOE spectra superposed 899 : 451, mean of <r> over 300 samples", np.mean(means), "0.441")
rep.value("two GOE spectra superposed 899 : 451, standard deviation of <r>", np.std(means, ddof=1), "0.009")
a8, b8 = distinct(H8["qgen__leaf0"]), distinct(H8["qgen__leaf1"])
rep.check("E = 8: 9216 values split into 6137 and 3079", (a8.size + b8.size, a8.size, b8.size) == (9216, 6137, 3079))
dev = np.max(np.abs(np.sort(np.concatenate([a8, b8])) - distinct(E8["rho0_labpp"])))
rep.check("E = 8: union of the eigenspaces = the unresolved sub-block of the earlier record", dev < 2e-9, "largest difference %.1e" % dev)
rep.check("E = 8: the gates of the two-direction computation passed", all(v is not False for v in H8j["gates"].values()), str(H8j["gates"]))
lab = np.concatenate([np.zeros(a8.size), np.ones(b8.size)])[np.argsort(np.concatenate([a8, b8]))]
close = np.argsort(np.diff(np.sort(np.concatenate([a8, b8]))))[:200]
rep.info("E = 8: of the 200 smallest spacings of the unresolved sub-block, %d join different eigenspaces" % int(np.sum(lab[close] != lab[close + 1])))
for E, ka, kb in ((4, "E4_leaf1", "E4_leaf2"), (6, "E6_leaf1", "E6_leaf2")):
    hz = Z("Hidden_firstorder_E%d_values.npz" % E)
    two = sorted([distinct(hz["qgen__leaf0"]), distinct(hz["qgen__leaf1"])], key=len)
    ham = sorted([distinct(L7[ka]), distinct(L7[kb])], key=len)
    dev = max(float(np.max(np.abs(x - y))) for x, y in zip(two, ham)) if [x.size for x in two] == [y.size for y in ham] else np.inf
    rep.check("E = %d: the two-direction method gives the same values in each eigenspace as the truncated Hamiltonian" % E, dev < 2e-9,
              "sizes %s, largest difference %.1e" % ([x.size for x in two], dev))
for E, sizes in ((4, [26, 50]), (6, [451, 899]), (8, [3079, 6137])):
    hz = Z("Hidden_firstorder_E%d_values.npz" % E)
    dirs = sorted({k.split("__")[0] for k in hz.files})
    same = all(sorted([distinct(hz[d + "__leaf0"]).size, distinct(hz[d + "__leaf1"]).size]) == sizes for d in dirs)
    rep.check("E = %d: the same split %d + %d along all %d directions of the record" % (E, sizes[1], sizes[0], len(dirs)), same)
worst = 0.0
for n in RANDOM20:                                  # two independent methods, 20 random directions
    hz, lz = Z("Hidden_firstorder_E6_values.npz"), Z("Hidden_lean_%s_L6_values.npz" % n)
    two = sorted([distinct(hz[n + "__leaf0"]), distinct(hz[n + "__leaf1"])], key=len)
    ham = sorted([distinct(lz["E6__b1+"]), distinct(lz["E6__b1-"])], key=len)
    worst = max(worst, max(float(np.max(np.abs(x - y))) for x, y in zip(two, ham)))
rep.check("E = 6: same values in each eigenspace by both methods, for each of the 20 random directions", worst < 2e-9, "largest difference %.1e" % worst)

# ---------------------------------------------------------------------------------------------- (D) singlet
rep.section("(D) The singlet direction (Results paragraph and End Matter)")
NMIN = 20          # as in the stored record, sub-blocks with fewer than 20 distinct values are left out of <r>
snull = J("Singlet_firstorder_L8_null_v3.json")["levels"]
sz = {}
for E in (5, 6, 7, 8):
    r = pooled(singlet_blocks("B", E), nmin=NMIN)[0]
    st = snull[str(E)]["nt_x_distinct"]
    sz[E] = (r - st["mean"]) / st["sd"]
    rep.check("E = %d: fully resolved <r> = %.4f lies below the GOE value" % (E, r), r < R_GOE, "%+.1f standard deviations of the matched null" % sz[E])
    rep.check("E = %d: equal to the stored summary" % E, abs(r - SINGj["B"][str(E)]["r_mean"]) < 1e-9)
rep.check("about two standard deviations below at E = 7, 8", all(-2.6 < sz[E] < -1.5 for E in (7, 8)), "%.1f, %.1f" % (sz[7], sz[8]))
rep.value("E = 6: hidden Z_2 resolved", pooled(singlet_blocks("B", 6), nmin=NMIN)[0], "0.486")
rep.value("E = 6: hidden Z_2 unresolved", pooled(singlet_blocks("A", 6), nmin=NMIN)[0], "0.465")
rep.value("E = 8: hidden Z_2 resolved", pooled(singlet_blocks("B", 8), nmin=NMIN)[0], "0.524")
rep.value("E = 8: hidden Z_2 unresolved", pooled(singlet_blocks("A", 8), nmin=NMIN)[0], "0.494")
tot = nd = at_pi = 0
for b in singlet_blocks("B", 8):
    v = np.asarray(b, dtype=float)
    v = np.round(v[v > 1e-9 * v.max()], 9)
    u, c = np.unique(v, return_counts=True)
    tot += v.size
    nd += u.size
    at_pi += int((c[np.abs(u - np.pi) < 1e-6] - 1).sum())
rep.integer("E = 8: non-zero values in the resolved sub-blocks", tot, 10721)
rep.integer("E = 8: values that repeat an earlier one (total minus distinct)", tot - nd, 583)
rep.check("E = 8: most of the repeats at the single value pi", at_pi > 0.5 * (tot - nd), "%d of %d" % (at_pi, tot - nd))
meta = SING["B_meta"]          # columns: k, rho, two signs, E, nu, nt, rank, distinct, k, tag index, sign of the hidden Z_2
for E in (6, 8):
    rows = meta[(meta[:, 4] == E) & (meta[:, 11] != 0)]
    plus, minus = rows[rows[:, 11] == 1], rows[rows[:, 11] == -1]
    per = [p[8] / m[8] for p in plus for m in minus if p[0] == m[0] and m[8]]
    rep.check("E = %d: the hidden Z_2 of the singlet splits %d sub-blocks, eigenspaces in a ratio of about two to one" % (E, len(per)),
              1.8 < plus[:, 8].sum() / minus[:, 8].sum() < 2.2,
              "distinct values %d : %d = %.2f in all, %.2f to %.2f in the single sub-blocks"
              % (plus[:, 8].sum(), minus[:, 8].sum(), plus[:, 8].sum() / minus[:, 8].sum(), min(per), max(per)))

# ---------------------------------------------------------------------------------------------- (E) rank bound
rep.section("(E) Rank bound and zero modes (End Matter)")
ratio = {}
for E, deficit in ((3, 0), (4, 1), (5, 0), (6, 84), (7, 0), (8, 57)):
    rows = L7j["leaves"][str(E)]["rows"] if E <= 7 else E8j["rows"]
    dtil = sum(r["nu"] for r in rows if r["nt"] > 0)       # untwisted states with Z_R = (-1)^E (stored integers)
    rank = sum(distinct(b).size for b in ref_blocks(E))
    dp = int(table[E][0])
    ratio[E] = int(table[E][1]) / dtil
    rep.integer("E = %d: untwisted states that can couple (%d) minus the rank" % (E, dtil), dtil - rank, deficit)
    if E % 2:
        rep.check("E = %d: these are half of the untwisted states" % E, 2 * dtil == dp, "%d of %d" % (dtil, dp))
rep.check("twisted states outnumber the untwisted states that couple to them by a factor of about two",
          all(1.7 < x < 2.6 for x in ratio.values()), ", ".join("%.2f" % ratio[E] for E in sorted(ratio)))

# ---------------------------------------------------------------------------------------------- (F) E = 4
rep.section("(F) The E = 4 sentences (End Matter), from the record of r2g_E4_deficit.py")
D4 = J("E4_deficit.json")
al = D4["all_samples"]
ndir = sum(s["n_directions"] for s in D4["samples"])
rep.check("3000 random directions more than 4 degrees from every mirror", ndir == 3000 and all(s["min_angle_deg"] == 4.0 for s in D4["samples"]), str(ndir))
rep.value("reference direction, pooled over %d ratios" % D4["reference"]["n_ratios"], D4["reference"]["pooled"], table[4][3])
rep.value("pooled gap ratio, mean over the directions", al["pooled_mean"], "0.495")
rep.value("three sub-blocks without hidden Z_2: the ten ratios at the top", al["clean_top10"], "0.40")
rep.value("three sub-blocks without hidden Z_2: the remaining ratios", al["clean_rest"], "0.53")
rep.value("hidden Z_2, eigenspace of 50 values", al["leaf50"], "0.45")
rep.value("hidden Z_2, eigenspace of 26 values", al["leaf26"], "0.45")
tp = D4["top_positions"]
rep.check("the highest level is isolated and the two below it are nearly degenerate",
          tp["mean_gap"][0] > 5 * tp["bulk_mean_gap"] and tp["mean_gap"][1] < tp["bulk_mean_gap"],
          "mean gaps %.3f and %.3f at the top, %.3f in the bulk" % (tp["mean_gap"][0], tp["mean_gap"][1], tp["bulk_mean_gap"]))
for k, v in D4["singlet_top"].items():
    t = np.array(v["singlet_top10"])
    sl = np.array(v["octet_slopes_over_c"])
    pattern = np.array([np.sqrt(2), 1, 1, 0, 0, -1, -1, -np.sqrt(2)])
    rep.check("singlet, %s: nondegenerate top level, eightfold level below, split into two levels and three pairs" % k,
              t[0] - t[1] > 0.1 and np.ptp(t[1:9]) < 1e-9 and t[8] - t[9] > 0.1 and np.max(np.abs(sl - pattern)) < 0.05,
              "top %.4f, eightfold %.4f = pi sqrt((7 + sqrt 33)/4) = %.4f" % (t[0], t[1], np.pi * np.sqrt((7 + np.sqrt(33)) / 4)))
ot = D4["other_levels"]
rep.check("no comparable structure at E = 3, 5, 6", all(ot[E]["top10"] > 0.48 for E in ("3", "5", "6")),
          ", ".join("E = %s: top ten %.3f, rest %.3f" % (E, ot[E]["top10"], ot[E]["rest"]) for E in ("3", "5", "6")))
ji = D4["joint_invariant_subspaces"]
rep.check("no further exact symmetry in the three sub-blocks (one joint invariant subspace each)",
          all(ji["others"][k] == [76] for k in K8), str({k: ji["others"][k] for k in K8}))

# ---------------------------------------------------------------------------------------------- (G) direction, Fig. 1
rep.section("(G) Reference direction (eq. qgen) and Fig. 1")
q = np.array([d for d in J("Wall_scan_E5.json")["directions"] if d["name"] == "qgen"][0]["q"])
draw = np.random.default_rng(2026).standard_normal(4)
rep.check("q_ref = (-0.318, 0.096, -0.760, 0.559)", np.round(q, 3).tolist() == [-0.318, 0.096, -0.76, 0.559], str(np.round(q, 6).tolist()))
rep.check("q_ref is the normalized draw of numpy's default_rng(2026)", np.max(np.abs(q - draw / np.linalg.norm(draw))) < 1e-12)
rep.value("angle from q_ref to the nearest mirror (degrees)", mirror_angle(q), "5.5")
sizes = [L7j["blocks"][str(E)]["d_plus"] + L7j["blocks"][str(E)]["d_minus"] for E in range(1, 7)]
rep.check("block sizes at E = 1..6 (Fig. 1, left)", sizes == [12, 93, 396, 1513, 5868, 20024], str(sizes))
rep.integer("non-zero shifts of the E = 3 sub-block of Fig. 1, right", distinct(L7["E3_leaf0"]).size, 25)

# ---------------------------------------------------------------------------------------------- (H) unfolding
rep.section("(H) Unfolded spectra (Letter, Wigner surmise and number variance; Supplemental Material, Sec. S4)")
LS = np.array([1.0, 2.0, 5.0, 10.0])
goe = sigma2_goe(LS)
s2 = {}
for E in (6, 7, 8):
    unf = [unfold(b) for b in ref_blocks(E)]
    rep.check("E = %d: every sub-block has a monotone fitted staircase" % E, all(u[2] for u in unf), "degrees %s" % [u[1] for u in unf])
    s2[E] = np.average([number_variance(u[0], LS) for u in unf], axis=0, weights=[distinct(b).size for b in ref_blocks(E)])
    rep.info("E = %d: Sigma^2 at L = 1, 2, 5, 10: %s  (GOE %s)" % (E, np.round(s2[E], 3), np.round(goe, 3)))
    if E == 8:
        rep.integer("E = 8: unfolded spacings of the reference direction, five sub-blocks", sum(u[0].size - 1 for u in unf), 33157)
rep.check("number variance follows the GOE up to L = 10 at E = 7 and 8", all(np.max(np.abs(s2[E] / goe - 1)) < 0.03 for E in (7, 8)),
          "largest relative deviation %.3f" % max(np.max(np.abs(s2[E] / goe - 1)) for E in (7, 8)))
rep.check("at E = 6 it rises above the GOE curve at larger L", s2[6][2] / goe[2] > 1.04 and s2[6][3] / goe[3] > 1.1 and abs(s2[6][1] / goe[1] - 1) < 0.02,
          "ratio to GOE %s" % np.round(s2[6] / goe, 3))
n6 = sorted(distinct(b).size for b in ref_blocks(6))
rep.check("E = 6: sub-blocks have between 451 and 1366 levels", (n6[0], n6[-1]) == (451, 1366), str(n6))
used = []
for b in singlet_blocks("B", 8):
    x = distinct(b)
    if x.size >= 100:
        xi, deg, mono = unfold(x)
        if mono:
            used.append((b, xi.size - 1))
rep.integer("singlet, E = 8: unfolded spacings from the sub-blocks with a monotone staircase", sum(n for _, n in used), 5704)
zero = total = 0
for b, _ in used:                                    # the same sub-blocks with every degenerate value kept
    v = np.sort(np.asarray(b, dtype=float))
    v = np.round(v[v > 1e-9 * v[-1]], 9)
    k = int(0.05 * v.size)
    s = np.diff(v[k:v.size - k])
    zero += int(np.sum(s == 0))
    total += s.size
rep.value("singlet, E = 8: percentage of exactly zero spacings when degenerate values are kept", 100.0 * zero / total, "9")

# ---------------------------------------------------------------------------------------------- (I) Fig. 2
rep.section("(I) Fig. 2: the plotted gap ratios recomputed from the arrays")
L6 = Z("Generic_firstorder_qgen_L6.npz")
dev = max(float(np.max(np.abs(distinct(L6[k]) - distinct(L7[k])))) for k in L6.files)
rep.check("reference direction, E <= 6: the run through E = 6 gives the same sub-blocks as the run through E = 7", dev <= 2e-9,
          "%d sub-blocks, largest difference %.1e" % (len(L6.files), dev))
scan = {E: (Z("Wall_scan_E%d_values.npz" % E), {d["name"]: d for d in J("Wall_scan_E%d.json" % E)["directions"]}) for E in (3, 4, 5, 6)}
worst = {E: 0.0 for E in range(3, 9)}
values = {E: [] for E in range(3, 9)}
for n in RANDOM20:
    hid, hidj = Z("Hidden_lean_%s_L6_values.npz" % n), J("Hidden_lean_%s_L6.json" % n)
    for E in (3, 4, 5, 6):
        arrays, summary = scan[E]
        blocks = [arrays[k] for k in arrays.files if k.startswith(n + "__")]
        if E % 2 == 0:
            blocks += [hid["E%d__b1+" % E], hid["E%d__b1-" % E]]
            stored = hidj["r_full"][str(E)]["r_mean"]
        else:
            stored = summary[n]["r_mean"]
        r = pooled(blocks)[0]
        values[E].append(r)
        worst[E] = max(worst[E], abs(r - stored))
        old = os.path.join("scan_L6", "Generic_firstorder_%s_L6.json" % n)
        if os.path.exists(os.path.join(R2, old)):
            worst[E] = max(worst[E], abs(r - J(old)["leaves"][str(E)]["r_mean"]))
    z7, j7 = Z("Generic_probe_%s_E7_lean_values.npz" % n), J("Generic_probe_%s_E7_lean.json" % n)
    r = pooled([z7[k] for k in z7.files if k.startswith("rho")])[0]
    values[7].append(r)
    worst[7] = max(worst[7], abs(r - j7["r_mean"]))
    z8 = Z("Generic_probe_%s_E8_lean_values.npz" % n)
    leaves = [H8[n + "__leaf0"], H8[n + "__leaf1"]]
    worst[8] = max(worst[8], float(np.max(np.abs(np.sort(np.concatenate([distinct(x) for x in leaves])) - distinct(z8["rho0_labpp"])))))
    values[8].append(pooled([z8[k] for k in K8] + leaves)[0])
for E in (3, 4, 5, 6, 7):
    rep.check("20 random directions, E = %d: <r> from the arrays equals the stored summaries" % E, worst[E] < 1e-8,
              "largest difference %.1e, values %.3f to %.3f" % (worst[E], min(values[E]), max(values[E])))
rep.check("20 random directions, E = 8: union of the two eigenspaces = the unresolved sub-block of the lean probe", worst[8] <= 2e-9,
          "largest difference %.1e, <r> over five sub-blocks %.3f to %.3f" % (worst[8], min(values[8]), max(values[8])))
pz, pj = Z("Wall_scan_E5_targeted20c_values.npz"), J("Wall_scan_E5_targeted20c.json")["directions"]
dev = max(abs(pooled([pz[k] for k in pz.files if k.startswith(d["name"] + "__")])[0] - d["r_mean"]) for d in pj)
ang = max(abs(mirror_angle(d["q"]) - d["delta"]) for d in pj)
rep.check("20 placed directions, E = 5: <r> and the angle to the nearest mirror equal the stored summaries", dev < 1e-8 and ang < 1e-6,
          "largest differences %.1e and %.1e degrees" % (dev, ang))

# ---------------------------------------------------------------------------------------------- (J) counts
rep.section("(J) Mirrors, chambers and group orders (Sec. II, End Matter, Supplemental Material Sec. S3)")
import itertools
signed = [(p, s) for p in itertools.permutations(range(4)) for s in itertools.product((1, -1), repeat=4)]
def det(p, s):
    m = np.zeros((4, 4))
    for i, (j, sg) in enumerate(zip(p, s)):
        m[i, j] = sg
    return round(np.linalg.det(m))
rep.integer("signed permutations of four coordinates", len(signed), 384)
rep.integer("of determinant one (the point group of the square torus)", sum(det(p, s) == 1 for p, s in signed), 192)
mirrors = [("q%d=0" % m, np.eye(4)[m]) for m in range(4)]
mirrors += [("q%d=%sq%d" % (m, "+" if s == 1 else "-", n), (np.eye(4)[m] - s * np.eye(4)[n]) / np.sqrt(2))
            for m in range(4) for n in range(m + 1, 4) for s in (1, -1)]
full = [nm for nm, _ in mirrors if "2" not in nm or nm == "q2=0"]
rep.integer("mirrors in all", len(mirrors), 16)
rep.integer("mirrors of the full theory (q2 = 0, and q_mu = 0, q_mu = +-q_nu with mu, nu in {0, 1, 3})", len(full), 10)
rep.integer("further mirrors q2 = +-q_mu of the invariant sector", len(mirrors) - len(full), 6)
singlet_q = np.array([0.0, 0.0, -1.0, 0.0])
rep.integer("mirrors through the singlet direction", sum(abs(n @ singlet_q) < 1e-12 for _, n in mirrors), 9)
rng = np.random.default_rng(1)
pts = rng.standard_normal((400000, 4))
keys = {(tuple(np.sign(x)), tuple(np.argsort(np.abs(x)))) for x in pts}
rep.integer("chambers of S^3 (sign pattern and ordering of |q_mu| of 400000 random directions)", len(keys), 384)
pts3 = rng.standard_normal((100000, 3))
rep.integer("spherical triangles cut by the nine mirrors of S^2 (Fig. S2)",
            len({(tuple(np.sign(x)), tuple(np.argsort(np.abs(x)))) for x in pts3}), 48)
rep.integer("equivalences of the full theory: signed permutations of (q0, q1, q3), times the reversal of q2", 2 ** 3 * 6 * 2, 96)
rep.integer("order of the stabilizer of a generic direction (record of the reference direction)", J("Generic_probe_qgen_E7_mf.json")["H_order"], 16)
rep.integer("order of the stabilizer of the singlet (the whole double cover)", J("Generic_probe_singlet_L5.json")["H_order"], 384)

rep.finish()
