"""Route 2 Phase C, GATE 19: the hidden Z_2 beyond (Gtilde, Pi G, Z_R).

Does NOT modify any existing r2c_*.py file.  Imports r2c_gate10 (build_subsectors, the
validated builder), r2c_gate11 (window/unlifted/LS, the validated definitions), r2c_build
(IrrepSector/SubSector), r2c_sym (side_op -- the one-sided Gtilde action, reused unmodified
to test candidate matches for S), r2c_space/r2c_group/r2c_irreps/r2c_proj (the underlying
machinery, read-only) and r2c_stats (analyse).

This module is a LOCAL re-implementation of `r2c_sym.split_all`'s bookkeeping (same logic,
same fixed seed 17, so it reproduces `r2c_gate10.build_subsectors`'s sub-sectors bit-for-bit
-- checked in `r2c_gate19_t1.py`, not included in this repository) that additionally keeps the embedding Qu/Qt of every
sub-sector basis vector back into the parent `IrrepSector`'s multiplicity basis, so that the
per-vector momentum-orbit provenance (level, orbit representative (a0,b0), stabiliser K) can
be recovered for task T2.
"""
import os
import numpy as np

import r2c_group as gr
import r2c_space as sp
import r2c_build as bd
import r2c_sym as sy
from r2c_build import SubSector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
FIG = os.path.join(os.path.dirname(HERE), 'figures')

TARGET_K = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
TARGET_RHO = [0, 2, 3, 5, 6]           # the rho's hosting the 10 flagged sub-sectors
LAM_T1 = [0.2, 0.35, 0.5]
PHASEC2_GRID = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.30, 0.50]


# =============================================================================
# 1. split_all, with provenance (Qu, Qt, and per-level orbit bookkeeping)
# =============================================================================
def nw_from_ab(a, b):
    """(n, w) target-space momentum/winding, from the LEFT/RIGHT chiral momentum pair
    (a, b) of an untwisted sector (f1mom_momentum.py, not included in this repository: at the self-dual square T^4,
    a = n + w, b = n - w, so n = (a+b)/2, w = (a-b)/2; a == b mod 2 guarantees these are
    integers).  Returns (n, w) as tuples of 4 ints."""
    n = tuple((x + y) // 2 for x, y in zip(a, b))
    w = tuple((x - y) // 2 for x, y in zip(a, b))
    return n, w


def member_weight_table(sec, th, coef_vec):
    """For an untwisted level th of IrrepSector `sec` and a coefficient vector `coef_vec`
    of length sec.mu[th] (the level's FULL local multiplicity dimension, offset already
    stripped), return {(a, b): weight} -- the squared Fock-space amplitude the state
    carries on EACH individual momentum-pair member (not grouped by orbit as in
    orbit_ranges/orbit_weight_table: every (a, b) key of sec.Bu[th] is used directly)."""
    out = {}
    for (a, b), (c0, blk) in sec.Bu[th].items():
        m = blk.shape[2]
        sub = coef_vec[c0:c0 + m]
        amp = np.tensordot(blk, sub, axes=([2], [0]))     # (na, nb)
        out[(a, b)] = float(np.sum(np.abs(amp) ** 2))
    return out


def orbit_ranges(sec, th):
    """For an untwisted level th of IrrepSector `sec`: group sec.Bu[th] by its c0 offset
    (all (a,b) keys sharing one c0 belong to the SAME Gtilde-orbit block, by construction
    of `IrrepSector._untw_basis`) and match each group against `sec.S.orbits[th]` to recover
    the orbit dict (rep=(a0,b0), members, coset, K).  Returns a sorted list of
    (c0, m, orbit_dict)."""
    B = sec.Bu[th]
    groups = {}
    for key, (c0, blk) in B.items():
        groups.setdefault(c0, {'m': blk.shape[2], 'keys': set()})
        groups[c0]['keys'].add(key)
    out = []
    for orb in sec.S.orbits[th]:
        mset = set(orb['members'])
        for c0, g in groups.items():
            if g['keys'] <= mset or g['keys'] == mset:
                out.append((c0, g['m'], orb))
                break
        else:
            if groups:
                # every member of this orbit had zero local Hom space (m=0): skip
                pass
    return sorted(out, key=lambda x: x[0])


def split_all_prov(sec, S, tol=1e-6):
    """Same partition as `r2c_sym.split_all(sec, S, tol)` (identical RNG seed/coefficients,
    so `parts` matches bit-for-bit), but also returns `prov[label]` = dict with:
        Qu   (sec.nu, n_u)   embedding of the untwisted sub-sector columns
        Qt   (sec.nt, n_t)   embedding of the twisted   sub-sector columns
        lev_u, lev_t         the level (bare th) of each column (matches SubSector.E0 order)
    """
    syms, info = sy.exact_symmetries(sec, S)
    names = sorted(syms)
    rng = np.random.default_rng(17)
    coef = rng.standard_normal(len(names)) if names else np.zeros(0)
    parts = {}
    for tag, off, mm, lev in (('u', sec.uoff, sec.mu, sec.ulev),
                              ('t', sec.toff, sec.mt, sec.tlev)):
        for th in lev:
            o, m = off[th], mm[th]
            ops = [(syms[nm][0] if tag == 'u' else syms[nm][1])[o:o + m, o:o + m]
                   for nm in names]
            if not ops:
                V = np.eye(m)
                labs = [()] * m
            else:
                Z = sum(c * (Y + Y.conj().T) / 2 for c, Y in zip(coef, ops))
                w, V = np.linalg.eigh(Z)
                labs = []
                for i in range(m):
                    v = V[:, i]
                    lab = tuple(int(np.sign(np.real(np.conj(v) @ (Y @ v)))) for Y in ops)
                    for Y in ops:
                        e = np.real(np.conj(v) @ (Y @ v))
                        assert abs(abs(e) - 1) < tol, e
                    labs.append(lab)
            for i in range(m):
                full = np.zeros(sec.nu if tag == 'u' else sec.nt, complex)
                full[o:o + m] = V[:, i]
                parts.setdefault(labs[i], {}).setdefault(tag, []).append((float(th), full))
    out = {}
    prov = {}
    for lab, d in parts.items():
        qu = d.get('u', [])
        qt = d.get('t', [])
        Qu = np.stack([x[1] for x in qu], axis=1) if qu else np.zeros((sec.nu, 0))
        Qt = np.stack([x[1] for x in qt], axis=1) if qt else np.zeros((sec.nt, 0))
        Eu = np.array([x[0] for x in qu])
        Et = np.array([x[0] for x in qt])
        ou = np.argsort(Eu, kind='stable') if len(Eu) else np.zeros(0, int)
        ot = np.argsort(Et, kind='stable') if len(Et) else np.zeros(0, int)
        Qu, Eu = Qu[:, ou], Eu[ou]
        Qt, Et = Qt[:, ot], Et[ot]
        M = np.conj(Qt.T) @ sec.M @ Qu
        out[lab] = SubSector(np.concatenate([Eu, Et]), M, len(Eu))
        prov[lab] = dict(Qu=Qu, Qt=Qt, lev_u=Eu, lev_t=Et, names=names)
    return names, out, prov


_SPACE_CACHE = {}
_IRREP_CACHE = {}


def get_space(th_max, mode='prim', verbose=True):
    key = (th_max, mode)
    if key not in _SPACE_CACHE:
        _SPACE_CACHE[key] = sp.Space(th_max, mode=mode, verbose=verbose)
    return _SPACE_CACHE[key]


def get_irrep(th_max, rho, mode='prim', verbose=True):
    key = (th_max, mode, rho)
    if key not in _IRREP_CACHE:
        S = get_space(th_max, mode, verbose=verbose)
        _IRREP_CACHE[key] = bd.IrrepSector(S, rho, verbose=verbose)
    return _IRREP_CACHE[key]


def target_map(th_max, mode='prim'):
    """{k: (rho, label)} for the 10 flagged sub-sector indices, using the SAME enumeration
    order as `r2c_gate10.build_subsectors` (rho ascending over physical irreps, label
    ascending within each rho)."""
    g = gr.G()
    out = {}
    k = 0
    for r in range(23):
        if not g.phys[r]:
            continue
        # we don't need to build the sector to know how many sub-labels it has for every
        # rho -- only for the TARGET_RHO's do we need the actual labels, but we must still
        # advance k correctly across every physical rho.  Build cheaply: n==0 sectors are
        # skipped entirely in build_subsectors, so we must know sec.n; unavoidably build.
        sec = get_irrep(th_max, r, mode, verbose=False)
        if sec.n == 0:
            continue
        names, parts, prov = split_all_prov(sec, get_space(th_max, mode, verbose=False))
        for lab in sorted(parts):
            if parts[lab].n:
                out[k] = (r, lab)
                k += 1
    return out


# =============================================================================
# 2. eigenvectors, unlifted mask, overlap-graph components
# =============================================================================
def eig_full(sub, lam):
    """(w, v) = eigh(H(lam)), ascending."""
    return np.linalg.eigh(sub.H(lam))


def unlifted_mask(ev, th_max, tol=1e-9):
    bare = np.arange(1, th_max + 1, dtype=float)
    return np.min(np.abs(ev[:, None] - bare[None, :]), axis=1) < tol


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.p[rx] = ry


def overlap_components(v0, v1, thresh=1e-20):
    """Bipartite overlap graph between the columns of v0 (n x n0) and v1 (n x n1), edge
    when |<i|j>|^2 > thresh.  Returns (O, comps) where O is the (n0,n1) overlap matrix and
    comps is a list of dict(idx0=[...], idx1=[...]) -- one entry per connected component
    that touches at least one node (isolated all-zero-overlap singles ARE their own
    1(+1)-node components here, since a state's self-identity edge is not assumed; if a
    lam0 state has literally zero overlap with every lam1 state, and vice versa, they end
    up as separate singleton components on each side)."""
    n0, n1 = v0.shape[1], v1.shape[1]
    O = np.abs(v0.conj().T @ v1) ** 2
    uf = UnionFind(n0 + n1)
    ii, jj = np.nonzero(O > thresh)
    for i, j in zip(ii, jj):
        uf.union(int(i), n0 + int(j))
    groups = {}
    for i in range(n0):
        groups.setdefault(uf.find(i), {'idx0': [], 'idx1': []})['idx0'].append(i)
    for j in range(n1):
        groups.setdefault(uf.find(n0 + j), {'idx0': [], 'idx1': []})['idx1'].append(j)
    comps = sorted(groups.values(), key=lambda g: -(len(g['idx0']) + len(g['idx1'])))
    return O, comps


def find_exact_degenerate_pairs(w, tol=1e-9):
    """Indices of exact degenerate pairs in sorted eigenvalues w (gap < tol).  Returns a
    list of (i, i+1) index pairs (non-overlapping: a run of k degenerate levels gives
    floor(k/2) pairs greedily) and the set of 'removed' indices (the second of each pair)."""
    n = len(w)
    pairs = []
    i = 0
    while i < n - 1:
        if abs(w[i + 1] - w[i]) < tol:
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1
    removed = set(j for (_, j) in pairs)
    return pairs, removed


# =============================================================================
# 3. S = P_A - P_B construction and characterisation helpers
# =============================================================================
def commutator_norm(X, Y):
    return float(np.max(np.abs(X @ Y - Y @ X)))


def project_from_indices(v, idx, n):
    if len(idx) == 0:
        return np.zeros((n, n), complex)
    Vsel = v[:, idx]
    return Vsel @ Vsel.conj().T


def auto_threshold(O, lo=-40, hi=-2, min_gap=0.3):
    """Automatic overlap threshold: histogram log10(|overlap|^2) over the WHOLE matrix,
    find the largest gap (in log-decades) between the floating-point noise floor and the
    smallest genuine nonzero overlap, and place the threshold at its midpoint.  Validated
    against the hand-tuned per-sub-sector thresholds of r2c_gate19_t1.py (not included in this repository) at
    th_max=6/7 (reproduces all known splits, including k=9/k=13's escalation, with NO per-case tuning,
    typically landing in a 4-12 decade wide gap) AND at th_max=8/Lambda=4
    (r2c_gate19_t5_L8, not included in this repository):
    on the larger, cluster-built matrices there the gap can be much narrower (~0.7-1.5 dex,
    e.g. rho=0 label=(1,1): a single dip of only 0.74 dex between the two clusters) --
    min_gap is set low enough to still find it.  A LOW min_gap risks picking up a spurious
    fluctuation in an otherwise-continuous (no real separation) distribution; that risk is
    caught downstream, not here: `find_split` only accepts the result if the resulting graph
    is EXACTLY two extensive components (or the k=1 pattern), which a meaningless small
    fluctuation in a genuinely single ergodic block essentially never produces (checked: the
    genuinely-irreducible Lambda=4 sub-sectors give a clean SINGLE component even down to
    min_gap=0).  Falls back to 1e-20 if no gap >= min_gap decades is found."""
    vals = np.log10(O[O > 0])
    vals = vals[(vals > lo) & (vals < hi)]
    if len(vals) < 2:
        return 1e-20, 0.0
    vs = np.sort(vals)
    gaps = np.diff(vs)
    i = int(np.argmax(gaps))
    if gaps[i] < min_gap:
        return 1e-20, float(gaps[i])
    mid = 0.5 * (vs[i] + vs[i + 1])
    return float(10 ** mid), float(gaps[i])


def threshold_stability(v0, v1, thresh, factor=3.0):
    """Check that the component partition at `thresh` matches the partition one factor up
    and one factor down (same sorted component sizes) -- a plateau/stability check for the
    auto-detected threshold, reported alongside (not gating) find_split's result."""
    _, c0 = overlap_components(v0, v1, thresh=thresh / factor)
    _, c1 = overlap_components(v0, v1, thresh=thresh)
    _, c2 = overlap_components(v0, v1, thresh=thresh * factor)
    s0 = sorted(len(c['idx0']) for c in c0)
    s1 = sorted(len(c['idx0']) for c in c1)
    s2 = sorted(len(c['idx0']) for c in c2)
    return s0 == s1 == s2


def plateau_threshold(v0, v1, lo=-20, hi=-2, step=0.5):
    """Robust threshold selection by PLATEAU SCAN rather than a single-largest-gap
    heuristic: auto_threshold's gap search (largest single jump between consecutive sorted
    log-overlaps) works when the noise floor -> signal transition is a sharp cliff, but on
    larger/more ill-conditioned matrices (checked at Lambda=4, r2c_gate19_t5_L8.py, not included in this repository) the
    transition can be DIFFUSE -- many small steps spread over several decades with no single
    dominant gap, e.g. rho=3 label=(1,1): the (0.2,0.35) overlap graph is stuck at 1 spurious
    component for every threshold up to 1e-15, then cleanly locks onto a STABLE 2-component
    split (1336, 718) for every threshold from 1e-14 to 1e-11 and beyond -- a wide, genuine
    plateau that auto_threshold's gap metric completely misses.  This scans a log-spaced
    threshold grid, finds the LONGEST run of constant (sorted component size) pattern, and
    returns its middle threshold plus the run's width in decades (an honest stability
    measure to report, analogous to auto_threshold's gap_dex)."""
    O = np.abs(v0.conj().T @ v1) ** 2
    n0, n1 = O.shape
    grid = np.arange(lo, hi + 1e-9, step)
    # ---- efficient incremental sweep: sort all edges once, add them (most permissive
    # threshold last) into ONE running union-find, and read off the component-size
    # pattern at each grid point -- O(n0*n1 log(n0*n1)) total instead of O(n0*n1) PER grid
    # point (that naive version timed out on the largest, Lambda=4 sub-sectors).
    ii, jj = np.nonzero(O > 10 ** lo)
    order = np.argsort(-O[ii, jj])          # descending overlap
    ii, jj, vals = ii[order], jj[order], O[ii, jj][order]
    uf = UnionFind(n0 + n1)

    def pattern_now():
        sizes = {}
        for a in range(n0 + n1):
            r = uf.find(a)
            sizes[r] = sizes.get(r, 0) + 1
        return tuple(sorted(sizes.values()))

    patterns = []
    k = 0
    m = len(vals)
    for lg in grid[::-1]:                    # descending threshold = most permissive last
        thr = 10 ** lg
        while k < m and vals[k] > thr:
            uf.union(int(ii[k]), n0 + int(jj[k]))
            k += 1
        patterns.append(pattern_now())
    patterns = patterns[::-1]                # restore ascending-threshold order (matches grid)
    # longest run of an identical pattern
    runs = []
    i = 0
    n = len(patterns)
    while i < n:
        j = i
        while j + 1 < n and patterns[j + 1] == patterns[i]:
            j += 1
        runs.append((patterns[i], i, j))
        i = j + 1
    # prefer the longest run whose pattern is a genuine (>=2 nontrivial pieces, or the k=1
    # singles pattern) split over a longest run that is just n_comp==1
    def is_split_pattern(patt):
        # a combined (both-sides) component of size <=2 is a 'single' -- either a genuine
        # size-1 boundary artifact or an isolated (1 lam=0.2 state, 1 lam=0.35 state) pair
        # like k=1's 8 isolated states (r2c_gate19_t1.py, not included in this repository) -- NOT a 'main' split block.
        mains = [s for s in patt if s > 2]
        n_singles = sum(1 for s in patt if s <= 2)
        return len(mains) == 2 or (len(mains) == 1 and n_singles >= 1)
    # a genuine split shows up as a WIDE, robust plateau (every case checked at th_max=6/7/8
    # is stable over several decades at least); a plateau of width 0 (a single grid point)
    # that happens to LOOK like a split pattern is a near-certain artifact of the extreme,
    # almost-fully-fragmented end of the scan (large threshold, few edges survive at all,
    # so a random leftover blob-plus-strays satisfies the pattern shape by chance) -- REQUIRE
    # at least min_width_dex of stability before accepting a split classification.
    # every GENUINE split found so far (th_max=6, 7, 8) is
    # stable over >=11.5 decades; a handful of SPURIOUS 'k1-like' detections (a giant blob
    # plus 1-2 leftover trivial nodes) turned up at the ragged, near-fully-fragmented end of
    # the scan with width exactly at a 1-1.5 dex boundary -- an enormous, deliberately
    # conservative margin below the genuine minimum.
    min_width_dex = 5.0
    split_runs = [r for r in runs
                 if is_split_pattern(r[0]) and (r[2] - r[1]) * step >= min_width_dex]
    if split_runs:
        best = max(split_runs, key=lambda r: r[2] - r[1])
    else:
        non_split = [r for r in runs if not is_split_pattern(r[0])]
        best = max(non_split, key=lambda r: r[2] - r[1]) if non_split else \
            max(runs, key=lambda r: r[2] - r[1])
    patt, i0, i1 = best
    mid = 0.5 * (grid[i0] + grid[i1])
    width_dex = (i1 - i0) * step
    return float(10 ** mid), float(width_dex), patt


def find_split(sub, th_max):
    """General-purpose 'is this SubSector further reducible' test (the method validated at
    th_max=6, 7 and refined with the plateau-scan threshold above):
    eigh at lambda=0.2/0.35/0.5, drop unlifted, build the (0.2, 0.35) overlap graph at the
    plateau-scanned threshold, and take connected components (2 'main' components, or the
    k=1 pattern: one main block plus >=1 exactly-isolated singles).  Returns
    (Sop_or_None, info_dict); Sop is None if the SubSector is already a single component
    (irreducible) or shows some other, unhandled component structure (reported via
    info['n_components'] regardless)."""
    data = {}
    for lam in LAM_T1:
        w, v = eig_full(sub, lam)
        m = ~unlifted_mask(w, th_max)
        data[lam] = dict(w=w[m], v=v[:, m])
    n_unlifted = int(len(sub.E0) - len(data[0.2]['w']))
    thresh_used, gap_dex, patt = plateau_threshold(data[0.2]['v'], data[0.35]['v'])
    O, comps = overlap_components(data[0.2]['v'], data[0.35]['v'], thresh=thresh_used)
    escalated = thresh_used > 1.5e-20
    stable = gap_dex >= 1.0
    singles = [c for c in comps if len(c['idx0']) == 1 and len(c['idx1']) == 1]
    mains = sorted([c for c in comps if c not in singles], key=lambda c: -len(c['idx0']))
    n = sub.n
    info = dict(n=n, n_unlifted=n_unlifted, thresh_used=thresh_used, escalated=escalated,
                gap_dex=gap_dex, stable=stable,
                n_components=len(comps), comp_sizes=[len(c['idx0']) for c in comps],
                n_singles=len(singles))
    Sop = None
    if len(mains) == 2:
        v02, v035 = data[0.2]['v'], data[0.35]['v']
        PA = project_from_indices(v02, mains[0]['idx0'], n)
        PB = project_from_indices(v02, mains[1]['idx0'], n)
        PA35 = project_from_indices(v035, mains[0]['idx1'], n)
        info['proj_dist_035'] = float(np.linalg.norm(PA - PA35))
        Sop = PA - PB
        info.update(comp_A=len(mains[0]['idx0']), comp_B=len(mains[1]['idx0']))
    elif len(mains) == 1 and singles:
        v02 = data[0.2]['v']
        idxA, idxB = mains[0]['idx0'], [c['idx0'][0] for c in singles]
        PA = project_from_indices(v02, idxA, n)
        PB = project_from_indices(v02, idxB, n)
        Sop = PA - PB
        info.update(comp_A=len(idxA), comp_B=len(idxB), pattern='k1-like')
    return Sop, info


def fully_resolve(sub, th_max, tag='', depth=0, max_depth=6, log=None):
    """Recursively apply find_split until every leaf SubSector is a single component (or
    the k1-like pattern with singles, which IS the fully resolved form for that block: the
    'singles' are already individually 1-dimensional, hence trivially irreducible).  Returns
    a list of (tag, SubSector, depth) leaves."""
    if depth > max_depth:
        if log is not None:
            log.append((tag, 'max depth reached, stopping', sub.n))
        return [(tag, sub, depth)]
    Sop, info = find_split(sub, th_max)
    if log is not None:
        log.append((tag, info, sub.n))
    if Sop is None:
        return [(tag, sub, depth)]
    sub_p, sub_m, B = build_split_subsectors(sub, Sop)
    if sub_p.n == 0 or sub_m.n == 0:
        return [(tag, sub, depth)]
    return (fully_resolve(sub_p, th_max, tag + '+', depth + 1, max_depth, log) +
           fully_resolve(sub_m, th_max, tag + '-', depth + 1, max_depth, log))


def classify_by_projector(v, P, tol=0.4):
    """For each column of v, its expectation <v|P|v> (should be bimodal near 0/1).
    Returns (vals, idxA, idxB) with idxA: vals > 1 - tol, idxB: vals < tol."""
    vals = np.real(np.sum(v.conj() * (P @ v), axis=0))
    idxA = np.where(vals > 1 - tol)[0]
    idxB = np.where(vals < tol)[0]
    return vals, idxA, idxB


def level_blocks(E0, nu):
    """[(tag, th, slice)] for every contiguous same-th block of a SubSector's E0 array,
    'u' blocks first (indices < nu), then 't' blocks."""
    out = []
    Eu, Et = E0[:nu], E0[nu:]
    for tag, E, off in (('u', Eu, 0), ('t', Et, nu)):
        i = 0
        n = len(E)
        while i < n:
            j = i
            while j < n and E[j] == E[i]:
                j += 1
            out.append((tag, float(E[i]), slice(off + i, off + j)))
            i = j
    return out


def level_plusminus_basis(Sop, E0, nu, ztol=1e-6):
    """Block-diagonalise Sop (n x n, [Sop,diag(E0)]=0 assumed) one bare level at a time
    (separately within the untwisted and twisted segments, per T3's instruction) and split
    each block into its (Sop > 0) / (Sop < 0) eigenspaces.  A level block that is not fully
    covered by the graph-derived S (e.g. it contains 'unlifted'/kernel or singleton states
    for which S was left at exactly 0) shows near-zero eigenvalues there; those are assigned
    by a documented, physically inert convention (see module docstring / the T1 report):
    alternately, to keep the two new sub-sectors close in size.  Returns
    dict(Qu_p, Qu_m, Qt_p, Qt_m, Eu_p, Eu_m, Et_p, Et_m, n_ambiguous)."""
    n = len(E0)
    cols = {('u', +1): [], ('u', -1): [], ('t', +1): [], ('t', -1): []}
    levs = {('u', +1): [], ('u', -1): [], ('t', +1): [], ('t', -1): []}
    n_amb = 0
    amb_toggle = 0
    for tag, th, sl in level_blocks(E0, nu):
        blk = Sop[sl, sl]
        blk = (blk + blk.conj().T) / 2
        w, V = np.linalg.eigh(blk)
        for i in range(len(w)):
            full = np.zeros(n, complex)
            full[sl] = V[:, i]
            if w[i] > ztol:
                sgn = +1
            elif w[i] < -ztol:
                sgn = -1
            else:
                n_amb += 1
                sgn = +1 if amb_toggle % 2 == 0 else -1
                amb_toggle += 1
            cols[(tag, sgn)].append(full)
            levs[(tag, sgn)].append(th)
    Qu_p = np.stack(cols[('u', +1)], axis=1)[:nu] if cols[('u', +1)] else np.zeros((nu, 0), complex)
    Qu_m = np.stack(cols[('u', -1)], axis=1)[:nu] if cols[('u', -1)] else np.zeros((nu, 0), complex)
    Qt_p = np.stack(cols[('t', +1)], axis=1)[nu:] if cols[('t', +1)] else np.zeros((n - nu, 0), complex)
    Qt_m = np.stack(cols[('t', -1)], axis=1)[nu:] if cols[('t', -1)] else np.zeros((n - nu, 0), complex)
    return dict(Qu_p=Qu_p, Qu_m=Qu_m, Qt_p=Qt_p, Qt_m=Qt_m,
                Eu_p=np.array(levs[('u', +1)]), Eu_m=np.array(levs[('u', -1)]),
                Et_p=np.array(levs[('t', +1)]), Et_m=np.array(levs[('t', -1)]),
                n_ambiguous=n_amb)


def build_split_subsectors(sub, Sop, ztol=1e-6):
    """Given a SubSector `sub` and its exact symmetry Sop (n x n), return the two
    (S=+1, S=-1) SubSector objects, splitting E0/M within untwisted and twisted parts
    separately (bare energies stay exactly diagonal -- see r2c_gate19_lib module doc)."""
    from r2c_build import SubSector
    B = level_plusminus_basis(Sop, sub.E0, sub.nu, ztol=ztol)
    Qu_p, Qu_m = B['Qu_p'], B['Qu_m']
    Qt_p, Qt_m = B['Qt_p'], B['Qt_m']
    Mp = Qt_p.conj().T @ sub.M @ Qu_p
    Mm = Qt_m.conj().T @ sub.M @ Qu_m
    sub_p = SubSector(np.concatenate([B['Eu_p'], B['Et_p']]), Mp, len(B['Eu_p']))
    sub_m = SubSector(np.concatenate([B['Eu_m'], B['Et_m']]), Mm, len(B['Eu_m']))
    return sub_p, sub_m, B
