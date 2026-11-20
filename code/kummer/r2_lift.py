"""Route 2 Phase A, Stages 4-5: the chiral half-deformation map T_Adot, the Guo-Mathur
primaries, and the second-order lift matrix (GATE 5).

The projected chiral operator.  Guo-Mathur's lifting operator is (1912.05567 eqs 3.6-3.8)
    Gbar^{alphabar (P)}_{Adot,0} = pi P ( G^+_{Adot,-1/2} sigma^{- alphabar} ) P .
Its LEFT-moving part is, by their (B.6) and the contour split (E.26),

    pi P G^+_{Adot,-1/2} sigma^-  =  -pi P G^-_{Adot,-1/2} sigma^+
                                  =  -pi P [ G^{-,tw}_{Adot,0} Sigma - Sigma G^{-,gl}_{Adot,0} ]
                                  =  -pi T_Adot ,

with Sigma = sigma_2^+(w0=0) the twist map of r2_twist, G^{-,tw} the doubly-wound-copy
supercharge zero mode and G^{-,gl} the global (copy-summed) untwisted one.  P projects on
the degenerate level, which here is automatic: every operator involved is level graded and
we simply keep the same level throughout.

The RIGHT-moving part of Gbar^{+(P)}_{Adot,0} acting on the bottom member of a long
multiplet is P sigmabar^+ |0bar_R^-> |0bar_R^-> = |0bar^{2-}_R> (1912.05567 eq E.4), of unit
norm, and Gbar^{-(P)}_{-,0} annihilates it (their E.3).  Hence for the bottom ("phi")
members of long multiplets -- the column chi^l_{j=1,h} chibar_{jbar3=-1} of 2008.01274
Table 1 -- eq (3.10)/(6.1) collapses to

    E^(2) = 2 lambda^2 || pi T_Adot |phi^L> ||^2 = 2 pi^2 lambda^2 || T_Adot |phi^L> ||^2

on a unit-normalised left mover |phi^L>, for either Adot.

Primaries.  2008.01274 sec 5: the left movers of the chi^l_{j=1,h} chibar_{-1} column are
built with the ANTISYMMETRIC operators O^(A)_n = O^(1)_n - O^(2)_n on |0_R^->|0_R^->, and
are singled out by J^{-(g)}_0 |phi> = J^{+(g)}_1 |phi> = G^{-(g)}_{Adot,0} |phi> = 0
(their eq 5.7), with j = 1, i.e. 2 j_L = -2.
"""
import numpy as np

import r2_algebra as al
import r2_twist as tw
from r2_algebra import UNTW, TW, PLUS, MINUS, VAC, two_h, vadd, vscale


# ---------------------------------------------------------------------------
# the chiral map
# ---------------------------------------------------------------------------
def T_Adot(vec_untw, Ad, th=None):
    """T_Adot = P [ G^{-,tw}_{Adot,0} Sigma - Sigma G^{-,gl}_{Adot,0} ] on an untwisted
    (1,1)-sector vector of definite level."""
    if not vec_untw:
        return {}
    if th is None:
        ths = {two_h(k) for k in vec_untw}
        assert len(ths) == 1
        th = ths.pop()
    mm = th + 4
    a = TW.G(tw.twist(vec_untw, th), MINUS, Ad, 0, mu_max=mm)
    a = {k: c for k, c in a.items() if two_h(k) == th}
    g = al.G_global(UNTW, vec_untw, MINUS, Ad, 0, mu_max=mm)
    g = {k: c for k, c in g.items() if two_h(k) == th}
    b = tw.twist(g, th) if g else {}
    return {k: c for k, c in vadd((a, 1.0), (b, -1.0)).items() if abs(c) > 1e-12}


# ---------------------------------------------------------------------------
# antisymmetric-operator states and the Guo-Mathur primaries
# ---------------------------------------------------------------------------
def _anti_b(vec, n, A, Ad):
    """alpha^(A)_{A Adot,-n} = alpha^(1)_{..,-n} - alpha^(2)_{..,-n}."""
    return vadd((UNTW.alpha(vec, -2 * n, A, Ad, copy=1), 1.0),
                (UNTW.alpha(vec, -2 * n, A, Ad, copy=2), -1.0))


def _anti_f(vec, n, alp, A):
    return vadd((UNTW.d(vec, -2 * n, alp, A, copy=1), 1.0),
                (UNTW.d(vec, -2 * n, alp, A, copy=2), -1.0))


def antisym_states(th):
    """All monomials in the antisymmetric creation operators with total two_h = th.
    Returned as (label, vector) with vectors in UNTW."""
    lmax = th // 2
    fslots = [(0, PLUS, A) for A in (PLUS, MINUS)] + \
             [(n, alp, A) for n in range(1, lmax + 1) for alp in (PLUS, MINUS)
              for A in (PLUS, MINUS)]
    bslots = [(n, A, Ad) for n in range(1, lmax + 1) for A in (PLUS, MINUS)
              for Ad in (PLUS, MINUS)]
    out = []

    def rec_b(i, rem, acc):
        if rem == 0:
            yield tuple(acc)
            return
        if i >= len(bslots):
            return
        n = bslots[i][0]
        for c in range(0, rem // (2 * n) + 1):
            yield from rec_b(i + 1, rem - c * 2 * n, acc + ([(bslots[i], c)] if c else []))

    def rec_f(i, rem, acc):
        if i >= len(fslots):
            yield (rem, tuple(acc))
            return
        yield from rec_f(i + 1, rem, acc)
        w = 2 * fslots[i][0]
        if w <= rem:
            yield from rec_f(i + 1, rem - w, acc + [fslots[i]])

    for rem, fs in rec_f(0, th, []):
        for bs in rec_b(0, rem, []):
            # Z_2 (orbifold) parity: each antisymmetric operator is Z_2 odd, so a monomial
            # with an odd number of them is projected out.  The right movers of the bottom
            # members of long multiplets are symmetric between the copies, hence so must be
            # the left movers (1912.05567 sec 6.2).
            nops = sum(c for (_, c) in bs) + len(fs)
            if nops % 2:
                continue
            v = {VAC: 1.0 + 0j}
            for (slot, c) in bs:
                n, A, Ad = slot
                for _ in range(c):
                    v = _anti_b(v, n, A, Ad)
            for slot in reversed(fs):     # apply right-to-left so the written order is fs
                n, alp, A = slot
                v = _anti_f(v, n, alp, A)
            if v:
                out.append(((bs, fs), v))
    return out


def _mat(vecs):
    keys = sorted(set().union(*[set(v) for v in vecs])) if vecs else []
    idx = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(vecs), len(keys)), dtype=complex)
    for i, v in enumerate(vecs):
        for k, c in v.items():
            M[i, idx[k]] = c
    return M, keys


def _vec(row, keys, tol=1e-12):
    return {k: c for k, c in zip(keys, row) if abs(c) > tol}


def gm_primaries(th, verbose=False):
    """The chi^l_{j=1,h} chibar_{-1} left-moving primaries at level two_h = th, as an
    ORTHONORMAL list of UNTW vectors."""
    cand = [v for (lab, v) in antisym_states(th) if v]
    # j = 1 lowest weight  =>  2 j_L = -2
    cand = [v for v in cand if UNTW.charges(next(iter(v)))[1] == -2]
    if not cand:
        return []
    basis = orthonormalize(UNTW, cand)
    if verbose:
        print('   antisym states with 2jL=-2: %d, independent: %d' % (len(cand), len(basis)))
    # constraints  J^-(g)_0, J^+(g)_1, G^-(g)_{Adot,0}
    mm = th + 4
    blocks = []
    for v in basis:
        blocks.append([al.Jm_global(UNTW, v, 0, mu_max=mm),
                       al.Jp_global(UNTW, v, 2, mu_max=mm),
                       al.G_global(UNTW, v, MINUS, PLUS, 0, mu_max=mm),
                       al.G_global(UNTW, v, MINUS, MINUS, 0, mu_max=mm)])
    keysets = [sorted(set().union(*[set(b[j]) for b in blocks]) or {VAC}) for j in range(4)]
    cols = sum(len(ks) for ks in keysets)
    C = np.zeros((len(basis), cols), dtype=complex)
    off = 0
    for j in range(4):
        kidx = {k: i for i, k in enumerate(keysets[j])}
        for i, b in enumerate(blocks):
            for k, c in b[j].items():
                C[i, off + kidx[k]] = c
        off += len(keysets[j])
    # null space of C^T acting from the left: rows x  with  x^T C = 0
    U, S, Vt = np.linalg.svd(C, full_matrices=True)
    tol = 1e-8 * max(S[0], 1.0) if len(S) else 1.0
    nulldim = len(basis) - int(np.sum(S > tol))
    prim = []
    for i in range(len(basis) - nulldim, len(basis)):
        coef = np.conj(U[:, i])
        v = {}
        for a, cc in enumerate(coef):
            if abs(cc) < 1e-14:
                continue
            for k, c in basis[a].items():
                v[k] = v.get(k, 0) + cc * c
        v = {k: c for k, c in v.items() if abs(c) > 1e-12}
        if v:
            prim.append(v)
    return orthonormalize(UNTW, prim)


def orthonormalize(sec, vecs, tol=1e-9):
    out = []
    for v in vecs:
        w = dict(v)
        for u in out:
            ov = sec.ip(u, w)
            if abs(ov) > 0:
                w = vadd((w, 1.0), (u, -ov))
        n2 = sec.norm2(w)
        if n2 > tol:
            out.append(vscale(w, 1.0 / np.sqrt(n2)))
    return out


# ---------------------------------------------------------------------------
# the lift
# ---------------------------------------------------------------------------
def lift_matrix(prim, Ad=PLUS, th=None):
    """E^(2)/(pi^2 lambda^2) matrix on an orthonormal set of bottom primaries."""
    imgs = [T_Adot(p, Ad, th=th) for p in prim]
    n = len(prim)
    M = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            M[i, j] = 2.0 * TW.ip(imgs[i], imgs[j])
    return M, imgs


# ---------------------------------------------------------------------------
# the chiral map as a dense matrix (Stage 4 groundwork)
# ---------------------------------------------------------------------------
def T_matrix(th, Ad=PLUS):
    """T_Adot : V^L_untw(two_h=th) -> V^L_tw(two_h=th) as a dense matrix in ORTHONORMAL
    monomial bases.  Returns (M, untw_basis, tw_basis)."""
    Bu = al.enumerate_basis(UNTW, th, mu_min=2)
    Bt = al.enumerate_basis(TW, th, mu_min=1)
    tidx = {k: i for i, k in enumerate(Bt)}
    nu = np.array([UNTW.mono_norm2(k) for k in Bu])
    nt = np.array([TW.mono_norm2(k) for k in Bt])
    M = np.zeros((len(Bt), len(Bu)), dtype=complex)
    for j, m in enumerate(Bu):
        for k, c in T_Adot({m: 1.0 + 0j}, Ad, th=th).items():
            M[tidx[k], j] = c
    M = np.sqrt(nt)[:, None] * M * (1.0 / np.sqrt(nu))[None, :]
    return M, Bu, Bt
