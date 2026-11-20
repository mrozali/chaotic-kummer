"""Route 2 Phase A, Stage 6+: the RELATIVE theory -- Virasoro modes, the SO(4)_I
(SU(2)_1 x SU(2)_2) index rotations, and the massive N = 4 primary projectors.

Relative vs centre of mass
--------------------------
Untwisted (1,1): the modes O^(rel) = (O^(1) - O^(2))/sqrt2 and O^(com) = (O^(1)+O^(2))/sqrt2
each satisfy the single-copy c = 6 algebra, and the global N = 4 generators split as
G^gl = G^com + G^rel (they are bilinear, and the cross terms cancel).  A state built only
from ANTISYMMETRIC creation operators has its c.o.m. factor in the c.o.m. R ground state, and
on such states G^com_r = J^com_r = L^com_r = 0 for every r > 0 (the c.o.m. annihilators act on
the c.o.m. vacuum) and J^{3,com}_0 = -1/2, J^{-,com}_0 = 0.  So the GLOBAL generators of
r2_algebra compute the RELATIVE ones on relative states, up to the constant -1/2 in J^3_0.

Twisted (2): the doubly wound string's modes with EVEN mu (integer mode number) are the c.o.m.
ones and carry the fermion zero modes; ODD mu (half-odd mode number) are relative.  A twisted
monomial with all mu odd therefore has its c.o.m. factor in the c.o.m. R ground state, and the
same statement holds.

Consequently, in BOTH sectors, a relative massive N = 4 primary is characterised by
    2 j_L = -2   (the global charge: -1 from the c.o.m. R vacuum, -1 from the relative one)
and annihilation by
    L_n, J^a_n, G^alpha_{Adot,n}  (n >= 1)   and   J^-_0,  G^-_{Adot,0} .
The last two come from spectral flow: an NS massive primary (h, j = 0) flows to an R state at
level h with j = -1/2, and the NS conditions J^-_n (n>=1), G^-_{Adot,r} (r>=1/2) become
J^-_n (n>=0), G^-_{Adot,n} (n>=0).

Virasoro modes
--------------
L_m is built from the free fields with the metrics implied by r2_algebra's own conventions,
    [alpha_{A Adot,m}, alpha_{B Bdot,n}] = (c/6) m g_{(A Adot)(B Bdot)} delta,
        g_{ij} = -eps_{AB} eps_{Adot Bdot},  g^{ij} = -eps^{AB} eps^{Adot Bdot}
    {d^{alpha A}_r, d^{beta B}_s}         = (c/6) h^{IJ} delta,
        h^{IJ} = -eps^{alpha beta} eps^{AB},  (h^{-1})_{IJ} = -eps_{alpha beta} eps_{AB}
    L_m = (1/2)(6/c) [ sum_n g^{ij} :alpha_{i,m-n} alpha_{j,n}:
                       + sum_r (r + m/2) (h^{-1})_{IJ} :d^I_{m-r} d^J_r: ]
which is verified in this module's self-test against [L_m, alpha_n] = -n alpha_{m+n},
[L_m, d_r] = -(r + m/2) d_{m+r}, the Virasoro algebra with c = 6 n_i, and L_0 = level + c/24.
"""
import numpy as np

import r2_algebra as al
from r2_algebra import (UNTW, TW, PLUS, MINUS, VAC, opp, eps_up, eps_lo, two_h,
                        vadd, vscale)


# ---------------------------------------------------------------------------
# Virasoro modes
# ---------------------------------------------------------------------------
def _sb(A, Ad):
    """g^{ij} for i = (A,Adot), j = (-A,-Adot)."""
    return -eps_up(A, opp(A)) * eps_up(Ad, opp(Ad))


def _sf(alp, A):
    """(h^{-1})_{IJ} for I = (alpha,A), J = (-alpha,-A)."""
    return -eps_lo(alp, opp(alp)) * eps_lo(A, opp(A))


def _is_cre_b(mu):
    return mu < 0


def _is_cre_f(mu, alp):
    return (mu < 0) or (mu == 0 and alp == PLUS)


def L_copy(sec, vec, m2, cp, mu_max):
    """L_{m2/2} of ONE component string (copy cp), applied to vec."""
    out = {}
    inv = 0.5 / sec.cfac
    # --- bosons ---------------------------------------------------------
    zero_ok = sec.mom is not None
    for mu in al.mrange(sec.mu_par, mu_max):
        nu = m2 - mu
        if abs(nu) > mu_max:
            continue
        if (mu == 0 or nu == 0) and not zero_ok:
            continue
        for A in (PLUS, MINUS):
            for Ad in (PLUS, MINUS):
                s = _sb(A, Ad)
                if s == 0:
                    continue
                # :alpha_{A Ad, nu/2} alpha_{-A,-Ad, mu/2}:
                if _is_cre_b(mu) and not _is_cre_b(nu):
                    v = sec.alpha(vec, nu, A, Ad, copy=cp)
                    v = sec.alpha(v, mu, opp(A), opp(Ad), copy=cp)
                else:
                    v = sec.alpha(vec, mu, opp(A), opp(Ad), copy=cp)
                    v = sec.alpha(v, nu, A, Ad, copy=cp)
                for k, c in v.items():
                    out[k] = out.get(k, 0) + inv * s * c
    # --- fermions -------------------------------------------------------
    for rr in al.mrange(sec.mu_par, mu_max):
        ss = m2 - rr
        if abs(ss) > mu_max:
            continue
        w = 0.5 * rr + 0.25 * m2          # (r + m/2) with r = rr/2, m = m2/2
        if w == 0:
            continue
        for alp in (PLUS, MINUS):
            for A in (PLUS, MINUS):
                t = _sf(alp, A)
                if t == 0:
                    continue
                # :d^{alpha A}_{ss/2} d^{-alpha,-A}_{rr/2}:
                c1 = _is_cre_f(ss, alp)
                c2 = _is_cre_f(rr, opp(alp))
                if c2 and not c1:
                    v = sec.d(vec, ss, alp, A, copy=cp)
                    v = sec.d(v, rr, opp(alp), opp(A), copy=cp)
                    sg = -1.0
                else:
                    v = sec.d(vec, rr, opp(alp), opp(A), copy=cp)
                    v = sec.d(v, ss, alp, A, copy=cp)
                    sg = 1.0
                for k, c in v.items():
                    out[k] = out.get(k, 0) + inv * t * w * sg * c
    return {k: v for k, v in out.items() if abs(v) > 1e-13}


def L_global(sec, vec, m2, mu_max=None):
    if mu_max is None:
        mu_max = max((two_h(k) for k in vec), default=0) + abs(m2) + 4
    out = {}
    for cp in sec.copies:
        for k, c in L_copy(sec, vec, m2, cp, mu_max).items():
            out[k] = out.get(k, 0) + c
    return {k: v for k, v in out.items() if abs(v) > 1e-13}


# ---------------------------------------------------------------------------
# SO(4)_I = SU(2)_1 x SU(2)_2 index rotations (outer automorphisms of the mode
# algebra; symmetries of the orbifold combinatorics, broken to the point group by
# the T^4 lattice once momenta are switched on)
# ---------------------------------------------------------------------------
def _replace_boson(mono, key, newkey):
    """Replace one factor `key` (count >= 1) by `newkey`; returns (newmono, mult)."""
    bos, fer = mono
    cnt = al._bos_count(bos, key)
    if cnt == 0:
        return None, 0
    b = al._bos_remove(bos, key)
    b = al._bos_add(b, newkey)
    return (b, fer), cnt


def _replace_fermion(mono, key, newkey):
    bos, fer = mono
    if key not in fer:
        return None, 0
    if newkey in fer:
        return None, 0
    rest = tuple(k for k in fer if k != key)
    # sign: remove key from its slot, insert newkey in its slot
    i0 = sorted(fer).index(key)
    nf = tuple(sorted(rest + (newkey,)))
    i1 = nf.index(newkey)
    sg = -1.0 if ((i0 + i1) & 1) else 1.0
    return (bos, nf), sg


def su2_index(sec, vec, which, raise_=True):
    """SU(2)_1 (which=1) or SU(2)_2 (which=2) raising/lowering ZERO MODE, as a
    derivation on monomials.

    Charges: d^{alpha A} has 2j_1 = +A; alpha_{A Adot} has
    2j_1 = -A and 2j_2 = -Adot (LOWER indices).  So the SU(2)_1 raising operator sends
    alpha_{+,Adot} -> alpha_{-,Adot} and d^{alpha,-} -> d^{alpha,+}; the SU(2)_2 raising
    operator sends alpha_{A,+} -> alpha_{A,-} (fermions carry no Adot)."""
    out = {}
    src_b = PLUS if raise_ else MINUS       # boson index value that gets moved
    src_f = MINUS if raise_ else PLUS       # fermion index value that gets moved
    for mono, coef in vec.items():
        bos, fer = mono
        for (k, cnt) in bos:
            mu, cp, Ai, Adi = k
            if which == 1:
                if Ai != src_b:
                    continue
                nk = (mu, cp, opp(Ai), Adi)
            else:
                if Adi != src_b:
                    continue
                nk = (mu, cp, Ai, opp(Adi))
            nm, m = _replace_boson(mono, k, nk)
            if m:
                out[nm] = out.get(nm, 0) + coef * m
        if which == 1:
            for k in fer:
                mu, cp, ai, Ai = k
                if Ai != src_f:
                    continue
                nk = (mu, cp, ai, opp(Ai))
                nm, sg = _replace_fermion(mono, k, nk)
                if nm is not None:
                    # RELATIVE MINUS between the fermion (upper A) and the boson (lower A)
                    # action: invariance of the contraction sum_A d^{alpha A} alpha_{A Adot}
                    # (which is what makes G^{alpha}_{Adot} an SU(2)_1 singlet) forces it.
                    out[nm] = out.get(nm, 0) - coef * sg
    return {k: v for k, v in out.items() if abs(v) > 1e-13}


# ---------------------------------------------------------------------------
# relative-sector bases
# ---------------------------------------------------------------------------
def rel_parity_untw(mono):
    """(-1)^(number of relative creation operators) for an UNTWISTED monomial written in
    the ANTISYMMETRIC operator basis (labels, not UNTW monomials) -- see rel_untw_labels."""
    raise NotImplementedError


def rel_untw_labels(th):
    """Labels of a basis of the untwisted RELATIVE chiral space at two_h = th:
    monomials in the antisymmetric operators.  Returns a list of
    (bos, fer, nops) with bos = tuple(((n,A,Ad), count)), fer = tuple((n,alpha,A)) sorted,
    n = physical mode number >= 0 (fermion zero modes n = 0, alpha = +)."""
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
            nops = sum(c for (_, c) in bs) + len(fs)
            out.append((bs, fs, nops))
    return out


def rel_untw_vector(lab):
    """Build the UNTW vector for a relative-basis label (unnormalised: uses the
    antisymmetric operators O^(1) - O^(2), i.e. sqrt2 x the canonically normalised
    relative modes; irrelevant once orthonormalised)."""
    bs, fs, _ = lab
    v = {VAC: 1.0 + 0j}
    for (slot, c) in bs:
        n, A, Ad = slot
        for _ in range(c):
            v = vadd((UNTW.alpha(v, -2 * n, A, Ad, copy=1), 1.0),
                     (UNTW.alpha(v, -2 * n, A, Ad, copy=2), -1.0))
    for slot in reversed(fs):
        n, alp, A = slot
        v = vadd((UNTW.d(v, -2 * n, alp, A, copy=1), 1.0),
                 (UNTW.d(v, -2 * n, alp, A, copy=2), -1.0))
    return v


def rel_tw_basis(th):
    """Monomials of the twisted RELATIVE chiral space at two_h = th: TW monomials all of
    whose modes have ODD mu (so the c.o.m. factor is in its R ground state)."""
    out = []
    for m in al.enumerate_basis(TW, th, mu_min=1):
        bos, fer = m
        ok = all(k[0] % 2 for (k, c) in bos) and all(k[0] % 2 for k in fer)
        if ok:
            out.append(m)
    return out


def tw_parity(mono):
    return 1 if (two_h(mono) % 2 == 0) else -1


# ---------------------------------------------------------------------------
# massive N = 4 primaries of the relative theory
# ---------------------------------------------------------------------------
def _to_matrix(vecs, keys=None):
    if keys is None:
        keys = sorted(set().union(*[set(v) for v in vecs])) if vecs else []
    idx = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(vecs), len(keys)), dtype=complex)
    for i, v in enumerate(vecs):
        for k, c in v.items():
            M[i, idx[k]] = c
    return M, keys


def gram_orthonormal(sec, vecs, tol=1e-9):
    """Orthonormal basis of span(vecs) as a list of vectors, via the Gram matrix
    (much faster than sequential Gram-Schmidt for large sets)."""
    if not vecs:
        return []
    keys = sorted(set().union(*[set(v) for v in vecs]))
    idx = {k: i for i, k in enumerate(keys)}
    w = np.array([sec.mono_norm2(k) for k in keys])
    M = np.zeros((len(vecs), len(keys)), dtype=complex)
    for i, v in enumerate(vecs):
        for k, c in v.items():
            M[i, idx[k]] = c
    X = M * np.sqrt(w)[None, :]           # rows in an orthonormal monomial frame
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = int(np.sum(S > tol * max(S[0], 1.0)))
    rows = Vt[:r] / np.sqrt(w)[None, :]   # back to the monomial frame
    out = []
    for row in rows:
        out.append({k: row[i] for i, k in enumerate(keys) if abs(row[i]) > 1e-14})
    return out


def common_kernel(sec, basis, ops, tol=1e-8):
    """Orthonormal basis of {v in span(basis) : O v = 0 for every O in ops}.
    `basis` must already be ORTHONORMAL in sec."""
    if not basis:
        return []
    blocks = [[O(v) for O in ops] for v in basis]
    cols = []
    for j in range(len(ops)):
        ks = sorted(set().union(*[set(b[j]) for b in blocks]) or {VAC})
        cols.append(ks)
    ncol = sum(len(k) for k in cols)
    C = np.zeros((len(basis), ncol), dtype=complex)
    off = 0
    for j, ks in enumerate(cols):
        w = np.sqrt(np.array([sec.mono_norm2(k) for k in ks]))
        kidx = {k: i for i, k in enumerate(ks)}
        for i, b in enumerate(blocks):
            for k, c in b[j].items():
                C[i, off + kidx[k]] = c * w[kidx[k]]
        off += len(ks)
    U, S, Vt = np.linalg.svd(C, full_matrices=True)
    rank = int(np.sum(S > tol * max(S[0], 1.0))) if len(S) else 0
    out = []
    for i in range(rank, len(basis)):
        coef = np.conj(U[:, i])
        v = {}
        for a, cc in enumerate(coef):
            if abs(cc) < 1e-14:
                continue
            for k, c in basis[a].items():
                v[k] = v.get(k, 0) + cc * c
        v = {k: c for k, c in v.items() if abs(c) > 1e-12}
        if v:
            out.append(v)
    return out


def primary_ops(sec, mm):
    """The operators whose common kernel (at 2 j_L = -2) is the massive-primary space."""
    ops = [lambda v: L_global(sec, v, 2, mu_max=mm),
           lambda v: L_global(sec, v, 4, mu_max=mm),
           lambda v: al.J3_global(sec, v, 2, mu_max=mm),
           lambda v: al.Jp_global(sec, v, 2, mu_max=mm),
           lambda v: al.Jm_global(sec, v, 2, mu_max=mm),
           lambda v: al.Jm_global(sec, v, 0, mu_max=mm)]
    for alp, r2 in ((MINUS, 0), (PLUS, 2), (MINUS, 2)):
        for Ad in (PLUS, MINUS):
            ops.append(lambda v, a=alp, A=Ad, r=r2: al.G_global(sec, v, a, A, r, mu_max=mm))
    return ops


def massive_primaries(sec, th, parity=None, verbose=False):
    """Orthonormal massive N = 4 primaries of the RELATIVE theory at two_h = th.

    sec = UNTW: relative states are monomials in the antisymmetric operators; `parity`
    selects (-1)^{#antisym operators} = +1 / -1 (None = both).
    sec = TW: relative states are monomials with all mu odd; the parity is fixed to
    (-1)^{th} and `parity` is ignored."""
    if sec is UNTW:
        labs = rel_untw_labels(th)
        if parity is not None:
            labs = [l for l in labs if (-1) ** l[2] == parity]
        cand = [rel_untw_vector(l) for l in labs]
        cand = [v for v in cand if v]
    else:
        cand = [{m: 1.0 + 0j} for m in rel_tw_basis(th)]
    cand = [v for v in cand if sec.charges(next(iter(v)))[1] == -2]
    if not cand:
        return []
    basis = gram_orthonormal(sec, cand)
    mm = th + 8
    prim = common_kernel(sec, basis, primary_ops(sec, mm))
    if verbose:
        print('   th=%d sec=%s cand=%d indep=%d prim=%d'
              % (th, sec.name, len(cand), len(basis), len(prim)))
    return prim
