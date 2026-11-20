"""Route 2 Phase A, GATE 7: untwisted RELATIVE-MOMENTUM states and their twist images
(displaced squeezed states).

Derivation
----------
Covering map (ACM 1002.3132 sec 3.3(b)):  z = z_0 + t^2 = a^2 + t^2, branch point (the twist)
at t = 0, the two untwisted copies at t = +ia (copy 1) and t = -ia (copy 2), the twisted
out-state at t = infinity.  We work at w_0 = 0, z_0 = a^2 = 1.

Copy 1 carries momentum p, copy 2 carries -p (zero total momentum: the relative theory).
With the index conventions of r2_algebra (1405.0259 eqs 2.4-2.5), p_{A Adot} = p^i sigma_i /
sqrt2 and p.p = -eps^{AB} eps^{Adot Bdot} p_{A Adot} p_{B Bdot}; the chiral weight of the
relative momentum state is h_p = p^2 (each copy contributes p^2/2).

1. Displacement.  The twist image is a coherent displacement of |chi>,

       sigma_2^+ |p>^(1)|-p>^(2) = N(p) exp(D_p) |chi>,
       D_p = - sum_{M>0 odd} (w_M / M) sum_{B Bdot} sigma(B) sigma(Bdot)
                    p_{-B,-Bdot} alpha_{B Bdot, -M} ,      sigma(+) = +1, sigma(-) = -1,

   normalised so that [alpha_{A Adot, M}, D_p] = w_M p_{A Adot} for M > 0.  The coefficients
   w_M are fixed EXACTLY by the single-mode intertwining relation of Avery-Chowdhury
   (1007.2202 eq 3.8),

       alpha^{(1,2)}_m  =  (1/2) alpha_{2m}  +/-  (i/2pi) sum_{n odd} z_0^{m-n/2}/(m - n/2) alpha_n ,

   applied to the momentum eigenstate.  Writing O^{(m)} for the right-hand side and using
   O^{(m)}|chi> = 0 for m >= 0 (which is the p = 0 statement alpha^{(1)}_m|0> = 0) together
   with the fact that [O^{(m)}, D_p] is a c-number,

       O^{(m)} exp(D_p)|chi> = [O^{(m)}, D_p] exp(D_p)|chi> ,

   so alpha^{(1)}_m |p> = delta_{m,0} p |p> becomes

       m = 0 :  -(i/pi) sum_{M>0 odd} w_M / M   = 1
       m >= 1:            sum_{M>0 odd} w_M / (m - M/2) = 0 .

   The unique solution is  w_M = 2 i b_{(M-1)/2},  b_j = Gamma(j+1/2)/(sqrt(pi) Gamma(j+1)),
   i.e. w_M = 2i Gamma(M/2)/(sqrt(pi) Gamma((M+1)/2)) for odd M and 0 for even M (the even,
   i.e. centre-of-mass, modes are not displaced -- the relative-theory statement).  The m = 0
   equation is satisfied because sum_j b_j/(2j+1) = arcsin(1) = pi/2, and the m >= 1 equations
   because sum_j b_j/(2j+1+2m) = int_0^1 x^{2m}(1-x^2)^{-1/2} dx = (pi/2) b_m combined with the
   partial-fraction identity checked numerically in r2_gate7 (not included in this repository).
   The same w_M follows from the covering-space classical field
   dX_cl/dt = i p [1/(t-ia) - 1/(t+ia)] = -2 a p/(t^2+a^2) via
   w_M = i oint (dt/2 pi i) (a^2+t^2)^{M/2} dX_cl/dt = -2 i a^M p b_{(M-1)/2}
   (the sign difference is the branch/orientation convention of the mode contour, fixed here
   by the intertwining relation, i.e. by the code's own f^B).
   General z_0: w_M carries z_0^{M/2}, which is the E_t - E_u bookkeeping.

2. Source contraction.  For a mode on top of the momentum state,
   [O^{(n)}, D_p] = S_c(n) p_{A Adot} with
       S_1(n) = (i/2pi) sum_{M>0 odd} w_M/(n - M/2) = b_{-n},   S_2(n) = -b_{-n},
   using sum_j b_j/(2j+1-2n) = (pi/2) b_{-n} for n <= 0.  At n = 0 this reproduces
   S_1(0) p = p (b_0 = 1), which is the momentum eigenvalue itself: a consistency check built
   into the formula.  So an untwisted BOSONIC creation operator alpha^{(c)}_{A Adot, n} may
   either transform by f^B (as on the vacuum), or pair with another operator by C^B, or
   contract with the classical source giving the c-number (+/-) b_{-n} p_{A Adot} (+ for
   copy 1, - for copy 2).  Fermions have no source contraction.  The centre-of-mass
   combination has s_1 + s_2 = 0, so c.o.m. factorisation is automatic.

3. Normalisation.  N(p) is the ratio of the covering-space amplitude with the two vertex
   operators to the one without (which is C_11 = 1):

       N(p) = (dt/dz)^{h_1}|_{t_1} (dt/dz)^{h_1}|_{t_2} (t_1 - t_2)^{-p^2}
            = (2ia)^{-p^2/2} (-2ia)^{-p^2/2} (2ia)^{-p^2} = (4 i z_0)^{-p^2}

   with h_1 = p^2/2.  Its MODULUS, 4^{-p^2} z_0^{-p^2} = 4^{-h_p} z_0^{-h_p}, is unambiguous
   and is checked by the centre-of-mass limit: for copy 1 and copy 2 carrying the SAME
   momentum p (pure c.o.m., no relative momentum) the same formula gives
   |N| = |2ia|^{-(p_1-p_2)^2/2} = 1, as c.o.m. factorisation demands.  The residual phase is a
   branch choice (t_1 <-> t_2 changes it by (-1)^{p^2}); it depends on p only through p^2, so
   it is a diagonal unitary on the untwisted side of every block and does not affect any
   singular value.  We take N(p) = 4^{-p^2} real positive.
"""
import numpy as np
import mpmath as mp

import r2_algebra as al
import r2_twist as tw
import r2_rel as rl
from r2_algebra import UNTW, TW, PLUS, MINUS, VAC, opp, eps_lo, eps_up, two_h, vadd, vscale

mp.mp.dps = 30
_SQPI = float(mp.sqrt(mp.pi))

# source sign of copy 1 in the intertwining relation (copy 2 is minus it).  Fixed by the
# numerical intertwining test in r2_gate7 (not included in this repository)
# (G^{+,tw} Sigma = Sigma G^{+(g)} on momentum states).
SRC_SIGN1 = +1.0


def sgn(A):
    return 1.0 if A == PLUS else -1.0


_BC = {}


def bco(j):
    """b_j = Gamma(j+1/2)/(sqrt(pi) Gamma(j+1)) = binom(2j,j)/4^j."""
    v = _BC.get(j)
    if v is None:
        v = float(mp.gamma(mp.mpf(j) + mp.mpf(1) / 2) / (mp.sqrt(mp.pi) * mp.gamma(j + 1)))
        _BC[j] = v
    return v


def wcoef(M):
    """w_M with [alpha_{A Adot,M}, D_p] = w_M p_{A Adot}."""
    if M <= 0 or M % 2 == 0:
        return 0.0
    return 2j * bco((M - 1) // 2)


def pdict(p4):
    """p_{A Adot} = (1/sqrt2) p^i sigma_i from a real 4-vector (p1,p2,p3,p4)."""
    p1, p2, p3, p4_ = [float(x) for x in p4]
    s = 1.0 / np.sqrt(2.0)
    return {(PLUS, PLUS): s * (p3 + 1j * p4_),
            (PLUS, MINUS): s * (p1 - 1j * p2),
            (MINUS, PLUS): s * (p1 + 1j * p2),
            (MINUS, MINUS): s * (-p3 + 1j * p4_)}


def psq(p4):
    return float(sum(float(x) ** 2 for x in p4))


def mom_context(p4):
    """UNTW.mom for copy 1 = p, copy 2 = -p."""
    pd = pdict(p4)
    return {1: pd, 2: {k: -v for k, v in pd.items()}}


def Nfac(p4):
    return 4.0 ** (-psq(p4))


# ---------------------------------------------------------------------------
# |chi_p> = exp(D_p)|chi>
# ---------------------------------------------------------------------------
_CHIP = {}


def build_chi_pd(pd, th_max):
    key = (tuple(sorted((k, complex(v)) for k, v in pd.items())), th_max)
    if key in _CHIP:
        return _CHIP[key]
    graded = {l: dict(v) for l, v in tw.build_chi(th_max).items()}
    for M in range(1, th_max + 1, 2):
        wm = wcoef(M)
        for B in (PLUS, MINUS):
            for Bd in (PLUS, MINUS):
                coef = -(wm / M) * sgn(B) * sgn(Bd) * pd[(opp(B), opp(Bd))]
                if abs(coef) < 1e-16:
                    continue
                # multiply by exp(coef * alpha_{B Bd, -M})
                out = {l: dict(v) for l, v in graded.items()}
                cur = graded
                k = 1
                while True:
                    nxt = {}
                    for l, v in cur.items():
                        if l + M > th_max:
                            continue
                        wv = TW.alpha(v, -M, B, Bd)
                        if not wv:
                            continue
                        tgt = nxt.setdefault(l + M, {})
                        for kk, c in wv.items():
                            tgt[kk] = tgt.get(kk, 0) + c * coef / k
                    nxt = {l: {kk: c for kk, c in v.items() if abs(c) > 1e-18} for l, v in nxt.items()}
                    nxt = {l: v for l, v in nxt.items() if v}
                    if not nxt:
                        break
                    for l, v in nxt.items():
                        t = out.setdefault(l, {})
                        for kk, c in v.items():
                            t[kk] = t.get(kk, 0) + c
                    cur = nxt
                    k += 1
                graded = out
    graded = {l: v for l, v in graded.items() if v}
    _CHIP[key] = graded
    return graded


# ---------------------------------------------------------------------------
# the twist map on momentum states
# ---------------------------------------------------------------------------
def _src_value(op, pd):
    """c-number for contracting an untwisted creation operator with the classical source."""
    kind, cp, sp, n = op
    if kind != 'b':
        return 0.0
    s = SRC_SIGN1 if cp == 1 else -SRC_SIGN1
    return s * bco(-n) * pd[sp]


def _pairings_src(ops, pd):
    if not ops:
        yield (1.0 + 0j, ())
        return
    head, rest = ops[0], ops[1:]
    for (c, tail) in _pairings_src(rest, pd):
        yield (c, (head,) + tail)
    s = _src_value(head, pd)
    if s != 0:
        for (c, tail) in _pairings_src(rest, pd):
            yield (s * c, tail)
    for j, other in enumerate(rest):
        cval = tw.C_pair(head, other)
        if cval == 0:
            continue
        if other[0] == 'f':
            nf = sum(1 for o in rest[:j] if o[0] == 'f')
            sg = -1.0 if (nf & 1) else 1.0
        else:
            sg = 1.0
        sub = rest[:j] + rest[j + 1:]
        for (c, tail) in _pairings_src(sub, pd):
            yield (cval * sg * c, tail)


_TWM_CACHE = {}


def build_chi_p(p4, th_max):
    return build_chi_pd(pdict(p4), th_max)


def twist_mono_pd(mono, pd, th_target):
    """sigma_2^+(0) on (untwisted oscillator monomial) x |p>^(1)|-p>^(2), projected on the
    twisted level two_h == th_target.  th_target is measured from the TWISTED R ground state
    and must equal 2*(h_p + oscillator level) for the level-preserving map."""
    key = (mono, tuple(sorted((k, complex(v)) for k, v in pd.items())), th_target)
    got = _TWM_CACHE.get(key)
    if got is not None:
        return got
    ops = tw.mono_ops(mono)
    chip = build_chi_pd(pd, th_target)
    total = {}
    for (coeff, rest) in _pairings_src(ops, pd):
        if coeff == 0:
            continue
        graded = {l: v for l, v in chip.items() if l <= th_target}
        for op in reversed(rest):
            graded = tw._apply_hat(graded, op, th_target)
            if not graded:
                break
        v = graded.get(th_target)
        if not v:
            continue
        for k, c in v.items():
            total[k] = total.get(k, 0) + coeff * c
    out = {k: c for k, c in total.items() if abs(c) > 1e-15}
    _TWM_CACHE[key] = out
    return out


def twist_pd(vec, pd, th_target, Nf=1.0):
    out = {}
    for m, c in vec.items():
        for k, cc in twist_mono_pd(m, pd, th_target).items():
            out[k] = out.get(k, 0) + Nf * c * cc
    return {k: c for k, c in out.items() if abs(c) > 1e-14}


def twist_mom(vec, p4, th_target):
    """Includes the amplitude N(p) = 4^{-p^2}."""
    return twist_pd(vec, pdict(p4), th_target, Nfac(p4))


def T_Adot_mom(vec, p4, Ad, th):
    """T_Adot on a momentum-sector untwisted vector (all monomials at oscillator level
    th/2 - h_p)."""
    if not vec:
        return {}
    mm = th + 6
    save = UNTW.mom
    UNTW.mom = mom_context(p4)
    try:
        g = al.G_global(UNTW, vec, MINUS, Ad, 0, mu_max=mm)
    finally:
        UNTW.mom = save
    a = TW.G(twist_mom(vec, p4, th), MINUS, Ad, 0, mu_max=mm)
    a = {k: c for k, c in a.items() if two_h(k) == th}
    b = twist_mom(g, p4, th) if g else {}
    return {k: c for k, c in vadd((a, 1.0), (b, -1.0)).items() if abs(c) > 1e-12}


# ---------------------------------------------------------------------------
# massive N = 4 primaries in a momentum sector
# ---------------------------------------------------------------------------
def massive_primaries_mom(p4, th_osc, parity=None):
    """Orthonormal chiral massive N = 4 primaries of the untwisted relative theory in the
    momentum sector p (copy 1) / -p (copy 2), with oscillator level th_osc/2 on top.
    The conditions are the same as in r2_rel.massive_primaries but every operator now sees
    the boson zero mode alpha^{(c)}_0 = +/- p."""
    labs = rl.rel_untw_labels(th_osc)
    if parity is not None:
        labs = [l for l in labs if (-1) ** l[2] == parity]
    cand = [rl.rel_untw_vector(l) for l in labs]
    cand = [v for v in cand if v]
    cand = [v for v in cand if UNTW.charges(next(iter(v)))[1] == -2]
    if not cand:
        return []
    save = UNTW.mom
    UNTW.mom = mom_context(p4)
    try:
        basis = rl.gram_orthonormal(UNTW, cand)
        mm = th_osc + 8
        prim = rl.common_kernel(UNTW, basis, rl.primary_ops(UNTW, mm))
    finally:
        UNTW.mom = save
    return prim
