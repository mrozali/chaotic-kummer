"""Route 2 Phase A: the Z_2 twist map sigma_2^+ : (1,1) untwisted -> (2) twisted, at M=N=1,
as a genuine linear operator, from Avery-Chowdhury-Mathur closed forms.

Sources (equation numbers are the papers' own):
  * arXiv:1002.3132 eq (4.20)  squeezed state |chi> = sigma_2^+(w0)|0_R^->^(1)|0_R^->^(2)
                    eq (5.12)  gamma^B_{mn},  eq (5.25) gamma^F_{mn}   (C_11 = 1)
                    eq (6.2)   G^-_{Adot,0} on the doubly wound circle
                    eq (6.10)  closed form for G^-_{Adot,0}|chi>       [validation target]
  * arXiv:1003.2746 eq (5.36)  f^B[n,p],  eq (5.52) f^{F+}[n,p],  eq (5.45) f^{F-}[n,p]
                    eq (6.7)   C^B[n1,n2], eq (6.15) C^F[n1,n2]  (pairwise Wick contraction)
                    sec 6.3    the general multi-excitation rule
  * arXiv:1912.05567 eqs (5.2)-(5.12) restate the same f's in the mode labelling used here.

Mode labels.  ACM write the twisted-sector modes with an integer label p whose physical
mode number is p/2 (doubly wound circle).  That integer p is exactly the `mu` of
r2_algebra for the twisted sector.  Untwisted modes n are integers = mu/2.

Copy dependence.  a = e^{w0/2}; copy 1 uses a, copy 2 uses -a (1003.2746 eqs 5.3/5.5/5.7).
We work at w0 = 0, a = 1, so copy 2 uses a = -1: the diagonal (even-p) coefficient 1/2 is
unchanged and every odd-p coefficient flips sign (their a-power 2(n-p')-1 is odd).  Because
the twist is level graded (the a power equals twice the level change), setting a = 1 loses
nothing as long as one always projects on a definite final level, which we do.

Cross-copy Wick contractions.  1003.2746 computes C^B, C^F only for two modes on the SAME
copy.  The cross-copy value is fixed exactly by centre-of-mass factorisation: the c.o.m.
combination O^(1)_n + O^(2)_n is a free field the twist does not act on, so it must pass
through with no contraction; summing the four copy assignments gives
2 C_same + 2 C_cross = 0, i.e. C_cross = -C_same.  (Verified independently in the test
suite via the c.o.m. identity itself.)

Ordering convention for C^F.  1003.2746 eq (6.15) is derived for d^{+A}_{n1} d^{-B}_{n2}
with the PLUS-charge fermion written first, and the explicit 1/n1 refers to the plus-charge
fermion.  The unambiguous covariant statement, which is antisymmetric under exchanging the
two operators as it must be (both are pure creation operators, so they anticommute), is
    C^F(O1,O2) = -eps^{a1 a2} eps^{A1 A2} Phi(n_+, n_-)
with n_+ / n_- the mode of the plus-/minus-charge fermion, and eps^{a1 a2} carrying the
actual left-to-right order.  Phi is smooth at n_+ = 0 (fermion zero mode):
Gamma[1/2-n]/(n Gamma[-n]) -> -sqrt(pi).
"""
import mpmath as mp
import numpy as np

import r2_algebra as al
from r2_algebra import UNTW, TW, PLUS, MINUS, VAC, opp, eps_up, eps_lo, two_h

mp.mp.dps = 30

_SQPI = float(mp.sqrt(mp.pi))


# ---------------------------------------------------------------------------
# 1. Bogoliubov coefficients gamma^B, gamma^F  (1002.3132 eqs 5.12, 5.25, z0 = 1)
# ---------------------------------------------------------------------------
def gammaB(m, n):
    """gamma^B_{mn}, ACM integer labels m,n >= 1 (= |mu| of the twisted modes)."""
    if m < 1 or n < 1 or m % 2 == 0 or n % 2 == 0:
        return 0.0
    mp_, np_ = (m - 1) // 2, (n - 1) // 2
    num = mp.gamma(mp.mpf(3) / 2 + mp_) * mp.gamma(mp.mpf(3) / 2 + np_)
    den = mp.gamma(mp_ + 1) * mp.gamma(np_ + 1)
    # NOTE the explicit factor 2 in eq (5.12).  Two
    # independent confirmations: (i) the unevaluated sum eq (5.9) gives gamma^B_{11}=1/2;
    # (ii) the general-M,N closed form of 1405.0259 at M=N=1, s=s'=1/2 gives 1/2.
    return float(2 * num / den / (m * n * (1 + mp_ + np_) * mp.pi))


def gammaF(m, n):
    """gamma^F_{mn}, ACM eq (5.25).  Not symmetric in m,n."""
    if m < 1 or n < 1 or m % 2 == 0 or n % 2 == 0:
        return 0.0
    mp_, np_ = (m - 1) // 2, (n - 1) // 2
    num = mp.gamma(mp.mpf(3) / 2 + mp_) * mp.gamma(mp.mpf(3) / 2 + np_)
    den = mp.gamma(mp_ + 1) * mp.gamma(np_ + 1)
    return float(-num / den / (n * (1 + mp_ + np_) * mp.pi))


# ---------------------------------------------------------------------------
# 2. Single-mode transform coefficients f  (1003.2746 eqs 5.36, 5.45, 5.52)
#    n <= 0 untwisted mode; p the twisted integer label (p = 2n, or p odd <= -1).
#    copy = 1 -> a = +1, copy = 2 -> a = -1.
# ---------------------------------------------------------------------------
def _fscalar(n, p, kind):
    """Gamma-ratio part of f, at a = 1.  kind in {'B','F+','F-'}."""
    if p == 2 * n:
        return 0.5 + 0j
    if p % 2 == 0 or p > -1:
        return 0j
    pp = (p - 1) // 2          # p = 2 p' + 1, p' <= -1
    if kind == 'F+':
        num = mp.gamma(mp.mpf(1) / 2 - n) * mp.gamma(mp.mpf(1) / 2 - pp)
        rden = mp.rgamma(1 - n) * mp.rgamma(-pp)
    else:                       # 'B' and 'F-' share the same skeleton
        num = mp.gamma(mp.mpf(1) / 2 - n) * mp.gamma(-mp.mpf(1) / 2 - pp)
        rden = mp.rgamma(-n) * mp.rgamma(-pp)   # rgamma = 1/Gamma, zero at the poles
    if rden == 0:
        return 0j
    val = 1j / mp.pi * (num * rden) / (2 * n - 2 * pp - 1)
    return complex(val)


_FCACHE = {}


def fcoef(kind, n, p, copy):
    """f^{kind}_{a}[n,p] with a = +1 (copy 1) / -1 (copy 2)."""
    key = (kind, n, p, copy)
    v = _FCACHE.get(key)
    if v is not None:
        return v
    v = _fscalar(n, p, kind)
    if copy == 2 and p != 2 * n:
        v = -v            # a -> -a, odd power of a
    _FCACHE[key] = v
    return v


# ---------------------------------------------------------------------------
# 3. Pairwise Wick-contraction c-numbers  (1003.2746 eqs 6.7, 6.15)
# ---------------------------------------------------------------------------
def _K(n):
    """Gamma[1/2-n]/Gamma[-n]  (n <= 0 integer); K(0) = 0."""
    if n == 0:
        return 0.0
    return float(mp.gamma(mp.mpf(1) / 2 - n) / mp.gamma(-n))


def _Ktil(n):
    """K(n)/n, smooth at n = 0 where it equals -sqrt(pi)."""
    if n == 0:
        return -_SQPI
    return _K(n) / n


def CB_pair(o1, o2):
    """C^B for two bosonic creation ops o = ('b', copy, (A,Ad), n), n <= -1."""
    _, c1, (A1, Ad1), n1 = o1
    _, c2, (A2, Ad2), n2 = o2
    e = eps_lo(A1, A2) * eps_lo(Ad1, Ad2)
    if e == 0:
        return 0.0
    s = 1.0 if c1 == c2 else -1.0
    return e * s * _K(n1) * _K(n2) / (2 * np.pi * (n1 + n2))


def CF_pair(o1, o2):
    """C^F for two fermionic creation ops o = ('f', copy, (al,A), n), n <= 0.

    1003.2746 eq (6.15) reads C^{F,alpha A beta B}[n1,n2] = -eps^{alpha beta} eps^{AB} Phi,
    but it is derived (their eqs 6.9-6.14) specifically for the ordering with the
    PLUS-charge fermion written first, and the explicit 1/n1 in Phi refers to that
    plus-charge fermion.  The two epsilons together are SYMMETRIC under exchanging the two
    operators, so read literally the formula would make C^F symmetric -- inconsistent,
    since two creation operators anticommute and the twist map must therefore be
    antisymmetric in them.  Writing eq (6.15) in the canonical (plus-first) order,
    -eps^{+-} eps^{A_+ A_-} = +eps^{A_+ A_-}, gives the manifestly covariant form

        C^F(o1,o2) = eps^{A1 A2} Phi(n_+, n_-)        (zero unless alpha_1 != alpha_2)

    which is antisymmetric under o1 <-> o2 and reduces to eq (6.15) in ACM's ordering."""
    _, c1, (a1, A1), n1 = o1
    _, c2, (a2, A2), n2 = o2
    if a1 == a2:
        return 0.0
    e = eps_up(A1, A2)
    if e == 0:
        return 0.0
    if a1 == PLUS:
        npl, nmi = n1, n2
    else:
        npl, nmi = n2, n1
    s = 1.0 if c1 == c2 else -1.0
    # OVERALL SIGN: eq (6.15) as printed, reduced to the covariant form above, gives
    # +eps^{A1 A2} Phi.  Two independent exact identities that the twist map must satisfy --
    #   (A)  G^{+,tw}_{Adot,0} Sigma = Sigma G^{+(g)}_{Adot,0}   [1912.05567 (B.5) + (E.26)]
    #   (B)  J^{+,tw}_0 Sigma = Sigma J^{+(g)}_0                 [sigma_2^+ is a j=m=1/2 hw state]
    # -- force the opposite sign relative to C^B: (A) fixes the RATIO C^B : C^F = 1 : -1 and
    # (B) fixes C^F on its own (through the fermion-zero-mode pairs).  With this sign every
    # test in the suite is satisfied to machine zero and Gate 5 reproduces Guo-Mathur.
    return -e * s * _Ktil(npl) * _K(nmi) / (2 * np.pi * (npl + nmi))


def C_pair(o1, o2):
    if o1[0] != o2[0]:
        return 0.0
    return CB_pair(o1, o2) if o1[0] == 'b' else CF_pair(o1, o2)


# ---------------------------------------------------------------------------
# 4. |chi>, level graded
# ---------------------------------------------------------------------------
def _exp_bilinear(graded, coef, apply_pair, th_max):
    """multiply a level-graded dict by exp(coef * pair), pair = product of two twisted
    creation ops of total weight dw."""
    if coef == 0:
        return graded
    out = {l: dict(v) for l, v in graded.items()}
    cur = graded
    k = 1
    while True:
        nxt = {}
        for l, v in cur.items():
            w = apply_pair(v)
            if not w:
                continue
            dl = None
            for kk in w:
                dl = two_h(kk) - l
                break
            nl = l + dl
            if nl > th_max:
                continue
            tgt = nxt.setdefault(nl, {})
            for kk, c in w.items():
                tgt[kk] = tgt.get(kk, 0) + c * coef / k
        nxt = {l: {kk: c for kk, c in v.items() if abs(c) > 1e-16} for l, v in nxt.items()}
        nxt = {l: v for l, v in nxt.items() if v}
        if not nxt:
            break
        for l, v in nxt.items():
            tgt = out.setdefault(l, {})
            for kk, c in v.items():
                tgt[kk] = tgt.get(kk, 0) + c
        cur = nxt
        k += 1
        if k > 60:
            break
    return out


_CHI_CACHE = {}


def build_chi(th_max):
    """|chi> = sigma_2^+(0)|0_R^->^(1)|0_R^->^(2) as {level two_h: vector} in TW,
    truncated at two_h <= th_max.  1002.3132 eq (4.20) with C_11 = 1."""
    if th_max in _CHI_CACHE:
        return _CHI_CACHE[th_max]
    graded = {0: {VAC: 1.0 + 0j}}
    odd = list(range(1, th_max + 1, 2))
    # bosonic:  gamma^B_mn [ -alpha_{++,-m} alpha_{--,-n} + alpha_{-+,-m} alpha_{+-,-n} ]
    for m in odd:
        for n in odd:
            if m + n > th_max:
                continue
            g = gammaB(m, n)
            if g == 0:
                continue
            for (sgn, s1, s2) in ((-1.0, (PLUS, PLUS), (MINUS, MINUS)),
                                  (+1.0, (MINUS, PLUS), (PLUS, MINUS))):
                def ap(v, m=m, n=n, s1=s1, s2=s2):
                    w = TW.alpha(v, -n, s2[0], s2[1])
                    return TW.alpha(w, -m, s1[0], s1[1])
                graded = _exp_bilinear(graded, sgn * g, ap, th_max)
    # fermionic: gamma^F_mn [ d^{++}_{-m} d^{--}_{-n} - d^{+-}_{-m} d^{-+}_{-n} ]
    for m in odd:
        for n in odd:
            if m + n > th_max:
                continue
            g = gammaF(m, n)
            if g == 0:
                continue
            for (sgn, s1, s2) in ((+1.0, (PLUS, PLUS), (MINUS, MINUS)),
                                  (-1.0, (PLUS, MINUS), (MINUS, PLUS))):
                def ap(v, m=m, n=n, s1=s1, s2=s2):
                    w = TW.d(v, -n, s2[0], s2[1])
                    return TW.d(w, -m, s1[0], s1[1])
                graded = _exp_bilinear(graded, sgn * g, ap, th_max)
    graded = {l: v for l, v in graded.items() if v}
    _CHI_CACHE[th_max] = graded
    return graded


# ---------------------------------------------------------------------------
# 5. the twist map as a linear operator
# ---------------------------------------------------------------------------
def mono_ops(mono):
    """Ordered list of creation operators of an UNTWISTED monomial:
    bosons first (they commute with everything), then fermions in canonical order.
    op = ('b', copy, (A,Ad), n) or ('f', copy, (al,A), n) with n = mu/2 <= 0."""
    bos, fer = mono
    ops = []
    for (k, cnt) in bos:
        mu, cp, Ai, Adi = k
        for _ in range(cnt):
            ops.append(('b', cp, (Ai, Adi), mu // 2))
    for k in fer:
        mu, cp, ai, Ai = k
        ops.append(('f', cp, (ai, Ai), mu // 2))
    return ops


def _pairings(ops):
    """Yield (coeff, remaining_ops) over all sets of Wick pairs, with fermionic signs."""
    if not ops:
        yield (1.0 + 0j, ())
        return
    head, rest = ops[0], ops[1:]
    for (c, tail) in _pairings(rest):
        yield (c, (head,) + tail)
    for j, other in enumerate(rest):
        cval = C_pair(head, other)
        if cval == 0:
            continue
        # sign from moving `other` left past rest[0:j]
        if other[0] == 'f':
            nf = sum(1 for o in rest[:j] if o[0] == 'f')
            sgn = -1.0 if (nf & 1) else 1.0
        else:
            sgn = 1.0
        sub = rest[:j] + rest[j + 1:]
        for (c, tail) in _pairings(sub):
            yield (cval * sgn * c, tail)


def _allowed_p(kind, n, budget):
    """Twisted mode labels p reachable from untwisted mode n with level budget."""
    out = []
    if abs(2 * n) <= budget:
        out.append(2 * n)
    p = -1
    while abs(p) <= budget:
        out.append(p)
        p -= 2
    return out


def _apply_hat(graded, op, th_target):
    """Apply the transformed operator sum_p f[n,p] O_p to a level-graded dict."""
    kind, cp, sp, n = op
    fkind = 'B' if kind == 'b' else ('F+' if sp[0] == PLUS else 'F-')
    out = {}
    for lvl, v in graded.items():
        budget = th_target - lvl
        if budget < 0:
            continue
        for p in _allowed_p(kind, n, budget):
            c = fcoef(fkind, n, p, cp)
            if c == 0:
                continue
            if kind == 'b':
                if p == 0:
                    continue
                w = TW.alpha(v, p, sp[0], sp[1])
            else:
                w = TW.d(v, p, sp[0], sp[1])
            if not w:
                continue
            nl = lvl + abs(p)
            tgt = out.setdefault(nl, {})
            for kk, cc in w.items():
                tgt[kk] = tgt.get(kk, 0) + c * cc
    return {l: {k: c for k, c in v.items() if abs(c) > 1e-15} for l, v in out.items() if v}


def twist_mono(mono, th_target=None):
    """sigma_2^+(0) applied to one untwisted monomial, projected on twisted level
    two_h == th_target (default: the monomial's own level, i.e. the P-projected map)."""
    if th_target is None:
        th_target = two_h(mono)
    ops = mono_ops(mono)
    chi = build_chi(th_target)
    total = {}
    for (coeff, rest) in _pairings(ops):
        if coeff == 0:
            continue
        graded = {l: v for l, v in chi.items() if l <= th_target}
        for op in reversed(rest):
            graded = _apply_hat(graded, op, th_target)
            if not graded:
                break
        v = graded.get(th_target)
        if not v:
            continue
        for k, c in v.items():
            total[k] = total.get(k, 0) + coeff * c
    return {k: c for k, c in total.items() if abs(c) > 1e-13}


_TW_MONO_CACHE = {}


def twist(vec, th_target=None):
    """Linear extension of twist_mono to a vector in UNTW."""
    if not vec:
        return {}
    if th_target is None:
        ths = {two_h(k) for k in vec}
        assert len(ths) == 1, 'twist() needs a level-homogeneous vector or explicit th_target'
        th_target = ths.pop()
    out = {}
    for m, c in vec.items():
        key = (m, th_target)
        w = _TW_MONO_CACHE.get(key)
        if w is None:
            w = twist_mono(m, th_target)
            _TW_MONO_CACHE[key] = w
        for k, cc in w.items():
            out[k] = out.get(k, 0) + c * cc
    return {k: c for k, c in out.items() if abs(c) > 1e-13}
