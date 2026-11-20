"""Route 2 Phase C: chiral spaces, LEVEL-CHANGING chiral maps T_Adot, and the Gtilde action
on them.

Beyond the radius of degenerate perturbation theory the truncated Hamiltonian needs the FULL
D_0 = int dsigma D(sigma), which conserves the spin h - hbar but NOT the energy: at w_0 = 0
(z_0 = 1) the amplitude <t|D_0|u> is nonzero for every pair of levels.  The existing Phase-A
machinery already computes those amplitudes -- the sum rule
sum_{t at E = 2n} |<t|D_0|0>|^2 = 4 pi^2 n^2, IS a level-changing statement, and it fixed
kappa = sqrt(2) pi -- but `r2_lift.T_Adot` projects on the source level.  Here the twist map
is evaluated once per source state and kept level-graded.

Chiral spaces (relative theory, R sector, massive N = 4 primaries)
------------------------------------------------------------------
    Wtw(h)   twisted, dim p_t = 1, 2, 6, 14, 28, 54, 98 at h = 0, 1/2, ..., 3
    Vun(h)   untwisted, = direct sum over momentum vectors a in Z^4 with |a|^2 <= 2h and
             |a|^2 = 2h mod 2, of the oscillator primaries at level h - |a|^2/2 on top of
             the relative momentum state (dim p_3(k) = 1, 3, 9, 22, ... for a != 0, and
             e + o for a = 0).

Gtilde acts on the momentum label by a -> R_m a (r2c_group.Rm) and on the oscillator content
by the index rotation.  Only ONE representative per momentum orbit is ever twisted: for
a = R a_0 the basis of V_a is DEFINED as U(g_a) V_{a_0} and then, from the exact covariance
U T_Adot = sum_Cdot Yb[Cdot,Adot] T_Cdot U (verified in the gate),

    A^{(a)}_Adot = U_W(g_a) sum_Cdot Yb(g_a^{-1})[Cdot,Adot] A^{(a_0)}_Cdot .
"""
import itertools
import os
import numpy as np

import r2_algebra as al
import r2_rel as rl
import r2_twist as tw
import r2_mom as mo
import r2c_group as gr
from r2_algebra import UNTW, TW, PLUS, MINUS, VAC, two_h, vadd


# ---------------------------------------------------------------------------
# level-graded twist map
# ---------------------------------------------------------------------------
def twist_mono_all(mono, th_max, pd=None):
    """sigma_2^+(0) on one untwisted monomial (optionally on top of the relative momentum
    state with p_{A Adot} = pd), as {twisted level: vector}, all levels <= th_max."""
    ops = tw.mono_ops(mono)
    if pd is None:
        chi = tw.build_chi(th_max)
        pairings = tw._pairings(ops)
    else:
        chi = mo.build_chi_pd(pd, th_max)
        pairings = mo._pairings_src(ops, pd)
    total = {}
    for (coeff, rest) in pairings:
        if coeff == 0:
            continue
        graded = {l: v for l, v in chi.items() if l <= th_max}
        for op in reversed(rest):
            graded = tw._apply_hat(graded, op, th_max)
            if not graded:
                break
        for l, v in graded.items():
            t = total.setdefault(l, {})
            for k, c in v.items():
                t[k] = t.get(k, 0) + coeff * c
    return {l: {k: c for k, c in v.items() if abs(c) > 1e-14}
            for l, v in total.items() if v}


_TMA = {}


def twist_vec_all(vec, th_max, pd=None, Nf=1.0):
    key_pd = None if pd is None else tuple(sorted((k, complex(v)) for k, v in pd.items()))
    out = {}
    for m, c in vec.items():
        key = (m, th_max, key_pd)
        w = _TMA.get(key)
        if w is None:
            w = twist_mono_all(m, th_max, pd=pd)
            _TMA[key] = w
        for l, v in w.items():
            t = out.setdefault(l, {})
            for k, cc in v.items():
                t[k] = t.get(k, 0) + Nf * c * cc
    return {l: {k: c for k, c in v.items() if abs(c) > 1e-13}
            for l, v in out.items() if v}


def T_Adot_all(vec, Ad, th_src, th_max, p4=None):
    """{twisted level: T_Adot |vec>} for every level <= th_max.  `vec` must be a
    level-homogeneous untwisted vector at oscillator level (th_src - 2 h_p)/2; th_src is its
    total two_h (oscillator + momentum)."""
    if not vec:
        return {}
    # Two points (found via the twisted-source weights):
    #  (i) the supercurrent mode sum must cover the SOURCE level too (th_src may exceed th_max
    #      when a high source is mapped onto a low twisted level);
    # (ii) G^-_0 preserves the OSCILLATOR level of `vec`; two_h(mono) is the oscillator level only,
    #      so the consistency filter must compare with the oscillator level, not with th_src
    #      (= oscillator + momentum).  A filter `two_h(k) == th_src` would silently drop the
    #      whole sigma(G^-_0 v) term in every momentum sector with oscillators.
    mm = max(th_max, th_src) + 6
    th_osc = max(two_h(k) for k in vec)
    if p4 is None:
        pd, Nf, ctx = None, 1.0, None
    else:
        pd, Nf, ctx = mo.pdict(p4), mo.Nfac(p4), mo.mom_context(p4)
    save = UNTW.mom
    UNTW.mom = ctx
    try:
        g = al.G_global(UNTW, vec, MINUS, Ad, 0, mu_max=mm)
        g = {k: c for k, c in g.items() if two_h(k) == th_osc}
    finally:
        UNTW.mom = save
    A = twist_vec_all(vec, th_max, pd=pd, Nf=Nf)
    B = twist_vec_all(g, th_max, pd=pd, Nf=Nf) if g else {}
    out = {}
    for l in set(A) | set(B):
        a = TW.G(A.get(l, {}), MINUS, Ad, 0, mu_max=mm) if l in A else {}
        a = {k: c for k, c in a.items() if two_h(k) == l}
        d = vadd((a, 1.0), (B.get(l, {}), -1.0))
        d = {k: c for k, c in d.items() if abs(c) > 1e-12}
        if d:
            out[l] = d
    return out


# ---------------------------------------------------------------------------
# chiral bases
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Two versions of the state space (tracker design question D1)
#   'prim' : only massive (x) massive N = 4 primaries of the undeformed algebra
#   'full' : every state of the relative theory in the same SU(2)_L x SU(2)_R
#            lowest-weight sector (2 j_L = -2, J^-_0 v = 0), i.e. N = 4 DESCENDANTS
#            included.  D_0 is an SU(2)_L x SU(2)_R singlet, so this sector is exactly
#            the "full truncated space" with every remaining symmetry resolved.
# ---------------------------------------------------------------------------
def _lw_only(sec, th, parity=None, p4=None):
    """States with 2 j_L = -2 and J^-_0 v = 0 -- no primary conditions."""
    if sec is UNTW:
        labs = rl.rel_untw_labels(th)
        if parity is not None:
            labs = [l for l in labs if (-1) ** l[2] == parity]
        cand = [rl.rel_untw_vector(l) for l in labs]
        cand = [v for v in cand if v]
    else:
        cand = [{m: 1.0 + 0j} for m in rl.rel_tw_basis(th)]
    cand = [v for v in cand if sec.charges(next(iter(v)))[1] == -2]
    if not cand:
        return []
    save = UNTW.mom
    if p4 is not None:
        UNTW.mom = mo.mom_context(p4)
    try:
        basis = rl.gram_orthonormal(sec, cand)
        mm = th + 8
        out = rl.common_kernel(sec, basis,
                               [lambda v: al.Jm_global(sec, v, 0, mu_max=mm)])
    finally:
        UNTW.mom = save
    return out


_TWB = {}


def tw_basis(th, mode='prim'):
    """Orthonormal twisted relative chiral basis at two_h = th."""
    key = (th, mode)
    if key not in _TWB:
        _TWB[key] = (rl.massive_primaries(TW, th) if mode == 'prim'
                     else _lw_only(TW, th))
    return _TWB[key]


_UNB = {}


def untw_basis_osc(th, mode='prim'):
    """Orthonormal untwisted relative chiral basis (a = 0) at two_h = th, BOTH Z_2
    parities, ordered even-first."""
    key = (th, mode)
    if key not in _UNB:
        if mode == 'prim':
            e = rl.massive_primaries(UNTW, th, parity=+1)
            o = rl.massive_primaries(UNTW, th, parity=-1)
        else:
            e = _lw_only(UNTW, th, parity=+1)
            o = _lw_only(UNTW, th, parity=-1)
        _UNB[key] = (e + o, len(e), len(o))
    return _UNB[key]


_UNM = {}


def untw_basis_mom(a, th_osc, mode='prim'):
    """Orthonormal untwisted chiral basis in momentum sector a at oscillator level
    th_osc/2."""
    key = (tuple(a), th_osc, mode)
    if key not in _UNM:
        p = [x / np.sqrt(2.0) for x in a]
        _UNM[key] = (mo.massive_primaries_mom(p, th_osc) if mode == 'prim'
                     else _lw_only(UNTW, th_osc, p4=p))
    return _UNM[key]


# ---------------------------------------------------------------------------
# momentum orbits
# ---------------------------------------------------------------------------
def lattice_vectors(n2max):
    R = int(np.floor(np.sqrt(n2max))) + 1
    out = []
    for v in itertools.product(range(-R, R + 1), repeat=4):
        if sum(x * x for x in v) <= n2max:
            out.append(v)
    return out


class MomOrbits:
    """Gtilde-orbits of momentum vectors with |a|^2 <= n2max."""

    def __init__(self, n2max):
        g = gr.G()
        self.n2max = n2max
        vecs = lattice_vectors(n2max)
        self.rep = {}        # a -> orbit representative
        self.coset = {}      # a -> group index g with Rm[g] a_0 = a
        self.orbits = []     # list of (a_0, [members])
        self.stab = {}       # a_0 -> list of group indices stabilising a_0
        seen = set()
        for a in vecs:
            if a in seen:
                continue
            a0 = a
            members = []
            stab = []
            for gi in range(g.n):
                b = tuple(int(x) for x in (g.Rm[gi] @ np.array(a0)))
                if b == a0:
                    stab.append(gi)
                if b in seen and b != a0:
                    continue
                if b not in self.rep:
                    self.rep[b] = a0
                    self.coset[b] = gi
                    members.append(b)
                    seen.add(b)
            self.orbits.append((a0, members))
            self.stab[a0] = stab

    def reps(self):
        return [o[0] for o in self.orbits]
