"""Route 2 Phase A: chiral mode algebra + Fock spaces for Sym^2(T^4) at N = 2, R sector.

Conventions are those of Guo-Mathur 1912.05567 App. A
(= Hampton-Mathur-Zadeh 1804.10097 App. A), which is also the convention of the
Avery-Chowdhury-Mathur papers 1002.3132 / 1003.2746 used for the twist data.

Conventions (1912.05567 eqs A.1-A.4, A.8-A.10)
---------------------------------------------
  eps_{+-} = +1,  eps^{+-} = -1                                            (A.1)
  [alpha_{A Adot,m}, alpha_{B Bdot,n}] = -(c/6) m eps_{AB} eps_{Adot Bdot} delta_{m+n,0}
  {d^{alpha A}_r, d^{beta B}_s}        = -(c/6) eps^{alpha beta} eps^{AB}  delta_{r+s,0}
  G^{alpha}_{Adot,r} = -(i/n_i) sum_n d^{alpha A}_{r-n} alpha_{A Adot,n}   (A.9)
with c = 6 n_i for a component string of winding n_i.  So

  untwisted, one singly wound copy : n_i = 1, c/6 = 1, modes in Z
  twisted,  one doubly wound copy  : n_i = 2, c/6 = 2, modes in Z/2

Mode bookkeeping
----------------
Every mode is stored as an INTEGER mu = 2 * (mode number).  Untwisted copies therefore
carry even mu, the twisted sector any mu.  A creation slot has mu < 0 (fermion zero modes
d^{+A}_0 have mu = 0).  two_h = sum |mu| = 2 * (level above the R ground state).  Both
sectors have R-ground-state h = c/24 = 1/2 (untwisted: 2 x 6/24; twisted: 12/24), so
"level" means the same thing on both sides and the twist map is level-graded.

R ground state of each component string is |0_R^->:  d^{-A}_0 |0_R^-> = 0, alpha_{..,0}|0_R^-> = 0.
Creation operators: alpha_{A Adot,-|mu|/2} (mu<0), d^{alpha A}_{-|mu|/2} (mu<0), d^{+A}_0.

Hermitian conjugation (fixed by positivity, and verified against 1912.05567 App. B):
  (alpha_{A Adot,m})^dag = -eps^{A,-A} eps^{Adot,-Adot} alpha_{-A,-Adot,-m}
  (d^{alpha A}_r)^dag    = -eps_{alpha,-alpha} eps_{A,-A} d^{-alpha,-A}_{-r}
These reproduce (G^{alpha}_{Adot,r})^dag = eps_{alpha beta} eps^{Adot Bdot} G^{beta}_{Bdot,-r}
= 1912.05567 (B.1).

State representation
--------------------
A monomial is a pair (bos, fer):
  bos = tuple of ((mu, copy, Ai, Adi), count), sorted, mu < 0, count >= 1
  fer = tuple of (mu, copy, ai, Ai),          sorted ascending, mu <= 0
with Ai/Adi/ai in {0,1} for {'+','-'}.  Fermion operators are understood written left to
right in this sorted (ascending) order.  A vector is a dict {monomial: complex}.
"""
import itertools
import numpy as np

PLUS, MINUS = 0, 1
CH = ('+', '-')


def opp(a):
    return 1 - a


def eps_up(a, b):
    """eps^{ab}: eps^{+-} = -1, eps^{-+} = +1."""
    if a == b:
        return 0
    return -1 if a == PLUS else 1


def eps_lo(a, b):
    """eps_{ab}: eps_{+-} = +1, eps_{-+} = -1."""
    if a == b:
        return 0
    return 1 if a == PLUS else -1


VAC = ((), ())


def mrange(mu_par, mu_max):
    """All allowed mode labels mu with |mu| <= mu_max, aligned to the sector's parity
    (so that 0 is always included when mu_par = 2, whatever the parity of mu_max)."""
    lo = -(mu_max // mu_par) * mu_par
    return range(lo, mu_max + 1, mu_par)


# ---------------------------------------------------------------------------
# monomial helpers
# ---------------------------------------------------------------------------
def _bos_add(bos, key):
    lst = list(bos)
    for i, (k, c) in enumerate(lst):
        if k == key:
            lst[i] = (k, c + 1)
            return tuple(lst)
    lst.append((key, 1))
    lst.sort()
    return tuple(lst)


def _bos_count(bos, key):
    for (k, c) in bos:
        if k == key:
            return c
    return 0


def _bos_remove(bos, key):
    lst = []
    for (k, c) in bos:
        if k == key:
            if c > 1:
                lst.append((k, c - 1))
        else:
            lst.append((k, c))
    return tuple(lst)


def _fer_sign(fer, key):
    """(-1)^{# fermion slots strictly below key in canonical order}."""
    n = 0
    for k in fer:
        if k < key:
            n += 1
        else:
            break
    return -1 if (n & 1) else 1


def two_h(mono):
    bos, fer = mono
    t = 0
    for (k, c) in bos:
        t += (-k[0]) * c
    for k in fer:
        t += -k[0]
    return t


class Sector:
    """A chiral R-sector Fock space.

    copies : tuple of copy labels present (untwisted: (1,2); twisted: (0,))
    cfac   : c/6 of each component string (1 untwisted, 2 twisted)
    mu_par : 2 if only even mu allowed (untwisted, integer modes), 1 if any mu (twisted)
    """

    def __init__(self, name, copies, cfac, mu_par):
        self.name = name
        self.copies = tuple(copies)
        self.cfac = cfac
        self.mu_par = mu_par
        # SU(2)_L charge of the sector's R ground state, in units of 1/2
        self.vac_2jL = -len(self.copies)
        self.c = 6 * cfac * len(self.copies)
        # optional zero-mode boson eigenvalue (momentum) per copy:
        #   mom = {copy: {(A, Adot): complex}}   (None = zero momentum)
        self.mom = None

    # -- basic charges -----------------------------------------------------
    def charges(self, mono):
        """(two_h, 2jL, 2j1, 2j2) with 2jL including the R-ground-state offset."""
        bos, fer = mono
        th = 0
        jL, j1, j2 = self.vac_2jL, 0, 0
        for (k, c) in bos:
            mu, cp, Ai, Adi = k
            th += (-mu) * c
            # alpha_{A Adot} carries LOWER SU(2)_1 / SU(2)_2 indices, hence charge -A/2,
            # -Adot/2 (fixed by G^{alpha}_{Adot} = -(i/n) d^{alpha A} alpha_{A Adot} being an
            # SU(2)_1 singlet: 2j1(d^{alpha A}) = +A and the contraction must be neutral).
            j1 += (-1 if Ai == PLUS else 1) * c
            j2 += (-1 if Adi == PLUS else 1) * c
        for k in fer:
            mu, cp, ai, Ai = k
            th += -mu
            jL += (1 if ai == PLUS else -1)
            j1 += (1 if Ai == PLUS else -1)
        return (th, jL, j1, j2)

    # -- monomial norms ----------------------------------------------------
    def mono_norm2(self, mono):
        """<mono|mono>; distinct monomials are orthogonal (each creation slot has a
        unique conjugate annihilator)."""
        bos, fer = mono
        v = 1.0
        for (k, c) in bos:
            mu = k[0]
            slot = self.cfac * (-mu) / 2.0
            v *= slot ** c
            for i in range(2, c + 1):
                v *= i
        v *= self.cfac ** len(fer)
        return v

    def ip(self, v1, v2):
        """<v1|v2>."""
        if len(v1) > len(v2):
            v1, v2 = v2, v1
            conj_first = False
        else:
            conj_first = True
        tot = 0.0 + 0.0j
        for k, c in v1.items():
            c2 = v2.get(k)
            if c2 is None:
                continue
            tot += (np.conj(c) * c2 if conj_first else np.conj(c2) * c) * self.mono_norm2(k)
        return tot

    def norm2(self, v):
        return self.ip(v, v).real

    # -- elementary mode operators ----------------------------------------
    def alpha(self, vec, mu, A, Ad, copy=None):
        """alpha^{(copy)}_{A Adot, mu/2} applied to a vector."""
        if copy is None:
            copy = self.copies[0]
        out = {}
        if mu == 0:
            if self.mom is None:
                return out
            v = self.mom.get(copy, {}).get((A, Ad), 0.0)
            if v == 0:
                return out
            return {k: c * v for k, c in vec.items()}
        if mu % self.mu_par != 0:
            raise ValueError('mode parity')
        if mu < 0:
            key = (mu, copy, A, Ad)
            for st, c in vec.items():
                bos, fer = st
                nk = (_bos_add(bos, key), fer)
                out[nk] = out.get(nk, 0) + c
        else:
            tkey = (-mu, copy, opp(A), opp(Ad))
            base = -self.cfac * (mu / 2.0) * eps_lo(A, opp(A)) * eps_lo(Ad, opp(Ad))
            for st, c in vec.items():
                bos, fer = st
                cnt = _bos_count(bos, tkey)
                if cnt == 0:
                    continue
                nk = (_bos_remove(bos, tkey), fer)
                out[nk] = out.get(nk, 0) + c * base * cnt
        return {k: v for k, v in out.items() if v != 0}

    def d(self, vec, mu, al, A, copy=None):
        """d^{(copy) al A}_{mu/2} applied to a vector."""
        if copy is None:
            copy = self.copies[0]
        if mu % self.mu_par != 0:
            raise ValueError('mode parity')
        out = {}
        creation = (mu < 0) or (mu == 0 and al == PLUS)
        if creation:
            key = (mu, copy, al, A)
            for st, c in vec.items():
                bos, fer = st
                if key in fer:
                    continue
                sgn = _fer_sign(fer, key)
                nf = tuple(sorted(fer + (key,)))
                nk = (bos, nf)
                out[nk] = out.get(nk, 0) + c * sgn
        else:
            tkey = (-mu, copy, opp(al), opp(A))
            base = -self.cfac * eps_up(al, opp(al)) * eps_up(A, opp(A))
            for st, c in vec.items():
                bos, fer = st
                if tkey not in fer:
                    continue
                sgn = _fer_sign(fer, tkey)
                nf = tuple(k for k in fer if k != tkey)
                nk = (bos, nf)
                out[nk] = out.get(nk, 0) + c * base * sgn
        return {k: v for k, v in out.items() if v != 0}

    # -- composite generators ---------------------------------------------
    def G(self, vec, al, Ad, r2, copy=None, mu_max=None):
        """G^{al}_{Adot, r2/2} = -(i/cfac) sum_n d^{al A}_{r2/2-n} alpha_{A Adot,n},
        i.e. in mu units  -(i/cfac) sum_mu d^{al A}_{(r2-mu)/2} alpha_{A Adot, mu/2}.
        mu_max bounds the |mu| range (must exceed the level of every state involved)."""
        if copy is None:
            copy = self.copies[0]
        if mu_max is None:
            mu_max = max((two_h(k) for k in vec), default=0) + abs(r2) + 2
        out = {}
        pref = -1j / self.cfac
        for mu in mrange(self.mu_par, mu_max):
            if mu == 0 and self.mom is None:
                continue
            for A in (PLUS, MINUS):
                v1 = self.alpha(vec, mu, A, Ad, copy=copy)
                if not v1:
                    continue
                v2 = self.d(v1, r2 - mu, al, A, copy=copy)
                for k, c in v2.items():
                    out[k] = out.get(k, 0) + pref * c
        return {k: v for k, v in out.items() if abs(v) > 1e-14}

    def _ff_bilinear(self, vec, terms, m2, mu_max):
        """sum over pairs: coeff * sum_{r} :d^{a1 A1}_{r/2} d^{a2 A2}_{(m2-r)/2}:  ,
        terms = list of (coeff, a1, A1, a2, A2).  Normal ordered."""
        out = {}
        for (coeff, a1, A1, a2, A2) in terms:
            for r in mrange(self.mu_par, mu_max):
                s = m2 - r
                if abs(s) > mu_max:
                    continue
                cre1 = (r < 0) or (r == 0 and a1 == PLUS)
                cre2 = (s < 0) or (s == 0 and a2 == PLUS)
                if (not cre1) and cre2:
                    # normal order: swap (fermionic sign -1)
                    v1 = self.d(vec, r, a1, A1)
                    v2 = self.d(v1, s, a2, A2)
                    sg = -1.0
                else:
                    v1 = self.d(vec, s, a2, A2)
                    v2 = self.d(v1, r, a1, A1)
                    sg = 1.0
                for k, c in v2.items():
                    out[k] = out.get(k, 0) + coeff * sg * c
        return {k: v for k, v in out.items() if abs(v) > 1e-14}

    def Jp(self, vec, m2, mu_max=None):
        """J^+_{m2/2} = (1/cfac) sum_r d^{++}_r d^{+-}_{m-r}."""
        if mu_max is None:
            mu_max = max((two_h(k) for k in vec), default=0) + abs(m2) + 2
        return self._ff_bilinear(vec, [(1.0 / self.cfac, PLUS, PLUS, PLUS, MINUS)], m2, mu_max)

    def Jm(self, vec, m2, mu_max=None):
        """J^-_{m2/2} = (1/cfac) sum_r d^{--}_r d^{-+}_{m-r}."""
        if mu_max is None:
            mu_max = max((two_h(k) for k in vec), default=0) + abs(m2) + 2
        return self._ff_bilinear(vec, [(1.0 / self.cfac, MINUS, MINUS, MINUS, PLUS)], m2, mu_max)

    def J3(self, vec, m2, mu_max=None):
        """J^3_{m2/2} = -(1/2cfac) sum_r [ d^{++}_r d^{--}_{m-r} + d^{-+}_r d^{+-}_{m-r} ].

        The r = 0 fermion-zero-mode term is the usual R-sector normal-ordering ambiguity:
        normal ordering (needed for r != 0 to keep the sum finite) drops exactly -1/2 per
        component string, which is the SU(2)_L charge of the R ground state |0_R^->.  It is
        restored here as an explicit constant for m2 = 0, so that J^3_0 measures the true
        charge (verified: [J^+_0, J^-_0] = 2 J^3_0 then holds exactly)."""
        if mu_max is None:
            mu_max = max((two_h(k) for k in vec), default=0) + abs(m2) + 2
        t = [(-0.5 / self.cfac, PLUS, PLUS, MINUS, MINUS),
             (-0.5 / self.cfac, MINUS, PLUS, PLUS, MINUS)]
        out = self._ff_bilinear(vec, t, m2, mu_max)
        if m2 == 0:
            for k, c in vec.items():
                out[k] = out.get(k, 0) - 0.5 * c
        return {k: v for k, v in out.items() if abs(v) > 1e-14}


# ---------------------------------------------------------------------------
# multi-copy wrappers (the untwisted (1,1) sector has two independent copies whose
# fermions mutually anticommute -- handled automatically since `copy` is part of the
# canonically-ordered fermion key)
# ---------------------------------------------------------------------------
UNTW = Sector('untwisted(1,1)', copies=(1, 2), cfac=1, mu_par=2)
TW = Sector('twisted(2)', copies=(0,), cfac=2, mu_par=1)


def G_global(sec, vec, al, Ad, r2, mu_max=None):
    """Global (sum over copies) supercharge zero/other mode."""
    out = {}
    for cp in sec.copies:
        v = sec.G(vec, al, Ad, r2, copy=cp, mu_max=mu_max)
        for k, c in v.items():
            out[k] = out.get(k, 0) + c
    return {k: v for k, v in out.items() if abs(v) > 1e-14}


def _bilin_global(sec, vec, terms, m2, mu_max):
    out = {}
    for cp in sec.copies:
        # temporarily bind the copy by wrapping d
        sub = _CopyView(sec, cp)
        v = sub._ff_bilinear(vec, terms, m2, mu_max)
        for k, c in v.items():
            out[k] = out.get(k, 0) + c
    return {k: v for k, v in out.items() if abs(v) > 1e-14}


class _CopyView:
    """Sector view with a fixed default copy (so the generic bilinear helpers work
    per copy)."""

    def __init__(self, sec, copy):
        self.sec = sec
        self.copy = copy
        self.cfac = sec.cfac
        self.mu_par = sec.mu_par

    def d(self, vec, mu, al, A):
        return self.sec.d(vec, mu, al, A, copy=self.copy)

    def alpha(self, vec, mu, A, Ad):
        return self.sec.alpha(vec, mu, A, Ad, copy=self.copy)

    _ff_bilinear = Sector._ff_bilinear


def Jp_global(sec, vec, m2, mu_max=None):
    if mu_max is None:
        mu_max = max((two_h(k) for k in vec), default=0) + abs(m2) + 2
    return _bilin_global(sec, vec, [(1.0 / sec.cfac, PLUS, PLUS, PLUS, MINUS)], m2, mu_max)


def Jm_global(sec, vec, m2, mu_max=None):
    if mu_max is None:
        mu_max = max((two_h(k) for k in vec), default=0) + abs(m2) + 2
    return _bilin_global(sec, vec, [(1.0 / sec.cfac, MINUS, MINUS, MINUS, PLUS)], m2, mu_max)


def J3_global(sec, vec, m2, mu_max=None):
    if mu_max is None:
        mu_max = max((two_h(k) for k in vec), default=0) + abs(m2) + 2
    t = [(-0.5 / sec.cfac, PLUS, PLUS, MINUS, MINUS),
         (-0.5 / sec.cfac, MINUS, PLUS, PLUS, MINUS)]
    out = _bilin_global(sec, vec, t, m2, mu_max)
    if m2 == 0:
        for k, c in vec.items():
            out[k] = out.get(k, 0) - 0.5 * len(sec.copies) * c
    return {k: v for k, v in out.items() if abs(v) > 1e-14}


# ---------------------------------------------------------------------------
# vector helpers
# ---------------------------------------------------------------------------
def vzero():
    return {}


def vadd(*pairs):
    """vadd((v1,s1),(v2,s2),...) -> sum s_i v_i."""
    out = {}
    for v, s in pairs:
        for k, c in v.items():
            out[k] = out.get(k, 0) + s * c
    return {k: v for k, v in out.items() if abs(v) > 1e-14}


def vscale(v, s):
    return {k: c * s for k, c in v.items()}


def vlevel(v, th):
    return {k: c for k, c in v.items() if two_h(k) == th}


def vclean(v, tol=1e-13):
    return {k: c for k, c in v.items() if abs(c) > tol}


def normalize(sec, v):
    n = np.sqrt(sec.norm2(v))
    return vscale(v, 1.0 / n)


# ---------------------------------------------------------------------------
# basis enumeration for a sector at fixed level
# ---------------------------------------------------------------------------
def enumerate_basis(sec, th, mu_min=1):
    """All monomials of the sector with two_h == th (mu_min = smallest |mu| allowed for
    an oscillator; fermion zero modes mu = 0 are always included as optional slots)."""
    # fermion zero-mode slots (mu = 0, alpha = +) for each copy and each A
    zslots = [(0, cp, PLUS, A) for cp in sec.copies for A in (PLUS, MINUS)]
    # oscillator slots
    fslots = [(-mu, cp, al, A) for mu in range(mu_min, th + 1, sec.mu_par)
              for cp in sec.copies for al in (PLUS, MINUS) for A in (PLUS, MINUS)]
    bslots = [(-mu, cp, A, Ad) for mu in range(mu_min, th + 1, sec.mu_par)
              for cp in sec.copies for A in (PLUS, MINUS) for Ad in (PLUS, MINUS)]
    res = []

    def rec_b(i, rem, acc):
        if rem == 0:
            yield tuple(sorted(acc))
            return
        if i >= len(bslots):
            return
        k = bslots[i]
        w = -k[0]
        cmax = rem // w
        for c in range(0, cmax + 1):
            acc2 = acc + ([(k, c)] if c else [])
            yield from rec_b(i + 1, rem - c * w, acc2)

    allf = zslots + fslots

    def rec_f(i, rem, acc):
        if i >= len(allf):
            if rem >= 0:
                yield (rem, tuple(sorted(acc)))
            return
        k = allf[i]
        w = -k[0]
        yield from rec_f(i + 1, rem, acc)
        if w <= rem:
            yield from rec_f(i + 1, rem - w, acc + [k])

    for rem, fer in rec_f(0, th, []):
        for bos in rec_b(0, rem, []):
            res.append((bos, fer))
    return res
