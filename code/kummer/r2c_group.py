"""Route 2 Phase C, PART 1: the action of the point-group double cover Gtilde (order 384)
on the untwisted (two-copy, relative) and twisted chiral Fock spaces of `r2_algebra`, and
the isotypic projectors built from it.

Group data
----------
`f1mom_group.build_Gtilde()` returns 384 elements, each a pair of unit quaternions (g1, g2)
together with the SO(4) matrix R with R(x) = g1 x conj(g2), in the QUATERNION basis
(1, i, j, k).  The code's momentum 4-vectors (p1, p2, p3, p4) are in a different basis:
`r2_mom.pdict` gives

    p_{A Adot} = (1/sqrt2) [[p3 + i p4, p1 - i p2], [p1 + i p2, -p3 + i p4]]

whose determinant is -p^2/2, i.e. p_{A Adot} = i x (the standard quaternion embedding
q0 + q1 i + q2 j + q3 k -> [[q0+i q1, q2+i q3],[-q2+i q3, q0-i q1]]) of the quaternion with
coordinates (q0,q1,q2,q3) = (p4, -p3, -p2, -p1)/sqrt2.  So the change of basis is the signed
permutation L: p -> q, and a point-group element acts on the code's momentum vectors by
R_p = L^{-1} R L (still an integer signed permutation with det +1).

Fock-space action
-----------------
Every creation operator carries SO(4)_I indices, and a monomial transforms by replacing each
operator by its rotated linear combination (the "column" convention, U O_k U^{-1} = sum_l
M_{lk} O_l, which makes M(g) M(h) = M(gh)):

    alpha_{A Adot, m}  ->  sum_{B Bdot} Xb[B,A] Yb[Bdot,Adot] alpha_{B Bdot, m}
    d^{alpha A}_r      ->  sum_B  F[B,A] d^{alpha B}_r        (SU(2)_L index alpha untouched)

with, VERIFIED here for all 384 elements,

    W := S R_p S^{-1} = kron(Xb, Yb),    Xb = Q(g1),   Yb = conj(Q(g2)) up to the sign fixed
                                                            by kron(Xb, Yb) = W,
    F  = conj(Xb) .

The relative minus of Gate 6 between the fermion (upper A) and the boson (lower A) rotations
is exactly this complex conjugation: invariance of sum_A d^{alpha A} alpha_{A Adot} (which is
what makes G^alpha_Adot an SU(2)_1 singlet) forces F Xb^T = 1, i.e. F = (Xb^{-1})^T = conj(Xb)
for unitary Xb; infinitesimally that is `r2_rel.su2_index`'s relative minus.

Two central elements matter:
    zF := (g1, g2) = (-1, -1)   acts as (-1)^{#fermions}       = (-1)^F  (bosons untouched)
    z  := (g1, g2) = (-1, +1)   acts as (-1)^{#operators} and a -> -a
and z IS the orbifold Z_2 of the Sym^2 relative theory.  The physical (Z_2-projected) states
are therefore the +1 eigenspace of U(z), i.e. exactly the Gtilde irreps with chi_rho(z) =
+dim rho.  There are ten such irreps of dimension 1,1,2,3,3,3,3,3,3,6 with chi(zF) = +dim
("bosonic") and five with chi(zF) = -dim of dimension 2,2,4,6,6; the 4- and 8-dimensional
"bosonic" irreps have chi(z) = -dim and CANNOT appear on physical states.  This proves the
inference from the observed degeneracies.
"""
import os
import numpy as np

import f1mom_group as fg
import r2_algebra as al
import r2_mom as mo
from r2_algebra import UNTW, TW, PLUS, MINUS, VAC

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, '_cache', 'gtilde_cache.npz')      # written on first use, not part of the data

IDX = [(PLUS, PLUS), (PLUS, MINUS), (MINUS, PLUS), (MINUS, MINUS)]


def _Smat():
    S = np.zeros((4, 4), complex)
    for i in range(4):
        e = [0.0] * 4
        e[i] = 1.0
        pd = mo.pdict(e)
        for r, k in enumerate(IDX):
            S[r, i] = pd[k]
    return S


S = _Smat()
SINV = np.linalg.inv(S)
L = np.zeros((4, 4))
L[0, 3] = 1.0
L[1, 2] = -1.0
L[2, 1] = -1.0
L[3, 0] = -1.0
LINV = np.linalg.inv(L)


def quat_su2(q):
    """q0 + q1 i + q2 j + q3 k -> SU(2) (a homomorphism for f1mom_group.qmul)."""
    q0, q1, q2, q3 = q
    return np.array([[q0 + 1j * q1, q2 + 1j * q3],
                     [-q2 + 1j * q3, q0 - 1j * q1]])


def _factor_kron(W, Xb):
    """Yb with kron(Xb, Yb) = W (Xb invertible)."""
    Xi = np.linalg.inv(Xb)
    W4 = W.reshape(2, 2, 2, 2)          # A, Adot, B, Bdot
    Yb = np.zeros((2, 2), complex)
    for Ad in (0, 1):
        for Bd in (0, 1):
            Yb[Ad, Bd] = sum(Xi[0, A] * W4[A, Ad, 0, Bd] for A in (0, 1))
    return Yb


class GtildeRep:
    """Group data + index matrices."""

    def __init__(self):
        self.els = fg.build_Gtilde()
        self.n = len(self.els)
        self.Rp = [np.rint(LINV @ e['R'] @ L).astype(int) for e in self.els]
        self.Xb, self.Yb, self.F, self.W = [], [], [], []
        for i, e in enumerate(self.els):
            W = S @ self.Rp[i] @ SINV
            Xb = quat_su2(e['g1'])
            Yb = _factor_kron(W, Xb)
            assert np.allclose(np.kron(Xb, Yb), W, atol=1e-9), i
            self.Xb.append(Xb)
            self.Yb.append(Yb)
            self.F.append(Xb.conj())
            self.W.append(W)
        # momentum action.  With the "column" convention on creation operators the
        # momentum eigenvalue transforms by W^{-T} = conj(W) (see the module test), and
        # since S^* = S eta with eta = diag(1,-1,1,-1) that is the point-group element
        # eta R eta.  (eta is itself a det = +1 signed permutation, so this is an inner
        # automorphism -- orbits and stabilisers are simply relabelled.)
        self.eta = np.diag([1, -1, 1, -1])
        self.Rm = [self.eta @ R @ self.eta for R in self.Rp]
        self.idx = fg.build_index(self.els)
        self._load_or_build()

    # -- group combinatorics ------------------------------------------------
    def _load_or_build(self):
        if os.path.exists(CACHE):
            d = np.load(CACHE, allow_pickle=True)
            self.table = d['table']
            self.classes = [list(c) for c in d['classes']]
            self.chars = d['chars']
            self.class_sizes = d['class_sizes']
            self.id_idx = int(d['id_idx'])
            self.elem_class = d['elem_class']
            self.inv = d['inv']
        else:
            self.table = fg.multiplication_table(self.els)
            self.classes = fg.conjugacy_classes(self.els)
            self.chars, self.class_sizes, self.id_idx = fg.character_table(self.els, self.classes)
            ec = np.zeros(self.n, dtype=np.int32)
            for ci, c in enumerate(self.classes):
                for e in c:
                    ec[e] = ci
            self.elem_class = ec
            inv = np.zeros(self.n, dtype=np.int32)
            for i in range(self.n):
                inv[i] = fg.find_index(self.idx, fg.group_inverse(self.els[i]))
            self.inv = inv
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            np.savez_compressed(CACHE, table=self.table,
                                classes=np.array(self.classes, dtype=object),
                                chars=self.chars, class_sizes=self.class_sizes,
                                id_idx=self.id_idx, elem_class=self.elem_class, inv=inv)
        self.dims = np.rint(self.chars[:, self.id_idx].real).astype(int)
        # the two central involutions
        self.i_zF = self.find(-np.array([1.0, 0, 0, 0]), -np.array([1.0, 0, 0, 0]))
        self.i_z = self.find(-np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0, 0]))
        cz = self.elem_class[self.i_z]
        czF = self.elem_class[self.i_zF]
        self.phys = np.array([abs(self.chars[r, cz].real - self.dims[r]) < 1e-6
                              for r in range(len(self.dims))])
        self.bosonic = np.array([abs(self.chars[r, czF].real - self.dims[r]) < 1e-6
                                 for r in range(len(self.dims))])

    def find(self, g1, g2):
        return self.idx[(fg.quat_key(np.asarray(g1, float)), fg.quat_key(np.asarray(g2, float)))]

    def mult(self, i, j):
        return int(self.table[i, j])

    def generators(self):
        """A two-element generating set if one exists (it does for Gtilde)."""
        if getattr(self, '_gens', None) is not None:
            return self._gens
        rng = np.random.default_rng(0)
        for _ in range(4000):
            a, b = rng.integers(0, self.n, 2)
            if len(self._closure([int(a), int(b)])) == self.n:
                self._gens = [int(a), int(b)]
                return self._gens
        gens = []
        for i in range(self.n):
            gens.append(i)
            if len(self._closure(gens)) == self.n:
                break
        self._gens = gens
        return gens

    def id_idx_element(self):
        return self.find([1.0, 0, 0, 0], [1.0, 0, 0, 0])

    def _closure(self, gens):
        e0 = self.id_idx_element()
        seen = {e0}
        frontier = [e0]
        while frontier:
            nf = []
            for x in frontier:
                for g in gens:
                    y = self.mult(x, g)
                    if y not in seen:
                        seen.add(y)
                        nf.append(y)
            frontier = nf
        return seen

    def subgroup_gens(self, subset):
        """A small generating set for the subgroup `subset` (a list/set of indices)."""
        sub = set(int(x) for x in subset)
        e0 = self.id_idx_element()
        gens = []
        cur = {e0}
        for x in sorted(sub):
            if x in cur:
                continue
            gens.append(x)
            cur = self._closure(gens)
            if len(cur) == len(sub):
                break
        assert cur == sub, (len(cur), len(sub))
        return gens

    def word_tree(self, gens, subset=None):
        """BFS over `subset` (default the whole group) from the identity using right
        multiplication by generators; returns {element: (parent, gen)}."""
        e0 = self.id_idx_element()
        tree = {e0: None}
        frontier = [e0]
        while frontier:
            nf = []
            for x in frontier:
                for g in gens:
                    y = self.mult(x, g)
                    if y in tree:
                        continue
                    if subset is not None and y not in subset:
                        continue
                    tree[y] = (x, g)
                    nf.append(y)
            frontier = nf
        return tree


_G = None


def G():
    global _G
    if _G is None:
        _G = GtildeRep()
    return _G


# ---------------------------------------------------------------------------
# Fock-space action
# ---------------------------------------------------------------------------
def U_mono(sec, mono, Xb, Yb, F, tol=1e-14):
    """U(g) applied to one monomial of `sec`."""
    bos, fer = mono
    v = {VAC: 1.0 + 0j}
    for (k, cnt) in bos:
        mu, cp, A, Ad = k
        for _ in range(cnt):
            nv = {}
            for B in (PLUS, MINUS):
                xb = Xb[B, A]
                if abs(xb) < tol:
                    continue
                for Bd in (PLUS, MINUS):
                    c = xb * Yb[Bd, Ad]
                    if abs(c) < tol:
                        continue
                    for kk, cc in sec.alpha(v, mu, B, Bd, copy=cp).items():
                        nv[kk] = nv.get(kk, 0) + c * cc
            v = {kk: cc for kk, cc in nv.items() if abs(cc) > 1e-15}
    for k in reversed(fer):
        mu, cp, a, A = k
        nv = {}
        for B in (PLUS, MINUS):
            c = F[B, A]
            if abs(c) < tol:
                continue
            for kk, cc in sec.d(v, mu, a, B, copy=cp).items():
                nv[kk] = nv.get(kk, 0) + c * cc
        v = {kk: cc for kk, cc in nv.items() if abs(cc) > 1e-15}
    return v


def U_vec(sec, vec, gi, cache=None):
    """U(g_i) on a vector of `sec`."""
    g = G()
    Xb, Yb, F = g.Xb[gi], g.Yb[gi], g.F[gi]
    out = {}
    for m, c in vec.items():
        w = None
        if cache is not None:
            w = cache.get((gi, m))
        if w is None:
            w = U_mono(sec, m, Xb, Yb, F)
            if cache is not None:
                cache[(gi, m)] = w
        for k, cc in w.items():
            out[k] = out.get(k, 0) + c * cc
    return {k: c for k, c in out.items() if abs(c) > 1e-14}


def rep_matrix(sec, basis_from, basis_to, gi, cache=None):
    """Matrix of U(g_i) : span(basis_from) -> span(basis_to), both ORTHONORMAL in `sec`."""
    n, m = len(basis_to), len(basis_from)
    M = np.zeros((n, m), complex)
    for j, v in enumerate(basis_from):
        w = U_vec(sec, v, gi, cache=cache)
        for i, u in enumerate(basis_to):
            z = sec.ip(u, w)
            if abs(z) > 1e-13:
                M[i, j] = z
    return M


def gen_all_reps(sec, basis, gens, subset=None, cache=None):
    """Matrices of U(g) on span(basis) for every g in `subset` (default the whole group),
    computed on the generators only and multiplied out along a BFS tree."""
    g = G()
    tree = g.word_tree(gens, subset=subset)
    d = len(basis)
    reps = {g.id_idx_element(): np.eye(d, dtype=complex)}
    gm = {gi: rep_matrix(sec, basis, basis, gi, cache=cache) for gi in gens}
    order = sorted(tree, key=lambda x: 0 if tree[x] is None else 1)
    todo = [x for x in tree if tree[x] is not None]
    while todo:
        prog = False
        rest = []
        for x in todo:
            p, gi = tree[x]
            if p in reps:
                reps[x] = reps[p] @ gm[gi]
                prog = True
            else:
                rest.append(x)
        todo = rest
        if not prog:
            raise RuntimeError('BFS tree inconsistent')
    return reps
