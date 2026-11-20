"""Route 2 Phase C: assemble the Gtilde-resolved truncated Hamiltonian H(lambda) = H_0 +
lambda D_0 in the multiplicity space of one irrep."""
import time
import numpy as np

import r2_d0 as d0
import r2c_group as gr
import r2c_irreps as ir
import r2c_proj as pj
from r2_algebra import PLUS, MINUS


class IrrepSector:
    """Multiplicity-space Hamiltonian of one Gtilde irrep at cutoff th_max."""

    def __init__(self, S, rho, chunk=64, verbose=True):
        self.S = S
        self.rho = rho
        self.g = gr.G()
        self.I = ir.I()
        self.d = int(self.I.dims[rho])
        self.verbose = verbose
        t0 = time.time()
        self._untw_basis()
        self._tw_basis()
        self.t_basis = time.time() - t0
        t0 = time.time()
        self._d0_block(chunk)
        self.t_block = time.time() - t0

    # ---------------------------------------------------------------- bases
    def _untw_basis(self):
        S, g, D = self.S, self.g, self.I.D[self.rho]
        self.Bu = {}
        self.mu = {}
        for th in S.levels:
            per = {}
            m_tot = 0
            for orb in S.orbits[th]:
                a0, b0 = orb['rep']
                K = orb['K']
                na, nb = S.vdim[th][a0], S.vdim[th][b0]
                pairs = [(S.Vmat(a0, k, th)[1], S.Vmat(b0, k, th)[1], D[k]) for k in K]
                phis = pj.hom_basis(pairs, D, na, nb)
                m = phis.shape[0]
                if m == 0:
                    continue
                loc = {}
                for s in orb['members']:
                    gs = orb['coset'][s]
                    x = D[int(g.inv[gs])][:, 0]
                    y = np.einsum('mijc,c->mij', phis, x)
                    ia = S.Vmat(a0, gs, th)[1]
                    ib = S.Vmat(b0, gs, th)[1]
                    loc[s] = np.einsum('pi,mij,qj->pqm', ia, y, ib)
                # orthonormalise within this orbit block
                X = np.concatenate([loc[s].reshape(-1, m) for s in orb['members']], axis=0)
                U, sv, Vt = np.linalg.svd(X, full_matrices=False)
                r = int(np.sum(sv > 1e-9 * sv[0]))
                assert r == m, (r, m)
                T = Vt.conj().T @ np.diag(1.0 / sv)      # X T has orthonormal columns
                for s in orb['members']:
                    blk = loc[s].reshape(-1, m) @ T
                    per[s] = (m_tot, blk.reshape(loc[s].shape))
                m_tot += m
            if m_tot:
                self.Bu[th] = per
                self.mu[th] = m_tot

    def _tw_basis(self):
        S = self.S
        self.Bt = {}
        self.mt = {}
        for th in S.levels:
            if S.nW.get(th, 0) == 0:
                continue
            cacheobj = S.__dict__.setdefault('_adapt', {})
            cache = cacheobj.setdefault(th, {})
            vs = pj.tensor_mult_basis(S.Utw[th], self.rho, adapt_cache=cache)
            if not vs:
                continue
            A = np.stack(vs, axis=-1)
            self.Bt[th] = A
            self.mt[th] = A.shape[-1]

    # ------------------------------------------------------------ D_0 block
    def _d0_block(self, chunk):
        S = self.S
        ulev = sorted(self.mu)
        tlev = sorted(self.mt)
        self.ulev, self.tlev = ulev, tlev
        uoff, o = {}, 0
        for th in ulev:
            uoff[th] = o
            o += self.mu[th]
        self.nu = o
        toff, o = {}, 0
        for th in tlev:
            toff[th] = o
            o += self.mt[th]
        self.nt = o
        self.uoff, self.toff = uoff, toff
        M = np.zeros((self.nt, self.nu), complex)
        for thu in ulev:
            nV = S.nV[thu]
            m = self.mu[thu]
            for start in range(0, m, chunk):
                sl = slice(start, min(start + chunk, m))
                k = sl.stop - sl.start
                C = np.zeros((nV, nV, k), complex)
                for (a, b), (c0, blk) in self.Bu[thu].items():
                    lo = max(sl.start, c0)
                    hi = min(sl.stop, c0 + blk.shape[2])
                    if hi <= lo:
                        continue
                    oa, ob = S.voff[thu][a], S.voff[thu][b]
                    C[oa:oa + blk.shape[0], ob:ob + blk.shape[1],
                      lo - sl.start:hi - sl.start] = blk[:, :, lo - c0:hi - c0]
                for tht in tlev:
                    Am = S.A[thu].get((tht, MINUS))
                    Ap = S.A[thu].get((tht, PLUS))
                    if Am is None or Ap is None:
                        continue
                    t1 = np.tensordot(Am, C, axes=([1], [0]))          # (nW, nV, k)
                    Y = np.tensordot(Ap, t1, axes=([1], [1]))          # (nW, nW, k)
                    Y = np.transpose(Y, (1, 0, 2))
                    t2 = np.tensordot(Ap, C, axes=([1], [0]))
                    Y2 = np.tensordot(Am, t2, axes=([1], [1]))
                    Y -= np.transpose(Y2, (1, 0, 2))
                    Y *= d0.KAPPA
                    Bt = self.Bt[tht]
                    blk = np.tensordot(np.conj(Bt), Y, axes=([0, 1], [0, 1]))  # (mt, k)
                    M[toff[tht]:toff[tht] + self.mt[tht], uoff[thu] + sl.start:
                      uoff[thu] + sl.stop] = blk
        self.M = M
        self.Eu = np.concatenate([np.full(self.mu[th], float(th)) for th in ulev]) \
            if ulev else np.zeros(0)
        self.Et = np.concatenate([np.full(self.mt[th], float(th)) for th in tlev]) \
            if tlev else np.zeros(0)
        self.E0 = np.concatenate([self.Eu, self.Et])
        self.n = self.nu + self.nt

    # ------------------------------------------------- L <-> R exchange parity
    def parity_operator(self):
        """Pi = exchange of the left and right chiral factors, in the multiplicity bases.
        Pi is level graded and grading preserving, commutes with Gtilde, and satisfies
        Pi_t M Pi_u = -M (the deformation is a WEDGE A_- (x) A_+ - A_+ (x) A_-).  Hence
        Lambda = Pi (x) diag(+1 untwisted, -1 twisted) commutes with H(lambda) for every
        lambda: worldsheet parity times the quantum (twisted-sector) Z_2 of the orbifold.
        Returns (Pu, Pt)."""
        Pu = np.zeros((self.nu, self.nu), complex)
        for th in sorted(self.mu):
            off = self.uoff[th]
            B = self.Bu[th]
            for (a, b), (c0, blk) in B.items():
                c1, blk2 = B[(b, a)]
                Pu[off + c1:off + c1 + blk2.shape[2], off + c0:off + c0 + blk.shape[2]] += \
                    np.einsum('jib,ija->ba', np.conj(blk2), blk)
        Pt = np.zeros((self.nt, self.nt), complex)
        for th in sorted(self.mt):
            off = self.toff[th]
            B = self.Bt[th]
            m = B.shape[2]
            Pt[off:off + m, off:off + m] = np.einsum('lkb,kla->ba', np.conj(B), B)
        return Pu, Pt

    def parity_split(self, tol=1e-6):
        """Split the multiplicity space into the two eigenspaces of Lambda.  Returns
        {+1: (E0, M), -1: (E0, M)} with H_0 still diagonal (the split is done inside each
        level block)."""
        Pu, Pt = self.parity_operator()
        out = {}
        Qs = {}
        for tag, P, off, mm, lev, gsign in (('u', Pu, self.uoff, self.mu, self.ulev, +1),
                                            ('t', Pt, self.toff, self.mt, self.tlev, -1)):
            cols = {+1: [], -1: []}
            for th in lev:
                o, m = off[th], mm[th]
                blk = P[o:o + m, o:o + m]
                w, V = np.linalg.eigh((blk + blk.conj().T) / 2)
                assert np.max(np.abs(np.abs(w) - 1)) < tol, np.max(np.abs(np.abs(w) - 1))
                for sgn in (+1, -1):
                    # Lambda = gsign * Pi on this grading, so Lambda = sgn means
                    # Pi = sgn * gsign
                    ps = sgn * gsign
                    sel = V[:, (w > 0) if ps > 0 else (w < 0)]
                    if sel.shape[1]:
                        full = np.zeros((P.shape[0], sel.shape[1]), complex)
                        full[o:o + m] = sel
                        cols[sgn].append((th, full))
            Qs[tag] = cols
        for sgn in (+1, -1):
            qu = Qs['u'][sgn]
            qt = Qs['t'][sgn]
            Qu = np.concatenate([x[1] for x in qu], axis=1) if qu else np.zeros((self.nu, 0))
            Qt = np.concatenate([x[1] for x in qt], axis=1) if qt else np.zeros((self.nt, 0))
            Eu = np.concatenate([np.full(x[1].shape[1], float(x[0])) for x in qu]) if qu \
                else np.zeros(0)
            Et = np.concatenate([np.full(x[1].shape[1], float(x[0])) for x in qt]) if qt \
                else np.zeros(0)
            Mp = np.conj(Qt.T) @ self.M @ Qu
            out[sgn] = SubSector(np.concatenate([Eu, Et]), Mp, len(Eu))
        return out

    # ------------------------------------------------------------ Hamiltonian
    def H(self, lam):
        n = self.n
        H = np.zeros((n, n), complex)
        H[np.arange(n), np.arange(n)] = self.E0
        if lam:
            H[self.nu:, :self.nu] = lam * self.M
            H[:self.nu, self.nu:] = lam * self.M.conj().T
        return H

    def eig(self, lam):
        return np.linalg.eigvalsh(self.H(lam))


class SubSector:
    """A parity-resolved multiplicity block: H(lambda) = diag(E0) + lambda [[0,M^dag],[M,0]]."""

    def __init__(self, E0, M, nu):
        self.E0 = E0
        self.M = M
        self.nu = nu
        self.nt = len(E0) - nu
        self.n = len(E0)

    def H(self, lam):
        n = self.n
        H = np.zeros((n, n), complex)
        H[np.arange(n), np.arange(n)] = self.E0
        if lam:
            H[self.nu:, :self.nu] = lam * self.M
            H[:self.nu, self.nu:] = lam * self.M.conj().T
        return H

    def eig(self, lam):
        return np.linalg.eigvalsh(self.H(lam))
