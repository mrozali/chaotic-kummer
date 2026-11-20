"""Route 2 Phase C: the spin-0 state space at cutoff Lambda, the Gtilde-resolved multiplicity
bases, and the full (level-changing) D_0 block.

Layout
------
Chiral spaces per level th = 2h:
    Wtw(th)          twisted relative massive primaries,  dim nW[th]
    Vun(th) = (+)_a V_a(th)  untwisted, ordered by the momentum vector a (|a|^2 <= th,
                     th - |a|^2 even), dim nV[th], offsets voff[th][a]

Spin-0 states:
    untwisted, level th:  (+)_{(a,b): a = b mod 2}  V_a (x) V_b   ("raw"; the orbifold Z_2 is
                          the CENTRAL Gtilde element z = (-1, +1), so the physical projection
                          is part of the isotypic projection)
    twisted,   level th:  Wtw(th) (x) Wtw(th)

D_0 (spin conserving, energy changing):
    <t(th_t) | D_0 | u(th_u)> = kappa [ A_-(th_t<-th_u) (x) A_+(th_t<-th_u)
                                      - A_+(th_t<-th_u) (x) A_-(th_t<-th_u) ]
with an extra sqrt2 for a != 0 sectors from the Z_2 pairing IF one works with the halved
sector list; here the raw (unhalved) sector list is used and the Z_2 lives in the group, so
NO extra factor appears.

Multiplicity bases (the "weight e_1 slice" of the rho-isotypic component, so that the
Hamiltonian restricted to it is exactly the multiplicity-space Hamiltonian H_rho):
    twisted side  : Wtw (x) Wtw carries the DIAGONAL Gtilde action, so irrep-adapt Wtw once
                    and use the Clebsch-Gordan slices of sigma (x) tau.
    untwisted side: the sectors form Gtilde-orbits, so each orbit block is an INDUCED
                    representation Ind_K^G(V_{a_0} (x) V_{b_0}) and Frobenius reciprocity
                    gives the multiplicity vectors as
                        Phi_phi(e_1)_s = iota_s( phi( D^rho(gamma_s^{-1}) e_1 ) )
                    with phi running over a basis of Hom_K(Res_K rho, V_{a_0} (x) V_{b_0}).
"""
import itertools
import time
import numpy as np

import r2_algebra as al
import r2_d0 as d0
import r2c_group as gr
import r2c_chiral as ch
import r2c_irreps as ir
from r2_algebra import UNTW, TW, PLUS, MINUS


def lat_vectors(th):
    """Momentum vectors a with |a|^2 <= th and th - |a|^2 even, sorted."""
    R = int(np.floor(np.sqrt(th))) + 1
    out = []
    for v in itertools.product(range(-R, R + 1), repeat=4):
        n2 = sum(x * x for x in v)
        if n2 <= th and (th - n2) % 2 == 0:
            out.append(v)
    return sorted(out, key=lambda v: (sum(x * x for x in v), v))


class Space:
    def __init__(self, th_max, mode='prim', verbose=True):
        self.th_max = th_max
        self.mode = mode
        self.g = gr.G()
        self.irr = ir.I()
        self.orb = ch.MomOrbits(th_max)
        self.levels = list(range(1, th_max + 1))
        self.verbose = verbose
        t0 = time.time()
        self._chiral_bases()
        self._log('chiral bases + twisted group matrices', t0)
        t0 = time.time()
        self._little_groups()
        self._log('little-group representations', t0)
        t0 = time.time()
        self._chiral_maps()
        self._log('chiral maps T_Adot (all level pairs)', t0)
        t0 = time.time()
        self._sectors()
        self._log('sector pairs and orbits', t0)

    def _log(self, what, t0):
        if self.verbose:
            print('  [%6.1fs] %s' % (time.time() - t0, what), flush=True)

    # ------------------------------------------------------------------ bases
    def _chiral_bases(self):
        g = self.g
        gens = g.generators()
        self.Wb, self.nW, self.Utw = {}, {}, {}
        cache = {}
        for th in self.levels:
            b = ch.tw_basis(th, self.mode)
            self.Wb[th] = b
            self.nW[th] = len(b)
            if b:
                self.Utw[th] = gr.gen_all_reps(TW, b, gens, cache=cache)
        self.vecs, self.vdim, self.voff, self.nV = {}, {}, {}, {}
        for th in self.levels:
            vs = lat_vectors(th)
            dims, off, tot = {}, {}, 0
            for a in vs:
                n2 = sum(x * x for x in a)
                if n2 == 0:
                    bb, e, o = ch.untw_basis_osc(th, self.mode)
                    n = len(bb)
                else:
                    a0 = self.orb.rep[a]
                    n = len(ch.untw_basis_mom(a0, th - n2, self.mode))
                if n == 0:
                    continue
                dims[a] = n
                off[a] = tot
                tot += n
            self.vecs[th] = [a for a in vs if a in dims]
            self.vdim[th] = dims
            self.voff[th] = off
            self.nV[th] = tot

    # --------------------------------------------------------- little groups
    def _little_groups(self):
        g = self.g
        self.rho_a = {}
        cache = {}
        for a0 in self.orb.reps():
            n2 = sum(x * x for x in a0)
            stab = self.orb.stab[a0]
            sgens = g.subgroup_gens(stab)
            for tho in range(0, self.th_max - n2 + 1, 2):
                th = tho + n2
                if th < 1 or th > self.th_max:
                    continue
                if n2 == 0:
                    bb, e, o = ch.untw_basis_osc(th, self.mode)
                else:
                    bb = ch.untw_basis_mom(a0, tho, self.mode)
                if not bb:
                    continue
                self.rho_a[(a0, th)] = gr.gen_all_reps(UNTW, bb, sgens,
                                                       subset=set(stab), cache=cache)

    # ----------------------------------------------------------- chiral maps
    def _chiral_maps(self):
        g = self.g
        # per orbit representative
        A0 = {}
        for a0 in self.orb.reps():
            n2 = sum(x * x for x in a0)
            p4 = None if n2 == 0 else [x / np.sqrt(2.0) for x in a0]
            for tho in range(0, self.th_max - n2 + 1, 2):
                th = tho + n2
                if th < 1:
                    continue
                if n2 == 0:
                    bb, e, o = ch.untw_basis_osc(th, self.mode)
                else:
                    bb = ch.untw_basis_mom(a0, tho, self.mode)
                if not bb:
                    continue
                res = {Ad: {} for Ad in (PLUS, MINUS)}
                for j, v in enumerate(bb):
                    for Ad in (PLUS, MINUS):
                        gr_ = ch.T_Adot_all(v, Ad, th, self.th_max, p4=p4)
                        for tht, w in gr_.items():
                            if tht not in self.nW or self.nW[tht] == 0:
                                continue
                            M = res[Ad].setdefault(tht, np.zeros((self.nW[tht], len(bb)), complex))
                            for i, q in enumerate(self.Wb[tht]):
                                z = TW.ip(q, w)
                                if abs(z) > 1e-13:
                                    M[i, j] = z
                A0[(a0, th)] = res
        # assemble the full A[th_u][th_t][Adot]
        self.A = {}
        for th in self.levels:
            per = {}
            for a in self.vecs[th]:
                a0 = self.orb.rep[a]
                ga = self.orb.coset[a]
                Yinv_T = np.linalg.inv(g.Yb[ga]).T
                res = A0.get((a0, th), {PLUS: {}, MINUS: {}})
                for tht in set(res[PLUS]) | set(res[MINUS]):
                    U = self.Utw[tht][ga]
                    for Ad in (PLUS, MINUS):
                        blk = np.zeros((self.nW[tht], self.vdim[th][a]), complex)
                        for Cd in (PLUS, MINUS):
                            if Cd in res and tht in res[Cd]:
                                blk = blk + Yinv_T[Ad, Cd] * (U @ res[Cd][tht])
                        M = per.setdefault((tht, Ad),
                                           np.zeros((self.nW[tht], self.nV[th]), complex))
                        o = self.voff[th][a]
                        M[:, o:o + blk.shape[1]] = blk
            self.A[th] = per

    # --------------------------------------------------------------- sectors
    def _sectors(self):
        g = self.g
        self.sect = {}
        self.orbits = {}
        for th in self.levels:
            vs = self.vecs[th]
            vset = set(vs)
            pairs = [(a, b) for a in vs for b in vs
                     if not any((x - y) % 2 for x, y in zip(a, b))]
            self.sect[th] = pairs
            pset = set(pairs)
            seen = set()
            orbs = []
            for s in pairs:
                if s in seen:
                    continue
                a0, b0 = s
                members, coset, K = [], {}, []
                for gi in range(g.n):
                    Ra = tuple(int(x) for x in (g.Rm[gi] @ np.array(a0)))
                    Rb = tuple(int(x) for x in (g.Rm[gi] @ np.array(b0)))
                    t = (Ra, Rb)
                    if t == s:
                        K.append(gi)
                    if t not in coset:
                        coset[t] = gi
                        members.append(t)
                        seen.add(t)
                assert all(m in pset for m in members)
                orbs.append(dict(rep=s, members=members, coset=coset, K=K))
            self.orbits[th] = orbs

    # -------------------------------------------------- transport of V_a bases
    def Vmat(self, a, gi, th):
        """(a', matrix of U(g_i) : V_a(th) -> V_{a'}(th)) in the induced bases
        basis(V_a) = U(g_a) basis(V_{rep a})."""
        g = self.g
        ap = tuple(int(x) for x in (g.Rm[gi] @ np.array(a)))
        A = self.orb.rep[a]
        ga, gap = self.orb.coset[a], self.orb.coset[ap]
        k = g.mult(g.mult(int(g.inv[gap]), gi), ga)
        return ap, self.rho_a[(A, th)][k]

    # ------------------------------------------- raw spin-0 group matrices
    def sect_offsets(self, th):
        off, tot = {}, 0
        for s in self.sect[th]:
            off[s] = tot
            tot += self.vdim[th][s[0]] * self.vdim[th][s[1]]
        return off, tot

    def U_untw_raw(self, th, gi):
        """Dense U(g_i) on the RAW untwisted spin-0 space at level th."""
        off, tot = self.sect_offsets(th)
        M = np.zeros((tot, tot), complex)
        for s in self.sect[th]:
            a, b = s
            ap, Ma = self.Vmat(a, gi, th)
            bp, Mb = self.Vmat(b, gi, th)
            sp_ = (ap, bp)
            blk = np.kron(Ma, Mb)
            i0, j0 = off[sp_], off[s]
            M[i0:i0 + blk.shape[0], j0:j0 + blk.shape[1]] = blk
        return M

    def M_raw(self, th_t, th_u):
        """kappa (A_- (x) A_+ - A_+ (x) A_-) : raw untwisted level th_u -> twisted th_t."""
        Am = self.A[th_u].get((th_t, MINUS))
        Ap = self.A[th_u].get((th_t, PLUS))
        nW = self.nW[th_t]
        off, tot = self.sect_offsets(th_u)
        M = np.zeros((nW * nW, tot), complex)
        if Am is None or Ap is None:
            return M
        for s in self.sect[th_u]:
            a, b = s
            oa, ob = self.voff[th_u][a], self.voff[th_u][b]
            na, nb = self.vdim[th_u][a], self.vdim[th_u][b]
            blk = d0.KAPPA * (np.kron(Am[:, oa:oa + na], Ap[:, ob:ob + nb])
                              - np.kron(Ap[:, oa:oa + na], Am[:, ob:ob + nb]))
            M[:, off[s]:off[s] + blk.shape[1]] = blk
        return M

    # ------------------------------------------------------------ dimensions
    def dims(self):
        du = {th: sum(self.vdim[th][a] * self.vdim[th][b] for (a, b) in self.sect[th])
              for th in self.levels}
        dt = {th: self.nW[th] ** 2 for th in self.levels}
        return du, dt
