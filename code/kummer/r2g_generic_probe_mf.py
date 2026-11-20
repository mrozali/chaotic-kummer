"""r2g_generic_probe_mf.py -- the generic-direction first-order probe with a MATRIX-FREE first-order
matrix, for E = th where the raw block M_c (nW^2 x du) no longer fits in memory (E = 7: 28224 x 56576,
25 GB dense).

M_c = kappa sum_{Ad,Bd} c[Ad,Bd] A_Ad (x) A_Bd acts on an untwisted vector x, sector (a,b) reshaped to
X_ab (na x nb), as  (M x)_{ij} = kappa sum_{Ad,Bd} c[Ad,Bd] (A_Ad[:,a] X_ab A_Bd[:,b]^T)_{ij}; we apply it
in two GEMM stages (T_Bd = concat_a sum_b X_ab A_Bd[:,b]^T over the chiral index, then A_Ad @ T_Bd),
chunked over columns.  Everything else (stabiliser, characters, randomised isotypic bases, extra Z_2's,
joint split, singular values, distinct values, gap ratios) is the method of r2g_generic_probe_L6.py.

Validation: --validate compares the matrix-free product with the dense M_c_raw on random vectors; at
E = 5, 6 the results must reproduce Generic_probe_<dir>_E{5,6}.json (rows, ranks, <r>).

Usage: python r2g_generic_probe_mf.py <direction> <th> [--validate] [--chunk=N]
Group action: sector-wise BLAS (class SectorAction), validated against the sparse matrices with --validate.
"""
import json
import os
import sys
import time

import numpy as np

import f2_common as f2
import r2c_space as sp
import r2g_generic_probe as gp
import r2g_generic_probe_L6 as pl6
from r2_algebra import PLUS, MINUS
import r2_d0 as d0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
R_GOE, R_GUE, R_POI = 0.5307, 0.5996, 0.3863


class MFree:
    """Matrix-free first-order block at level th along direction c (twisted <- untwisted)."""

    def __init__(self, S, th, c, chunk=800):
        self.A = {Ad: S.A[th][(th, Ad)] for Ad in (PLUS, MINUS)}      # nW x nV
        self.nW = S.nW[th]
        self.nV = self.A[PLUS].shape[1]
        self.off, self.du = S.sect_offsets(th)
        self.dt = self.nW ** 2
        self.sect = list(S.sect[th])
        self.voff, self.vdim = S.voff[th], S.vdim[th]
        self.c = c
        self.chunk = chunk
        self.terms = [(Ad, Bd) for Ad in (PLUS, MINUS) for Bd in (PLUS, MINUS) if abs(c[Ad, Bd]) > 1e-14]
        self.Bds = sorted(set(Bd for Ad, Bd in self.terms))
        self.shape = (self.dt, self.du)

    def matvec(self, X):
        X = np.asarray(X)
        one = X.ndim == 1
        if one:
            X = X[:, None]
        k = X.shape[1]
        out = np.empty((self.dt, k), complex)
        for k0 in range(0, k, self.chunk):
            k1 = min(k, k0 + self.chunk)
            kc = k1 - k0
            T = {Bd: np.zeros((self.nV, self.nW, kc), complex) for Bd in self.Bds}
            for (a, b) in self.sect:
                oa, ob = self.voff[a], self.voff[b]
                na, nb = self.vdim[a], self.vdim[b]
                o = self.off[(a, b)]
                Xs = X[o:o + na * nb, k0:k1].reshape(na, nb, kc)
                Xt = np.transpose(Xs, (0, 2, 1))                       # (na, kc, nb)
                for Bd in self.Bds:
                    Q = self.A[Bd][:, ob:ob + nb]                       # (nW, nb)
                    R = Xt @ Q.T                                        # (na, kc, nW)
                    T[Bd][oa:oa + na] += np.transpose(R, (0, 2, 1))
            Y = np.zeros((self.nW, self.nW * kc), complex)
            for Ad, Bd in self.terms:
                Y += self.c[Ad, Bd] * (self.A[Ad] @ T[Bd].reshape(self.nV, self.nW * kc))
            out[:, k0:k1] = d0.KAPPA * Y.reshape(self.dt, kc)
        return out[:, 0] if one else out

    def __matmul__(self, X):
        return self.matvec(X)


class SectorAction:
    """The untwisted point-group action at level th applied sector by sector, (Ma (x) Mb) X_s placed at the
    image sector (a', b'), with BLAS matmuls instead of a sparse matrix (the sparse products were the E = 8
    bottleneck).  right_only=True applies 1 (x) Mb (the right-only central elements Z_R, F_R)."""

    def __init__(self, S, th):
        self.S, self.th = S, th
        self.off, self.du = S.sect_offsets(th)
        self.sect = list(S.sect[th])
        self.vdim = S.vdim[th]
        self.moms = sorted(set(x for s in self.sect for x in s))
        self.cache = {}

    def maps(self, gi):
        if gi not in self.cache:
            self.cache[gi] = {a: self.S.Vmat(a, gi, self.th) for a in self.moms}
        return self.cache[gi]

    def apply(self, gi, X, right_only=False, out=None, weight=1.0):
        """U(g) X, or out += weight * U(g) X when `out` is given (no full-size temporaries)."""
        mp = self.maps(gi)
        one = X.ndim == 1
        if one:
            X = X[:, None]
        k = X.shape[1]
        Y = np.zeros_like(X) if out is None else out
        wgt = 1.0 if out is None else weight
        for (a, b) in self.sect:
            na, nb = self.vdim[a], self.vdim[b]
            o = self.off[(a, b)]
            Xs = X[o:o + na * nb].reshape(na, nb, k)
            bp, Mb = mp[b]
            T = np.matmul(Mb[None, :, :], Xs)                     # (na, nb', k)
            if right_only:
                ap = a
            else:
                ap, Ma = mp[a]
                T = np.tensordot(Ma, T, axes=(1, 0))               # (na', nb', k)
            o2 = self.off[(ap, bp)]
            if wgt != 1.0:
                T *= wgt
            Y[o2:o2 + T.shape[0] * T.shape[1]] += T.reshape(-1, k)
        if out is not None:
            return out
        return Y[:, 0] if one else Y

    def trace(self, gi, right_only=False):
        mp = self.maps(gi)
        tr = 0.0 + 0.0j
        for (a, b) in self.sect:
            bp, Mb = mp[b]
            if right_only:
                ap, ta = a, self.vdim[a]
            else:
                ap, Ma = mp[a]
                ta = np.trace(Ma)
            if (ap, bp) == (a, b):
                tr += ta * np.trace(Mb)
        return complex(tr)


def validate_action(S, th, act, elems, log):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((act.du, 2)) + 1j * rng.standard_normal((act.du, 2))
    worst = 0.0
    for gi in elems:
        U = pl6.U_untw_sparse(S, th, gi)
        worst = max(worst, float(np.max(np.abs(U @ x - act.apply(gi, x)))))
        worst = max(worst, abs(complex(U.diagonal().sum()) - act.trace(gi)))
    for gi in (S.g.i_z, S.g.i_zF):
        R = pl6.right_only_sparse(S, th, gi)
        worst = max(worst, float(np.max(np.abs(R @ x - act.apply(gi, x, right_only=True)))))
    log('  sector-wise group action validated against the sparse matrices at th=%d: max deviation %.2e' % (th, worst))
    assert worst < 1e-10, worst


def gram_orth(Y, m, rel_tol=1e-10):
    """Orthonormal basis of the (rank-m) range of the tall matrix Y (n x m, no oversampling) by CHOLESKY QR,
    three passes: G = Y^+ Y, L = chol(G), Q = Y L^{-+}; the first pass reduces the condition number to O(1)
    and the next two reach machine-precision orthonormality (CholQR3; valid while cond(Y) < 1e7, which a
    random Gaussian sketch of the projector satisfies by orders of magnitude).  Only GEMMs, a Cholesky and a
    triangular inverse, all BLAS-3 and multithreaded, against the single-threaded Hermitian eigensolve of the
    earlier version (4 min at m = 7000, over an hour at m = 19000).  Falls back to the eigh route with the
    columns it has if the Cholesky fails."""
    import scipy.linalg as sl
    Q = Y
    try:
        for it in range(3):
            G = Q.conj().T @ Q
            L = np.linalg.cholesky(G)
            Linv = sl.solve_triangular(L, np.eye(G.shape[0], dtype=complex), lower=True, check_finite=False)
            Q = Q @ Linv.conj().T
    except np.linalg.LinAlgError:
        G = Y.conj().T @ Y
        w, V = np.linalg.eigh(G)
        idx = np.argsort(w)[::-1][:m]
        Q = Y @ (V[:, idx] / np.sqrt(w[idx]))
        G2 = Q.conj().T @ Q
        w2, V2 = np.linalg.eigh(G2)
        Q = Q @ (V2 / np.sqrt(w2))
    assert Q.shape[1] == m, (Q.shape[1], m)
    err = float(np.max(np.abs(Q[:, :64].conj().T @ Q[:, :64] - np.eye(min(64, m)))))
    assert err < 1e-11, err
    return np.ascontiguousarray(Q)


def validate_mf(S, th, c, log, ntest=3):
    Md = gp.M_c_raw(S, th, c)
    mf = MFree(S, th, c)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((Md.shape[1], ntest)) + 1j * rng.standard_normal((Md.shape[1], ntest))
    err = float(np.max(np.abs(Md @ x - mf @ x))) / float(np.max(np.abs(Md @ x)))
    log('  matrix-free validation at th=%d: max rel error %.2e (dense %d x %d)' % (th, err, *Md.shape))
    assert err < 1e-12, err
    del Md


def run(direction, th, validate=False, chunk=800):
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    c = gp.cmat(direction)
    log('# Generic-direction probe (matrix-free) at E = %d: %s  (%s)' % (th, direction, time.strftime('%Y-%m-%d %H:%M')))
    log('PHYSICAL: %s' % gp.is_physical(c))
    t0 = time.time()
    S = sp.Space(th, mode='prim', verbose=False)
    g = S.g
    log('Space(prim, %d) built in %.1f s' % (th, time.time() - t0))
    if validate:
        validate_mf(S, min(th, 5), c, log)
        if th >= 6:
            validate_mf(S, 6, c, log)
    Hc, sgn, rule = gp.stabiliser(S, c, 3)
    log('stabiliser order %d (%d with G)' % (len(Hc), sum(1 for v in sgn.values() if v < 0)))
    chars, dims, elem_class, phys, ncls = gp.subgroup_characters(g, Hc)
    irr = [r for r in range(len(dims)) if phys[r]]
    log('irreps %d (dims %s), usable %d' % (len(dims), sorted(dims.tolist()), len(irr)))

    M = MFree(S, th, c, chunk=chunk)
    dt, du = M.shape
    nW = S.nW[th]
    log('matrix-free M %d x %d (nV %d, %d sectors, %d (Ad,Bd) terms)' % (dt, du, M.nV, len(M.sect), len(M.terms)))

    t1 = time.time()
    act = SectorAction(S, th)
    if validate:
        validate_action(S, th, act, list(Hc)[:4], log)
    Ut = {gi: S.Utw[th][gi] for gi in Hc}
    trU = {gi: act.trace(gi) for gi in Hc}
    log('sector-wise group action ready (%d sectors, %d momenta), traces in %.1f s' % (len(act.sect), len(act.moms), time.time() - t1))
    trT = {gi: sgn[gi] * complex(np.trace(Ut[gi])) ** 2 for gi in Hc}
    nH = len(Hc)
    rng = np.random.default_rng(11)

    def proj_apply(r, X, side):
        acc = np.zeros_like(X)
        for k, gi in enumerate(Hc):
            w = dims[r] / nH * np.conj(chars[r, elem_class[k]])
            if abs(w) < 1e-15:
                continue
            if side == 'u':
                act.apply(gi, X, out=acc, weight=w)
            else:
                acc += (w * sgn[gi]) * pl6.apply_tw(Ut[gi], X, nW)
        return acc

    def range_basis(r, side):
        n = du if side == 'u' else dt
        tr = sum(dims[r] / nH * np.conj(chars[r, elem_class[k]]) * (trU[gi] if side == 'u' else trT[gi])
                 for k, gi in enumerate(Hc))
        m = int(np.rint(tr.real))
        if m <= 0:
            return np.zeros((n, 0), complex)
        X = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))     # exactly m columns for CholQR
        Y = proj_apply(r, X, side)
        del X
        Q = gram_orth(Y, m)                                # CholQR3 (GEMMs + Cholesky), eigh fallback
        del Y
        assert Q.shape[1] == m, (r, side, Q.shape[1], m)
        chk = proj_apply(r, Q[:, :4], side)
        assert np.max(np.abs(chk - Q[:, :4])) < 1e-8, float(np.max(np.abs(chk - Q[:, :4])))
        return Q

    # extra Z_2's: Z_R and F_R (right-only central elements)
    syms = {}
    for nm, gi in (('Z_R', g.i_z), ('F_R', g.i_zF)):
        Xt_mat = S.Utw[th][gi]
        x = rng.standard_normal((du, 3)) + 1j * rng.standard_normal((du, 3))
        Mx = M @ x
        lhs = pl6.apply_tw_right(Xt_mat, Mx, nW)
        rhs = M @ act.apply(gi, x, right_only=True)
        cp = float(np.max(np.abs(lhs - rhs))) / max(float(np.max(np.abs(rhs))), 1e-30)
        cm = float(np.max(np.abs(lhs + rhs))) / max(float(np.max(np.abs(rhs))), 1e-30)
        if cp < 1e-9:
            syms[nm] = (gi, Xt_mat, +1)
        elif cm < 1e-9:
            syms[nm] = (gi, Xt_mat, -1)
        log('    %s: commutes %.1e anticommutes %.1e -> %s' % (nm, cp, cm, 'sym' if nm in syms else 'not a symmetry'))
    names = sorted(syms)

    # a reference scale for the rank threshold (the dense code used max|M|): largest singular value estimate
    x = rng.standard_normal((du, 2)) + 1j * rng.standard_normal((du, 2))
    gmax = float(np.max(np.abs(M @ x))) / np.sqrt(du)
    log('    rank threshold reference %.3e' % gmax)

    gaps, rows, dist_store = [], [], {}
    n_sub = n_dist = 0
    tot_rank = tot_dp = tot_dm = 0
    for r in irr:
        t2 = time.time()
        Qu = range_basis(r, 'u')
        Qt = range_basis(r, 't')
        if Qu.shape[1] == 0 or Qt.shape[1] == 0:
            continue
        d = int(dims[r])
        log('    irrep %d: bases %d (u) %d (t) in %.0f s' % (r, Qu.shape[1], Qt.shape[1], time.time() - t2))
        MQ = M @ Qu
        Mr = Qt.conj().T @ MQ
        del MQ
        labs_u = None
        if names:
            coef = rng.standard_normal(len(names))
            Yu = [Qu.conj().T @ act.apply(syms[nm][0], Qu, right_only=True) for nm in names]
            Yt = []
            for nm in names:
                Xt_mat, sg = syms[nm][1], syms[nm][2]
                T = pl6.apply_tw_right(Xt_mat, Qt, nW)
                Yt.append(sg * (Qt.conj().T @ T))
                del T

            def split(Ys):
                Z = sum(cf * (Y + Y.conj().T) / 2 for cf, Y in zip(coef, Ys))
                w, V = np.linalg.eigh(Z)
                labs = []
                for i in range(V.shape[1]):
                    v = V[:, i]
                    lab = []
                    for Y in Ys:
                        e = float(np.real(np.conj(v) @ (Y @ v)))
                        assert abs(abs(e) - 1) < 1e-6, e
                        lab.append(int(np.sign(e)))
                    labs.append(tuple(lab))
                return V, labs
            Vu, lu = split(Yu)
            Vt, lt = split(Yt)
            parts = {}
            for lab in set(lu) | set(lt):
                cu = [i for i, l in enumerate(lu) if l == lab]
                ct = [i for i, l in enumerate(lt) if l == lab]
                if cu and ct:
                    parts[lab] = (Vu[:, cu], Vt[:, ct])
        else:
            parts = {(): (np.eye(Qu.shape[1]), np.eye(Qt.shape[1]))}
        del Qu, Qt
        sv_all = np.linalg.svd(Mr, compute_uv=False)
        nz_all = sv_all[sv_all > 1e-9 * gmax]
        dp, dm = Mr.shape[1] // d, Mr.shape[0] // d
        rank = len(nz_all) / d
        tot_rank += rank
        tot_dp += dp
        tot_dm += dm
        for lab, (Vu_, Vt_) in sorted(parts.items()):
            Ms = Vt_.conj().T @ Mr @ Vu_
            s2 = np.linalg.svd(Ms, compute_uv=False)
            s2 = s2[s2 > 1e-9 * gmax]
            dist = np.unique(np.round(s2, 9))
            n_sub += 1
            n_dist += len(dist)
            rows.append(dict(rho=int(r), lab=list(lab), nu=int(Vu_.shape[1]), nt=int(Vt_.shape[1]),
                             rank=int(len(s2)), distinct=int(len(dist))))
            dist_store['rho%d_lab%s' % (r, ''.join('p' if l > 0 else 'm' for l in lab))] = dist
            if len(dist) >= 20:
                gaps.append(f2.gap_ratios(dist))
        log('    irrep %d (dim %d): mult dims %d/%d, rank %g, %d sub-blocks  [%.1f s]'
            % (r, d, dp, dm, rank, len(parts), time.time() - t2))
        del Mr
    gaps = np.concatenate(gaps) if gaps else np.zeros(0)
    rmean = float(np.mean(gaps)) if len(gaps) else float('nan')
    rsem = float(np.std(gaps) / np.sqrt(len(gaps))) if len(gaps) else float('nan')
    log('E = %d %s: multiplicity space d_+ %d, d_- %d, rank %g, zero modes %g (fraction %.2f)'
        % (th, direction, tot_dp, tot_dm, tot_rank, tot_dp + tot_dm - 2 * tot_rank,
           (tot_dp + tot_dm - 2 * tot_rank) / max(tot_dp + tot_dm, 1)))
    log('    resolved sub-blocks %d (irreps x %s), distinct values %d, gap ratios %d : <r> = %.4f +- %.4f'
        '   [GOE %.3f, GUE %.3f, Poisson %.3f]' % (n_sub, names, n_dist, len(gaps), rmean, rsem, R_GOE, R_GUE, R_POI))
    for x in rows:
        log('    | %d | %s | %d | %d | %d | %d |' % (x['rho'], x['lab'], x['nu'], x['nt'], x['rank'], x['distinct']))
    res = dict(direction=direction, th=th, H_order=len(Hc), d_plus=tot_dp, d_minus=tot_dm, rank=tot_rank,
               n_sub=n_sub, n_distinct=n_dist, n_gaps=int(len(gaps)), r_mean=rmean, r_sem=rsem,
               extra=names, rows=rows, seconds=time.time() - t0, method='matrix-free')
    with open(os.path.join(OUT, 'Generic_probe_%s_E%d_mf.json' % (direction, th)), 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, 'Generic_probe_%s_E%d_mf_values.npz' % (direction, th)), **dist_store)
    with open(os.path.join(OUT, 'Generic_probe.md'), 'a') as fh:
        fh.write('\n'.join(lines) + '\n\n')
    log('[total %.1f s]' % (time.time() - t0))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    chunk = 800
    for a in sys.argv[1:]:
        if a.startswith('--chunk'):
            chunk = int(a.split('=')[1]) if '=' in a else 800
    run(args[0], int(args[1]), validate='--validate' in sys.argv, chunk=chunk)
