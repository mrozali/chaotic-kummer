"""r2g_generic_sweep.py -- the truncated Hamiltonian H(lambda) = H_0 + lambda D_0 along an ARBITRARY
direction c of the Z_2^4-symmetric blow-up moduli (version A = massive-primary bases, cutoff
Lambda = th_max/2), resolved by the stabiliser H_c of the direction in Gtilde and the surviving
Z_2's (Pi, Z_R, F_R, each possibly times the grading G), with the paper's statistics
(r2c_stats.analyse: unfolded gap ratio, number variance, calibrated ramp coefficient) in the paper's
window (r2c_gate11.window).

Construction (all in the RAW spin-0 spaces of r2c_space.Space, z-odd states included and projected
out by the isotypic projectors):
  D_0 (twisted th_t <- untwisted th_u) = kappa sum_{Ad,Bd} c[Ad,Bd] A_Ad(th_t<-th_u) (x) A_Bd(th_t<-th_u)
  H_c from the level-diagonal block at th=3, characters of H_c, isotypic bases LEVEL BY LEVEL by a
  randomised range finder (so every basis vector has a definite bare energy), joint split by the
  extra Z_2's inside each level, sub-blocks (irrep, labels) with E0 and M_sub saved to npz.
Validation: direction 'singlet' at th_max = 4, 5 against r2c_gate11.run('prim', th_max) on the saved
PhaseC_spectra_prim_L{th_max}.npz (same window, same estimator).

Usage: python r2g_generic_sweep.py <direction> <th_max> [lambdas comma-separated]
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sps

import r2c_gate11 as g11
import r2c_space as sp
import r2c_stats as st
import r2g_generic_probe as gp
from r2_algebra import PLUS, MINUS
import r2_d0 as d0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
LS = (1, 2, 3, 5, 8)
DEFAULT_LAMS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5)


# ------------------------------------------------------------------ raw blocks, all level pairs
def M_pair(S, th_t, th_u, c):
    A = {Ad: S.A[th_u].get((th_t, Ad)) for Ad in (PLUS, MINUS)}
    nW = S.nW.get(th_t, 0)
    off, tot = S.sect_offsets(th_u)
    M = np.zeros((nW * nW, tot), complex)
    if nW == 0 or A[PLUS] is None or A[MINUS] is None:
        return M
    for s in S.sect[th_u]:
        a, b = s
        oa, ob = S.voff[th_u][a], S.voff[th_u][b]
        na, nb = S.vdim[th_u][a], S.vdim[th_u][b]
        blk = np.zeros((nW * nW, na * nb), complex)
        for Ad in (PLUS, MINUS):
            for Bd in (PLUS, MINUS):
                if abs(c[Ad, Bd]) < 1e-14:
                    continue
                blk += c[Ad, Bd] * np.kron(A[Ad][:, oa:oa + na], A[Bd][:, ob:ob + nb])
        M[:, off[s]:off[s] + blk.shape[1]] = d0.KAPPA * blk
    return M


def U_untw_sparse(S, th, gi):
    off, tot = S.sect_offsets(th)
    rows, cols, vals = [], [], []
    for s in S.sect[th]:
        a, b = s
        ap, Ma = S.Vmat(a, gi, th)
        bp, Mb = S.Vmat(b, gi, th)
        blk = np.kron(Ma, Mb)
        i0, j0 = off[(ap, bp)], off[s]
        nz = np.nonzero(np.abs(blk) > 1e-14)
        rows.append(i0 + nz[0])
        cols.append(j0 + nz[1])
        vals.append(blk[nz])
    return sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(tot, tot), dtype=complex)


def right_only_sparse(S, th, gi):
    off, tot = S.sect_offsets(th)
    rows, cols, vals = [], [], []
    for s in S.sect[th]:
        a, b = s
        na = S.vdim[th][a]
        bp, Mb = S.Vmat(b, gi, th)
        blk = np.kron(np.eye(na), Mb)
        i0, j0 = off[(a, bp)], off[s]
        nz = np.nonzero(np.abs(blk) > 1e-14)
        rows.append(i0 + nz[0])
        cols.append(j0 + nz[1])
        vals.append(blk[nz])
    return sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(tot, tot), dtype=complex)


def pi_untw_sparse(S, th):
    off, tot = S.sect_offsets(th)
    rows, cols = [], []
    for (a, b) in S.sect[th]:
        na, nb = S.vdim[th][a], S.vdim[th][b]
        i = np.repeat(np.arange(na), nb)
        j = np.tile(np.arange(nb), na)
        rows.append(off[(b, a)] + j * na + i)
        cols.append(off[(a, b)] + i * nb + j)
    r, cc = np.concatenate(rows), np.concatenate(cols)
    return sps.csr_matrix((np.ones(len(r), complex), (r, cc)), shape=(tot, tot))


def tw_apply(kind, U, X, nW):
    """Apply (U (x) U) [kind 'both'], (1 (x) U) [kind 'right'] or the transpose [kind 'pi'] to X."""
    k = X.shape[1]
    Y = X.reshape(nW, nW, k)
    if kind == 'both':
        Y = np.einsum('ia,abk->ibk', U, Y)
        Y = np.einsum('jb,ibk->ijk', U, Y)
    elif kind == 'right':
        Y = np.einsum('jb,ibk->ijk', U, Y)
    elif kind == 'pi':
        Y = np.transpose(Y, (1, 0, 2))
    return Y.reshape(nW * nW, k)


# ------------------------------------------------------------------ main
def run(direction, th_max, lams=DEFAULT_LAMS):
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    c = gp.cmat(direction)
    log('# Generic-direction sweep: %s, th_max = %d (Lambda = %g)  [%s]' % (direction, th_max, th_max / 2,
                                                                          time.strftime('%Y-%m-%d %H:%M')))
    log('c = %s   physical: %s' % (np.array2string(c, precision=4).replace('\n', ' '), gp.is_physical(c)))
    T0 = time.time()
    S = sp.Space(th_max, mode='prim', verbose=False)
    g = S.g
    levels = [th for th in S.levels if S.nW.get(th, 0) > 0 and S.sect_offsets(th)[1] > 0]
    du = {th: S.sect_offsets(th)[1] for th in levels}
    dt = {th: S.nW[th] ** 2 for th in levels}
    uoff, o = {}, 0
    for th in levels:
        uoff[th] = o
        o += du[th]
    DU = o
    toff, o = {}, 0
    for th in levels:
        toff[th] = o
        o += dt[th]
    DT = o
    log('Space built in %.1f s; raw untwisted %d, twisted %d, levels %s' % (time.time() - T0, DU, DT, levels))

    # full raw D_0 (twisted <- untwisted), all level pairs
    t0 = time.time()
    M = np.zeros((DT, DU), complex)
    for th_u in levels:
        for th_t in levels:
            blk = M_pair(S, th_t, th_u, c)
            if blk.size and np.any(blk):
                M[toff[th_t]:toff[th_t] + dt[th_t], uoff[th_u]:uoff[th_u] + du[th_u]] = blk
    gmax = float(np.max(np.abs(M)))
    log('raw D_0 %d x %d built in %.1f s (%.2f GB)' % (DT, DU, time.time() - t0, M.nbytes / 1e9))

    # stabiliser and characters
    Hc, sgn, rule = gp.stabiliser(S, c, min(3, th_max))
    chars, dims, elem_class, phys, ncls = gp.subgroup_characters(g, Hc)
    irr = [r for r in range(len(dims)) if phys[r]]
    nH = len(Hc)
    log('stabiliser order %d (%d with G), %d irreps, %d usable' % (nH, sum(1 for v in sgn.values() if v < 0),
                                                                len(dims), len(irr)))

    # group action: sparse untwisted matrices are built LEVEL BY LEVEL inside the sub-block loop
    # (384 elements x 5 levels would not fit in memory at th_max = 6); twisted via reshapes
    Ut = {(th, gi): S.Utw[th][gi] for th in levels for gi in Hc}
    rng = np.random.default_rng(23)
    Uu_th = {}          # filled per level: {gi: sparse U_u(gi) at the current level}

    def proj_apply(r, th, X, side):
        acc = np.zeros_like(X)
        for k, gi in enumerate(Hc):
            w = dims[r] / nH * np.conj(chars[r, elem_class[k]])
            if abs(w) < 1e-15:
                continue
            if side == 'u':
                acc += w * (Uu_th[gi] @ X)
            else:
                acc += w * sgn[gi] * tw_apply('both', Ut[(th, gi)], X, S.nW[th])
        return acc

    def range_basis(r, th, side):
        n = du[th] if side == 'u' else dt[th]
        tr = 0.0
        for k, gi in enumerate(Hc):
            w = dims[r] / nH * np.conj(chars[r, elem_class[k]])
            if side == 'u':
                tr += w * complex(Uu_th[gi].diagonal().sum())
            else:
                tr += w * sgn[gi] * complex(np.trace(Ut[(th, gi)])) ** 2
        m = int(np.rint(tr.real))
        if m <= 0:
            return np.zeros((n, 0), complex)
        X = rng.standard_normal((n, m + 8)) + 1j * rng.standard_normal((n, m + 8))
        Y = proj_apply(r, th, X, side)
        Q, R = np.linalg.qr(Y)
        keep = np.abs(np.diag(R)) > 1e-8 * np.abs(R[0, 0])
        Q = Q[:, keep]
        assert Q.shape[1] == m, (r, th, side, Q.shape[1], m)
        return Q

    # extra Z_2 candidates, level by level; classification on the level-diagonal blocks
    cand = {}
    for nm, gi in (('Z_R', g.i_z), ('F_R', g.i_zF)):
        cand[nm] = ({th: right_only_sparse(S, th, gi) for th in levels},
                    {th: ('right', S.Utw[th][gi]) for th in levels})
    cand['Pi'] = ({th: pi_untw_sparse(S, th) for th in levels}, {th: ('pi', None) for th in levels})
    syms = {}
    for nm, (Xu, Xt) in cand.items():
        cp = cm = 0.0
        for th in levels:
            Mth = M[toff[th]:toff[th] + dt[th], uoff[th]:uoff[th] + du[th]]
            x = rng.standard_normal((du[th], 3)) + 1j * rng.standard_normal((du[th], 3))
            kind, U = Xt[th]
            lhs = tw_apply(kind, U, Mth @ x, S.nW[th])
            rhs = Mth @ (Xu[th] @ x)
            nrm = max(float(np.max(np.abs(rhs))), 1e-30)
            cp = max(cp, float(np.max(np.abs(lhs - rhs))) / nrm)
            cm = max(cm, float(np.max(np.abs(lhs + rhs))) / nrm)
        if cp < 1e-9:
            syms[nm] = (Xu, Xt, +1)
        elif cm < 1e-9:
            syms[nm] = (Xu, Xt, -1)
        log('    %s: commutes %.1e, anticommutes %.1e -> %s' % (nm, cp, cm, ('symmetry (x G)' if cm < 1e-9 else 'symmetry') if nm in syms else 'not a symmetry'))
    names = sorted(syms)
    coef = rng.standard_normal(len(names))

    def split_level(Q, th, side):
        """labels of the columns of Q (a level-th isotypic basis) under the extra Z_2's."""
        if not names or Q.shape[1] == 0:
            return [()] * Q.shape[1], Q
        Ys = []
        for nm in names:
            Xu, Xt, sg = syms[nm]
            if side == 'u':
                Y = Q.conj().T @ (Xu[th] @ Q)
            else:
                kind, U = Xt[th]
                Y = sg * (Q.conj().T @ tw_apply(kind, U, Q, S.nW[th]))
            Ys.append((Y + Y.conj().T) / 2)
        Z = sum(cf * Y for cf, Y in zip(coef, Ys))
        w, V = np.linalg.eigh(Z)
        labs = []
        for i in range(V.shape[1]):
            v = V[:, i]
            lab = []
            for Y in Ys:
                e = float(np.real(np.conj(v) @ (Y @ v)))
                assert abs(abs(e) - 1) < 1e-6, (nm, e)
                lab.append(int(np.sign(e)))
            labs.append(tuple(lab))
        return labs, Q @ V

    # sub-blocks (levels outermost so that the sparse group matrices of one level are cached)
    t0 = time.time()
    subs = {}
    for th in levels:
        Uu_th.clear()
        for gi in Hc:
            Uu_th[gi] = U_untw_sparse(S, th, gi)
        for r in irr:
            for side in ('u', 't'):
                Q = range_basis(r, th, side)
                if Q.shape[1] == 0:
                    continue
                labs, Qs = split_level(Q, th, side)
                for lab in set(labs):
                    cols = [i for i, l in enumerate(labs) if l == lab]
                    full = np.zeros((DU if side == 'u' else DT, len(cols)), complex)
                    o = uoff[th] if side == 'u' else toff[th]
                    full[o:o + Q.shape[0]] = Qs[:, cols]
                    subs.setdefault((r, lab), {}).setdefault(side, []).append((th, full))
    blocks = []
    for key in sorted(subs):
        d = subs[key]
        qu = d.get('u', [])
        qt = d.get('t', [])
        if not qu or not qt:
            continue
        Qu = np.concatenate([x[1] for x in qu], axis=1)
        Qt = np.concatenate([x[1] for x in qt], axis=1)
        Eu = np.concatenate([np.full(x[1].shape[1], float(x[0])) for x in qu])
        Et = np.concatenate([np.full(x[1].shape[1], float(x[0])) for x in qt])
        Ms = Qt.conj().T @ (M @ Qu)
        blocks.append(dict(key=key, Eu=Eu, Et=Et, M=Ms))
    log('%d sub-blocks (irreps x %s) built in %.1f s; isotypic sizes (u,t) %s' % (
        len(blocks), names, time.time() - t0, [(len(b['Eu']), len(b['Et'])) for b in blocks]))
    log('    multiplicity-space total (states each once): %d' % sum(
        (len(b['Eu']) + len(b['Et'])) // int(dims[b['key'][0]]) for b in blocks))
    del M
    # save matrices
    npz = {}
    for k, b in enumerate(blocks):
        npz['E0_%d' % k] = np.concatenate([b['Eu'], b['Et']])
        npz['M_%d' % k] = b['M']
        npz['nu_%d' % k] = np.array([len(b['Eu'])])
    npz['keys'] = np.array([str(b['key']) for b in blocks])
    np.savez_compressed(os.path.join(OUT, 'Generic_sweep_%s_L%d_matrices.npz' % (direction, th_max)), **npz)

    # second-order-vs-first-order measure (window states, untwisted side) : <Delta_2>/<Delta_1> = s * lam
    E0w, half = g11.window(th_max)
    num = den = 0.0
    cnt = 0
    for b in blocks:
        Eu, Et, Ms = b['Eu'], b['Et'], b['M']
        for j in range(len(Eu)):
            if abs(Eu[j] - E0w) > half:
                continue
            col = np.abs(Ms[:, j]) ** 2
            same = Et == Eu[j]
            d1 = np.sqrt(np.sum(col[same]))
            d2 = np.sum(col[~same] / np.abs(Eu[j] - Et[~same]))
            if d1 > 1e-12:
                num += d2
                den += d1
                cnt += 1
    slope = num / den if den else float('nan')
    log('window [%g +- %g]: <Delta_2>/<Delta_1> = %.3f lam over %d untwisted window states -> unity at lam = %.3f'
        % (E0w, half, slope, cnt, 1 / slope if slope else float('nan')))

    # diagonalise
    rows = []
    for lam in lams:
        t1 = time.time()
        sp_list = []
        nunl = 0
        for b in blocks:
            nu, nt = len(b['Eu']), len(b['Et'])
            H = np.zeros((nu + nt, nu + nt), complex)
            H[np.arange(nu + nt), np.arange(nu + nt)] = np.concatenate([b['Eu'], b['Et']])
            if lam:
                H[nu:, :nu] = lam * b['M']
                H[:nu, nu:] = lam * b['M'].conj().T
            ev = np.linalg.eigvalsh(H)
            # the isotypic component carries every level dim(rho) times: collapse the exact
            # degeneracy to the multiplicity-space spectrum (the paper's sub-sector spectrum)
            ev = np.unique(np.round(ev, 8))
            m = g11.unlifted(ev, th_max)
            nunl += int(np.sum(m))
            sp_list.append(np.sort(ev[~m]))
        res = st.analyse(sp_list, E0w, half, Ls=LS, seed=int(1000 * lam) + 1)
        row = dict(lam=float(lam), r_raw=res['r_mean'], r_unf=res['ru_mean'], r_sem=res['ru_sem'],
                   f_cal=res['f_cal'], sigma2=[float(x) for x in res['sigma2']], n_win=res['n_win_tot'],
                   n_unlifted=nunl, seconds=time.time() - t1)
        rows.append(row)
        log('  lam %.3f : <r> raw %.3f unf %.3f +- %.3f | f_cal %.2f | Sigma2(L=1,2,3,5,8) %s | window %d, unlifted %d  [%.0f s]'
            % (lam, row['r_raw'], row['r_unf'], row['r_sem'], row['f_cal'],
               ' '.join('%.2f' % x for x in row['sigma2']), row['n_win'], nunl, row['seconds']))
    out = dict(direction=direction, th_max=th_max, physical=bool(gp.is_physical(c)), H_order=nH,
               extra_Z2=names, n_blocks=len(blocks), sizes=[(len(b['Eu']), len(b['Et'])) for b in blocks],
               window=(E0w, half), delta_slope=slope, rows=rows, seconds=time.time() - T0)
    with open(os.path.join(OUT, 'Generic_sweep_%s_L%d.json' % (direction, th_max)), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(os.path.join(OUT, 'Generic_sweep.md'), 'a') as fh:
        fh.write('\n'.join(lines) + '\n\n')
    log('[total %.0f s]' % (time.time() - T0))
    return out


if __name__ == '__main__':
    direction = sys.argv[1]
    th_max = int(sys.argv[2])
    lams = tuple(float(x) for x in sys.argv[3].split(',')) if len(sys.argv) > 3 else DEFAULT_LAMS
    run(direction, th_max, lams)
