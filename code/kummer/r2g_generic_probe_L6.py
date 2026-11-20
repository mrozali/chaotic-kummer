"""r2g_generic_probe_L6.py -- the generic-direction first-order probe at E = 6 (th = 6), where the raw
untwisted space (20644 states) is too large for dense group matrices.

Same method as r2g_generic_probe.py, with three changes:
  * U_u(g) on the raw untwisted space is built as a scipy.sparse block matrix (kron blocks per sector);
  * U_t(g) = U (x) U on the twisted space is applied through reshapes, never formed;
  * isotypic bases come from the randomised range finder applied to P_sigma X with X random,
    the dimension m_sigma = Tr P_sigma computed from traces first.
Intended for directions with a SMALL stabiliser (generic unit quaternions: order 16, 1-dim irreps),
where the cost is |H_c| sparse products per column block.  For the singlet (order 384, irreps up to
dimension 6) use the paper's multiplicity-space pipeline instead.

Usage: python r2g_generic_probe_L6.py <direction> [th]      (th defaults to 6)
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sps

import f2_common as f2
import r2c_space as sp
import r2g_generic_probe as gp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
R_GOE, R_GUE, R_POI = 0.5307, 0.5996, 0.3863


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


def apply_tw(U, X, nW):
    """(U (x) U) X for X of shape (nW*nW, k), as two BLAS matmuls (the einsum form was 100x slower
    at nW = 168): Z[i,b,k] = sum_a U[i,a] X[a,b,k], then W[i,j,k] = sum_b U[j,b] Z[i,b,k]."""
    k = X.shape[1]
    Z = (U @ X.reshape(nW, nW * k)).reshape(nW, nW, k)
    W = np.matmul(U[None, :, :], Z)
    return W.reshape(nW * nW, k)


def apply_tw_right(U, X, nW):
    """(1 (x) U) X for X of shape (nW*nW, k): W[i,j,k] = sum_b U[j,b] X[i,b,k]."""
    k = X.shape[1]
    return np.matmul(U[None, :, :], X.reshape(nW, nW, k)).reshape(nW * nW, k)


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


def run(direction, th=6):
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    c = gp.cmat(direction)
    log('# Generic-direction probe at E = %d: %s  (%s)' % (th, direction, time.strftime('%Y-%m-%d %H:%M')))
    log('PHYSICAL: %s' % gp.is_physical(c))
    t0 = time.time()
    S = sp.Space(th, mode='prim', verbose=False)
    g = S.g
    log('Space(prim, %d) built in %.1f s' % (th, time.time() - t0))
    Hc, sgn, rule = gp.stabiliser(S, c, 3)
    log('stabiliser order %d (%d with G)' % (len(Hc), sum(1 for v in sgn.values() if v < 0)))
    chars, dims, elem_class, phys, ncls = gp.subgroup_characters(g, Hc)
    irr = [r for r in range(len(dims)) if phys[r]]
    log('irreps %d (dims %s), usable %d' % (len(dims), sorted(dims.tolist()), len(irr)))

    t1 = time.time()
    M = gp.M_c_raw(S, th, c)
    dt, du = M.shape
    nW = S.nW[th]
    gmax = float(np.max(np.abs(M)))
    log('raw M %d x %d built in %.1f s (%.1f GB)' % (dt, du, time.time() - t1, M.nbytes / 1e9))

    # sparse group matrices on the untwisted side, twisted via reshapes
    t1 = time.time()
    Uu = {gi: U_untw_sparse(S, th, gi) for gi in Hc}
    log('sparse U_u for %d elements in %.1f s, nnz %d' % (len(Hc), time.time() - t1,
                                                          sum(U.nnz for U in Uu.values())))
    Ut = {gi: S.Utw[th][gi] for gi in Hc}
    trU = {gi: complex(Uu[gi].diagonal().sum()) for gi in Hc}
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
                acc += w * (Uu[gi] @ X)
            else:
                acc += w * sgn[gi] * apply_tw(Ut[gi], X, nW)
        return acc

    def range_basis(r, side):
        n = du if side == 'u' else dt
        tr = sum(dims[r] / nH * np.conj(chars[r, elem_class[k]]) * (trU[gi] if side == 'u' else trT[gi])
                 for k, gi in enumerate(Hc))
        m = int(np.rint(tr.real))
        if m <= 0:
            return np.zeros((n, 0), complex)
        X = rng.standard_normal((n, m + 8)) + 1j * rng.standard_normal((n, m + 8))
        Y = proj_apply(r, X, side)
        Q, R = np.linalg.qr(Y)
        keep = np.abs(np.diag(R)) > 1e-8 * np.abs(R[0, 0])
        Q = Q[:, keep]
        assert Q.shape[1] == m, (r, side, Q.shape[1], m)
        # idempotence check on a few columns
        chk = proj_apply(r, Q[:, :4], side)
        assert np.max(np.abs(chk - Q[:, :4])) < 1e-8
        return Q

    # extra Z_2's: Z_R and F_R (right-only central elements); Pi only if the direction allows it
    ops = {}
    for nm, gi in (('Z_R', g.i_z), ('F_R', g.i_zF)):
        Xu = right_only_sparse(S, th, gi)
        Xt_mat = S.Utw[th][gi]
        ops[nm] = (Xu, Xt_mat)
    syms = {}
    for nm, (Xu, Xt_mat) in ops.items():
        # test on a few random untwisted vectors: Xt M x  vs  M Xu x
        x = rng.standard_normal((du, 3)) + 1j * rng.standard_normal((du, 3))
        lhs = apply_tw_right(Xt_mat, M @ x, nW)   # (1 (x) Xt) M x
        rhs = M @ (Xu @ x)
        cp = float(np.max(np.abs(lhs - rhs))) / max(float(np.max(np.abs(rhs))), 1e-30)
        cm = float(np.max(np.abs(lhs + rhs))) / max(float(np.max(np.abs(rhs))), 1e-30)
        if cp < 1e-9:
            syms[nm] = (Xu, Xt_mat, +1)
        elif cm < 1e-9:
            syms[nm] = (Xu, Xt_mat, -1)
        log('    %s: commutes %.1e anticommutes %.1e -> %s' % (nm, cp, cm,
                                                            'sym' if nm in syms else 'not a symmetry'))
    names = sorted(syms)

    gaps, rows = [], []
    n_sub = n_dist = 0
    tot_rank = tot_dp = tot_dm = 0
    for r in irr:
        t2 = time.time()
        Qu = range_basis(r, 'u')
        Qt = range_basis(r, 't')
        if Qu.shape[1] == 0 or Qt.shape[1] == 0:
            continue
        d = int(dims[r])
        MQ = M @ Qu                                   # (dt, mu)
        Mr = Qt.conj().T @ MQ                         # (mt, mu)
        # joint split by the extra Z_2's inside the component
        labs_u = [()] * Qu.shape[1]
        if names:
            coef = rng.standard_normal(len(names))
            Yu = [Qu.conj().T @ (syms[nm][0] @ Qu) for nm in names]
            Yt = []
            for nm in names:
                Xt_mat, sg = syms[nm][1], syms[nm][2]
                T = apply_tw_right(Xt_mat, Qt, nW)
                Yt.append(sg * (Qt.conj().T @ T))
            def split(Ys, Q):
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
            Vu, lu = split(Yu, Qu)
            Vt, lt = split(Yt, Qt)
            parts = {}
            for lab in set(lu) | set(lt):
                cu = [i for i, l in enumerate(lu) if l == lab]
                ct = [i for i, l in enumerate(lt) if l == lab]
                if cu and ct:
                    parts[lab] = (Vu[:, cu], Vt[:, ct])
        else:
            parts = {(): (np.eye(Qu.shape[1]), np.eye(Qt.shape[1]))}
        sv_all = np.linalg.svd(Mr, compute_uv=False)
        nz_all = sv_all[sv_all > 1e-9 * gmax]
        dp, dm = Qu.shape[1] // d, Qt.shape[1] // d
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
            if len(dist) >= 20:
                gaps.append(f2.gap_ratios(dist))
        log('    irrep %d (dim %d): mult dims %d/%d, rank %g, %d sub-blocks  [%.1f s]'
            % (r, d, dp, dm, rank, len(parts), time.time() - t2))
    gaps = np.concatenate(gaps) if gaps else np.zeros(0)
    rmean = float(np.mean(gaps)) if len(gaps) else float('nan')
    rsem = float(np.std(gaps) / np.sqrt(len(gaps))) if len(gaps) else float('nan')
    log('E = %d %s: multiplicity space d_+ %d, d_- %d, rank %g, zero modes %g (fraction %.2f)'
        % (th, direction, tot_dp, tot_dm, tot_rank, tot_dp + tot_dm - 2 * tot_rank,
           (tot_dp + tot_dm - 2 * tot_rank) / max(tot_dp + tot_dm, 1)))
    log('    resolved sub-blocks %d (irreps x %s), distinct values %d, gap ratios %d : <r> = %.3f +- %.3f'
        '   [GOE %.3f, GUE %.3f, Poisson %.3f]' % (n_sub, names, n_dist, len(gaps), rmean, rsem,
                                                    R_GOE, R_GUE, R_POI))
    for x in rows:
        log('    | %d | %s | %d | %d | %d | %d |' % (x['rho'], x['lab'], x['nu'], x['nt'], x['rank'], x['distinct']))
    res = dict(direction=direction, th=th, H_order=len(Hc), d_plus=tot_dp, d_minus=tot_dm, rank=tot_rank,
               n_sub=n_sub, n_distinct=n_dist, n_gaps=int(len(gaps)), r_mean=rmean, r_sem=rsem,
               extra=names, rows=rows, seconds=time.time() - t0)
    with open(os.path.join(OUT, 'Generic_probe_%s_E%d.json' % (direction, th)), 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    with open(os.path.join(OUT, 'Generic_probe.md'), 'a') as fh:
        fh.write('\n'.join(lines) + '\n\n')
    log('[total %.1f s]' % (time.time() - t0))


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
