"""r2g_generic_probe_mf_lean.py -- the matrix-free first-order probe of r2g_generic_probe_mf.py with the
memory of E = 8 brought down from ~230 GB to ~50 GB per irrep, so that the levels E = 8 run as one modest
Slurm task per irrep.

Same object, same resolution: the first-order matrix M_c at level th along direction c, resolved by the
irreps r of the stabiliser H_c and the joint label of the right-only symmetries (F_R, Z_R).  The change is
in how the (r, label) sub-block is reached.  The old code built orthonormal bases of the whole irrep-r
space on BOTH sides (19221 + 19956 columns at E = 8), split each by the labels with two dense
diagonalisations, and took the SVD of the twisted-by-untwisted block.  Here only the untwisted side is
projected, and directly onto (r, label):

    P = P_r (1 + f F_R)/2 (1 + z Z_R)/2,   P_r = (d_r/|H|) sum_g chi_r(g)^* U(g),

applied to a Gaussian block with exactly m = tr P columns (traces sector by sector, below), orthonormalised
by CholQR3 in place.  Because M_c intertwines the stabiliser and the right-only symmetries (with the sign of
each, as in the old code), M_c maps the untwisted (r, label) space into the twisted (r, label) space, so the
singular values of M_c Q ARE those of the sub-block; no twisted basis is needed.  Labels whose twisted
trace vanishes carry no coupling and are skipped (their states are first-order zero modes).

Output: per (r, label), ALL singular values (zeros included), the rank and the distinct values (rounded to
1e-9 as in the old code, same npz keys 'rho<r>_lab<..>'), gap ratios on sub-blocks with >= 20 distinct
values.  --ref=<old values npz> compares the distinct values with an earlier run and exits non-zero on a
mismatch (the validation gate).  --irreps=0,2 restricts the irreps (one Slurm task per irrep at E = 8);
--merge pools the per-irrep files of a finished run.

Usage: python r2g_generic_probe_mf_lean.py <direction> <th> [--irreps=a,b] [--validate] [--chunk=N]
                                            [--ref=path] [--tag=name]
       python r2g_generic_probe_mf_lean.py <direction> <th> --merge [--tag=name]
"""
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import scipy.linalg as sl

import f2_common as f2
import r2c_space as sp
import r2g_generic_probe as gp
import r2g_generic_probe_L6 as pl6
import r2g_generic_probe_mf as mf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.environ.get('KUMMER_OUT') or os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'r2_out')))
R_GOE, R_GUE, R_POI = 0.5307, 0.5996, 0.3863


class LeanAction(mf.SectorAction):
    """SectorAction plus the trace of U(g) composed with right-only actions R(h_1) ... R(h_n)."""

    def trace_with_right(self, gi, rights):
        mp = self.maps(gi)
        mrs = [self.maps(h) for h in rights]
        tr = 0.0 + 0.0j
        for (a, b) in self.sect:
            bb, Mr = b, None
            for mh in reversed(mrs):                     # the rightmost R(h) acts first
                b2, Mh = mh[bb]
                Mr = Mh if Mr is None else Mh @ Mr
                bb = b2
            bp, Mb = mp[bb]
            ap, Ma = mp[a]
            if (ap, bp) != (a, b):
                continue
            tr += np.trace(Ma) * np.trace(Mb if Mr is None else Mb @ Mr)
        return complex(tr)


def validate_traces(S, th, act, elems, rights, log):
    """trace_with_right against the sparse matrices at a small level."""
    worst = 0.0
    R = {h: pl6.right_only_sparse(S, th, h) for h in rights}
    for gi in elems:
        U = pl6.U_untw_sparse(S, th, gi)
        for k in range(len(rights) + 1):
            for hs in itertools.combinations(rights, k):
                A = U
                for h in hs:
                    A = A @ R[h]
                worst = max(worst, abs(complex(A.diagonal().sum()) - act.trace_with_right(gi, list(hs))))
    log('  composite traces validated against the sparse matrices at th=%d: max deviation %.2e' % (th, worst))
    assert worst < 1e-8, worst


def cholqr_inplace(Y, passes=3, rows=8192):
    """Y (n x m, C order) -> orthonormal basis of its range, in place: CholQR repeated `passes` times, the
    Gram matrix and the triangular update done in row blocks so that no second n x m array is allocated.
    Returns the final orthonormality error max|Y^+ Y - 1|."""
    n, m = Y.shape

    def gram():
        G = np.zeros((m, m), complex)
        for i0 in range(0, n, rows):
            B = Y[i0:i0 + rows]
            G += B.conj().T @ B
        return (G + G.conj().T) / 2

    for it in range(passes):
        G = gram()
        L = np.linalg.cholesky(G)
        del G
        Linv = sl.solve_triangular(L, np.eye(m, dtype=complex), lower=True, check_finite=False)
        del L
        LH = np.ascontiguousarray(Linv.conj().T)
        del Linv
        for i0 in range(0, n, rows):
            Y[i0:i0 + rows] = Y[i0:i0 + rows] @ LH
        del LH
    G = gram()
    G[np.diag_indices(m)] -= 1.0
    return float(np.max(np.abs(G)))


def labkey(r, lab):
    return 'rho%d_lab%s' % (r, ''.join('p' if l > 0 else 'm' for l in lab))


def run(direction, th, irreps=None, validate=False, chunk=400, ref=None, tag=''):
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    c = gp.cmat(direction)
    log('# Generic-direction probe (matrix-free, lean) at E = %d: %s  (%s)'
        % (th, direction, time.strftime('%Y-%m-%d %H:%M')))
    log('PHYSICAL: %s' % gp.is_physical(c))
    t0 = time.time()
    S = sp.Space(th, mode='prim', verbose=False)
    g = S.g
    log('Space(prim, %d) built in %.1f s' % (th, time.time() - t0))
    if validate:
        mf.validate_mf(S, min(th, 5), c, log)
        if th >= 6:
            mf.validate_mf(S, 6, c, log)
    Hc, sgn, rule = gp.stabiliser(S, c, 3)
    log('stabiliser order %d (%d with G)' % (len(Hc), sum(1 for v in sgn.values() if v < 0)))
    chars, dims, elem_class, phys, ncls = gp.subgroup_characters(g, Hc)
    irr = [r for r in range(len(dims)) if phys[r]]
    log('irreps %d (dims %s), usable %s' % (len(dims), sorted(dims.tolist()), irr))
    if irreps is not None:
        irr = [r for r in irr if r in irreps]
        log('restricted to irreps %s' % irr)

    M = mf.MFree(S, th, c, chunk=chunk)
    dt, du = M.shape
    nW = S.nW[th]
    log('matrix-free M %d x %d (nV %d, %d sectors, %d (Ad,Bd) terms)' % (dt, du, M.nV, len(M.sect), len(M.terms)))
    act = LeanAction(S, th)
    if validate:
        mf.validate_action(S, th, act, list(Hc)[:4], log)
        S6 = S if th <= 6 else sp.Space(6, mode='prim', verbose=False)
        validate_traces(S6, min(th, 6), LeanAction(S6, min(th, 6)), list(Hc)[:4], [g.i_z, g.i_zF], log)
    Ut = {gi: S.Utw[th][gi] for gi in Hc}
    nH = len(Hc)
    rng = np.random.default_rng(11)

    # extra Z_2's: Z_R and F_R (right-only central elements), commuting (+1) or anticommuting (-1) with M
    syms = {}
    for nm, gi in (('Z_R', g.i_z), ('F_R', g.i_zF)):
        Xt_mat = S.Utw[th][gi]
        x = rng.standard_normal((du, 3)) + 1j * rng.standard_normal((du, 3))
        lhs = pl6.apply_tw_right(Xt_mat, M @ x, nW)
        rhs = M @ act.apply(gi, x, right_only=True)
        cp = float(np.max(np.abs(lhs - rhs))) / max(float(np.max(np.abs(rhs))), 1e-30)
        cm = float(np.max(np.abs(lhs + rhs))) / max(float(np.max(np.abs(rhs))), 1e-30)
        if cp < 1e-9:
            syms[nm] = (gi, Xt_mat, +1)
        elif cm < 1e-9:
            syms[nm] = (gi, Xt_mat, -1)
        log('    %s: commutes %.1e anticommutes %.1e -> %s' % (nm, cp, cm, 'sym' if nm in syms else 'not a symmetry'))
    names = sorted(syms)

    x = rng.standard_normal((du, 2)) + 1j * rng.standard_normal((du, 2))
    gmax = float(np.max(np.abs(M @ x))) / np.sqrt(du)
    log('    rank threshold reference %.3e' % gmax)

    def weights(r):
        return [dims[r] / nH * np.conj(chars[r, elem_class[k]]) for k in range(nH)]

    tr_cache = {}

    def tr_u(gi, hs):
        if (gi, hs) not in tr_cache:
            tr_cache[(gi, hs)] = act.trace_with_right(gi, list(hs))
        return tr_cache[(gi, hs)]

    def dims_lab(r, lab):
        """(nu, nt) = traces of the (r, lab) projector on the untwisted and twisted sides."""
        w = weights(r)
        tu = tt = 0.0 + 0.0j
        for k, gi in enumerate(Hc):
            if abs(w[k]) < 1e-15:
                continue
            trUt = np.trace(Ut[gi])
            for n_sub in range(len(names) + 1):
                for sub in itertools.combinations(range(len(names)), n_sub):
                    cu = np.prod([lab[i] for i in sub]) / 2 ** len(names)
                    ct = np.prod([lab[i] * syms[names[i]][2] for i in sub]) / 2 ** len(names)
                    hs = tuple(syms[names[i]][0] for i in sub)
                    tu += w[k] * cu * tr_u(gi, hs)
                    B = Ut[gi]
                    for i in sub:
                        B = B @ syms[names[i]][1]
                    tt += w[k] * sgn[gi] * ct * trUt * np.trace(B)
        nu, nt = int(np.rint(tu.real)), int(np.rint(tt.real))
        assert abs(tu - nu) < 1e-6 and abs(tt - nt) < 1e-6, (r, lab, tu, tt)
        return nu, nt

    def proj_lab(r, lab, X):
        """P_r prod (1 + l R)/2 applied to X (du x k)."""
        Y = X
        for i, nm in enumerate(names):
            Y = 0.5 * (Y + lab[i] * act.apply(syms[nm][0], Y, right_only=True))
        acc = np.zeros_like(Y)
        for k, wk in enumerate(weights(r)):
            if abs(wk) < 1e-15:
                continue
            act.apply(Hc[k], Y, out=acc, weight=wk)
        return acc

    gaps, rows, dist_store, sv_store = [], [], {}, {}
    tot_dp = tot_dm = tot_rank = 0
    labs = list(itertools.product((+1, -1), repeat=len(names)))
    for r in irr:
        for lab in labs:
            t2 = time.time()
            nu, nt = dims_lab(r, lab)
            tot_dp += nu
            tot_dm += nt
            if nu == 0 or nt == 0:
                log('    irrep %d %s: nu %d nt %d, no coupling' % (r, list(lab), nu, nt))
                continue
            rs = np.random.default_rng(1000 * r + 17 * (1 + sum((1 + l) // 2 * 2 ** i for i, l in enumerate(lab))))
            Q = np.empty((du, nu), complex)
            for k0 in range(0, nu, chunk):
                k1 = min(nu, k0 + chunk)
                X = rs.standard_normal((du, k1 - k0)) + 1j * rs.standard_normal((du, k1 - k0))
                Q[:, k0:k1] = proj_lab(r, lab, X)
            t3 = time.time()
            err = cholqr_inplace(Q)
            assert err < 1e-11, err
            chk = proj_lab(r, lab, np.ascontiguousarray(Q[:, :4]))
            pdev = float(np.max(np.abs(chk - Q[:, :4])))
            assert pdev < 1e-8, pdev
            t4 = time.time()
            Z = np.empty((dt, nu), complex, order='F')
            for k0 in range(0, nu, chunk):
                k1 = min(nu, k0 + chunk)
                Z[:, k0:k1] = M.matvec(np.ascontiguousarray(Q[:, k0:k1]))
            del Q
            t5 = time.time()
            s = sl.svd(Z, compute_uv=False, overwrite_a=True, check_finite=False)
            del Z
            t6 = time.time()
            nz = s > 1e-9 * gmax
            rank = int(nz.sum())
            dist = np.unique(np.round(s[nz], 9))
            key = labkey(r, lab)
            sv_store[key] = s
            dist_store[key] = dist
            tot_rank += rank
            gr = f2.gap_ratios(dist) if len(dist) >= 20 else np.zeros(0)
            if len(gr):
                gaps.append(gr)
            rows.append(dict(rho=int(r), lab=list(lab), nu=int(nu), nt=int(nt), rank=rank, distinct=int(len(dist)),
                             n_gaps=int(len(gr)), r_mean=float(gr.mean()) if len(gr) else float('nan'),
                             r_sem=float(gr.std() / np.sqrt(len(gr))) if len(gr) else float('nan'),
                             smallest_kept=float(s[nz].min()) if rank else float('nan'),
                             largest_dropped=float(s[~nz].max()) if rank < len(s) else 0.0))
            log('    irrep %d %s: nu %d nt %d, rank %d, distinct %d, <r> %.4f +- %.4f on %d  '
                '[orth %.1e, proj %.1e; basis %.0f s, CholQR %.0f s, M Q %.0f s, SVD %.0f s]'
                % (r, list(lab), nu, nt, rank, len(dist), rows[-1]['r_mean'], rows[-1]['r_sem'], len(gr),
                   err, pdev, t3 - t2, t4 - t3, t5 - t4, t6 - t5))
            log('      smallest kept %.3e, largest dropped %.3e (threshold %.3e)'
                % (rows[-1]['smallest_kept'], rows[-1]['largest_dropped'], 1e-9 * gmax))
    gaps = np.concatenate(gaps) if gaps else np.zeros(0)
    rmean = float(np.mean(gaps)) if len(gaps) else float('nan')
    rsem = float(np.std(gaps) / np.sqrt(len(gaps))) if len(gaps) else float('nan')
    log('E = %d %s (irreps %s): d_+ %d, d_- %d, rank %d; %d sub-blocks, gap ratios %d : <r> = %.4f +- %.4f'
        '   [GOE %.3f, GUE %.3f, Poisson %.3f]'
        % (th, direction, irr, tot_dp, tot_dm, tot_rank, len(rows), len(gaps), rmean, rsem, R_GOE, R_GUE, R_POI))

    ok = True
    if ref:
        R = np.load(ref)
        for x in rows:
            key = labkey(x['rho'], x['lab'])
            if key not in R.files:
                log('  REF: %s not in %s' % (key, os.path.basename(ref)))
                ok = False
                continue
            a, b = dist_store[key], np.asarray(R[key])
            if len(a) != len(b):
                log('  REF: %s length %d vs %d  FAIL' % (key, len(a), len(b)))
                ok = False
                continue
            dev = float(np.max(np.abs(a - b)))
            log('  REF: %s %d values, max |new - old| = %.2e  %s' % (key, len(a), dev, 'PASS' if dev < 2e-9 else 'FAIL'))
            ok = ok and dev < 2e-9
        log('REFERENCE CHECK %s' % ('PASS' if ok else 'FAIL'))

    sfx = tag + ('_rho' + '_'.join(str(r) for r in irr) if irreps is not None else '')
    base = 'Generic_probe_%s_E%d_lean%s' % (direction, th, sfx)
    res = dict(direction=direction, th=th, H_order=len(Hc), irreps=irr, d_plus=tot_dp, d_minus=tot_dm, rank=tot_rank,
               n_sub=len(rows), n_gaps=int(len(gaps)), r_mean=rmean, r_sem=rsem, extra=names, rows=rows,
               gmax=gmax, rank_threshold=1e-9 * gmax, ref=ref, ref_ok=ok if ref else None,
               seconds=time.time() - t0, method='matrix-free lean (untwisted (r, label) projection)')
    with open(os.path.join(OUT, base + '.json'), 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, base + '_values.npz'),
                        **dist_store, **{'sv_' + k: v for k, v in sv_store.items()})
    with open(os.path.join(OUT, 'Generic_probe.md'), 'a') as fh:
        fh.write('\n'.join(lines) + '\n\n')
    log('[total %.1f s]' % (time.time() - t0))
    return ok


def merge(direction, th, tag=''):
    """Pool the per-irrep files Generic_probe_<dir>_E<th>_lean<tag>_rho<r>.{json,_values.npz}."""
    fs = sorted(glob.glob(os.path.join(OUT, 'Generic_probe_%s_E%d_lean%s_rho*.json' % (direction, th, tag))))
    rows, dist, sv, parts = [], {}, {}, []
    for f in fs:
        d = json.load(open(f))
        parts.append(dict(file=os.path.basename(f), irreps=d['irreps'], seconds=d['seconds'], ref_ok=d['ref_ok']))
        rows += d['rows']
        z = np.load(f[:-5] + '_values.npz')
        for k in z.files:
            (sv if k.startswith('sv_') else dist)[k] = z[k]
    rows.sort(key=lambda x: (x['rho'], x['lab']))

    def pool(keep):
        g = [f2.gap_ratios(dist[labkey(x['rho'], x['lab'])]) for x in rows
             if keep(x) and x['distinct'] >= 20]
        g = np.concatenate(g) if g else np.zeros(0)
        return dict(n_gaps=int(len(g)), r_mean=float(g.mean()) if len(g) else float('nan'),
                    r_sem=float(g.std() / np.sqrt(len(g))) if len(g) else float('nan'))

    res = dict(direction=direction, th=th, parts=parts, rows=rows,
               rank=int(sum(x['rank'] for x in rows)), n_sub=len(rows),
               pooled_all=pool(lambda x: True),
               pooled_without_trivial_irrep=pool(lambda x: x['rho'] != 0),
               note=('at even E the trivial-irrep sub-block carries the hidden Z_2 of the even levels, unresolved here; '
                     'the quotable value pools the other irreps (as for qgen at E = 8)'))
    base = 'Generic_probe_%s_E%d_lean%s' % (direction, th, tag)
    with open(os.path.join(OUT, base + '.json'), 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, base + '_values.npz'), **dist, **sv)
    print(json.dumps({k: res[k] for k in ('n_sub', 'rank', 'pooled_all', 'pooled_without_trivial_irrep')}, indent=1))
    for x in rows:
        print('  | %d | %s | %d | %d | %d | %d | %.4f +- %.4f |' % (x['rho'], x['lab'], x['nu'], x['nt'], x['rank'],
                                                                 x['distinct'], x['r_mean'], x['r_sem']))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) if '=' in a else (a[2:], '1') for a in sys.argv[1:] if a.startswith('--'))
    direction, th = args[0], int(args[1])
    tag = opts.get('tag', '')
    if 'merge' in opts:
        merge(direction, th, tag=tag)
        sys.exit(0)
    irreps = [int(x) for x in opts['irreps'].split(',')] if 'irreps' in opts else None
    ok = run(direction, th, irreps=irreps, validate='validate' in opts, chunk=int(opts.get('chunk', 400)),
             ref=opts.get('ref'), tag=tag)
    sys.exit(0 if ok else 3)
